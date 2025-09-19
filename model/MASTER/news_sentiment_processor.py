"""
News Sentiment Processor for MASTER Model Integration
Uses FinBERT for financial sentiment analysis of news articles
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from collections import defaultdict
import re
import warnings
from scipy.stats import spearmanr

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsSentimentProcessor:
    """
    Processes news articles and extracts sentiment scores using FinBERT
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize the sentiment processor with FinBERT model
        
        Args:
            model_name: Hugging Face model name for FinBERT
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load FinBERT model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded FinBERT model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess news text for sentiment analysis
        
        Args:
            text: Raw news text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        return text.strip()
    
    def get_sentiment_score(self, text: str) -> Dict[str, float]:
        """
        Get sentiment score for a single text using FinBERT
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores and label
        """
        if not text or text.strip() == '':
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'label': 'neutral', 'score': 0.0}
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'label': 'neutral', 'score': 0.0}
            
            # Tokenize and encode
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)
            
            # Extract probabilities
            probs = probabilities.cpu().numpy()[0]
            
            # FinBERT labels: 0=positive, 1=negative, 2=neutral
            sentiment_scores = {
                'positive': float(probs[0]),
                'negative': float(probs[1]), 
                'neutral': float(probs[2])
            }
            
            # Determine label and overall score
            label_idx = np.argmax(probs)
            labels = ['positive', 'negative', 'neutral']
            label = labels[label_idx]
            
            # Create weighted score: positive=1, neutral=0, negative=-1
            score = sentiment_scores['positive'] - sentiment_scores['negative']
            
            sentiment_scores['label'] = label
            sentiment_scores['score'] = score
            
            return sentiment_scores
            
        except Exception as e:
            logger.error(f"Error processing sentiment for text: {e}")
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'label': 'neutral', 'score': 0.0}
    
    def process_news_batch(self, news_df: pd.DataFrame, 
                          text_columns: List[str] = ['title', 'description']) -> pd.DataFrame:
        """
        Process a batch of news articles for sentiment analysis
        
        Args:
            news_df: DataFrame with news articles
            text_columns: Columns containing text to analyze
            
        Returns:
            DataFrame with sentiment scores added
        """
        if news_df.empty:
            return news_df
        
        # Combine text columns
        news_df = news_df.copy()
        news_df['combined_text'] = ''
        
        for col in text_columns:
            if col in news_df.columns:
                news_df['combined_text'] += ' ' + news_df[col].fillna('').astype(str)
        
        # Process sentiment for each article
        sentiment_results = []
        
        for idx, row in news_df.iterrows():
            text = row['combined_text']
            sentiment = self.get_sentiment_score(text)
            sentiment_results.append(sentiment)
        
        # Add sentiment columns to DataFrame
        sentiment_df = pd.DataFrame(sentiment_results)
        result_df = pd.concat([news_df, sentiment_df], axis=1)
        
        return result_df
    
    def aggregate_sentiment_by_time(self, news_df: pd.DataFrame, 
                                   time_column: str = 'published_utc',
                                   ticker_column: str = 'ticker',
                                   freq: str = 'D') -> pd.DataFrame:
        """
        Aggregate sentiment scores by time period and ticker
        
        Args:
            news_df: DataFrame with news articles and sentiment scores
            time_column: Column containing timestamps
            ticker_column: Column containing ticker symbols
            freq: Time frequency for aggregation ('D' for daily, 'H' for hourly)
            
        Returns:
            DataFrame with aggregated sentiment by time and ticker
        """
        if news_df.empty:
            return pd.DataFrame()
        
        # Ensure datetime column
        news_df = news_df.copy()
        news_df[time_column] = pd.to_datetime(news_df[time_column])
        
        # Group by ticker and time period
        grouped = news_df.groupby([ticker_column, pd.Grouper(key=time_column, freq=freq)])
        
        # Aggregate sentiment metrics
        agg_sentiment = grouped.agg({
            'score': ['mean', 'std', 'count'],
            'positive': 'mean',
            'negative': 'mean', 
            'neutral': 'mean'
        }).round(4)
        
        # Flatten column names
        agg_sentiment.columns = ['_'.join(col).strip() for col in agg_sentiment.columns]
        agg_sentiment = agg_sentiment.reset_index()
        
        # Rename columns for clarity
        agg_sentiment = agg_sentiment.rename(columns={
            'score_mean': 'sentiment_score',
            'score_std': 'sentiment_std',
            'score_count': 'article_count',
            'positive_mean': 'positive_ratio',
            'negative_mean': 'negative_ratio',
            'neutral_mean': 'neutral_ratio'
        })
        
        return agg_sentiment
    
    def filter_high_quality_news(self, news_df: pd.DataFrame,
                                min_article_count: int = 1,
                                exclude_publishers: List[str] = None,
                                include_publishers: List[str] = None) -> pd.DataFrame:
        """
        Filter news articles for quality and relevance
        
        Args:
            news_df: DataFrame with news articles
            min_article_count: Minimum number of articles per time period
            exclude_publishers: List of publishers to exclude
            include_publishers: List of publishers to include (if specified, only these are included)
            
        Returns:
            Filtered DataFrame
        """
        if news_df.empty:
            return news_df
        
        filtered_df = news_df.copy()
        
        # Filter by publishers
        if include_publishers:
            filtered_df = filtered_df[filtered_df['publisher'].isin(include_publishers)]
        elif exclude_publishers:
            filtered_df = filtered_df[~filtered_df['publisher'].isin(exclude_publishers)]
        
        # Remove duplicates based on title similarity
        filtered_df = filtered_df.drop_duplicates(subset=['title'], keep='first')
        
        # Filter by article count per time period
        if min_article_count > 1:
            time_counts = filtered_df.groupby(['ticker', pd.Grouper(key='published_utc', freq='D')]).size()
            valid_periods = time_counts[time_counts >= min_article_count].index
            filtered_df = filtered_df.set_index(['ticker', 'published_utc']).loc[valid_periods].reset_index()
        
        return filtered_df
    
    def get_market_sentiment_features(self, news_df: pd.DataFrame,
                                     market_tickers: List[str],
                                     from_date: str, to_date: str) -> pd.DataFrame:
        """
        Generate market-wide sentiment features for MASTER model
        
        Args:
            news_df: DataFrame with processed news articles
            market_tickers: List of market tickers to include
            from_date: Start date for feature generation
            to_date: End date for feature generation
            
        Returns:
            DataFrame with market sentiment features
        """
        if news_df.empty:
            return pd.DataFrame()
        
        # Filter by date range and tickers
        news_df = news_df.copy()
        news_df['published_utc'] = pd.to_datetime(news_df['published_utc'])
        
        date_mask = (news_df['published_utc'] >= from_date) & (news_df['published_utc'] <= to_date)
        ticker_mask = news_df['ticker'].isin(market_tickers)
        
        filtered_news = news_df[date_mask & ticker_mask]
        
        if filtered_news.empty:
            return pd.DataFrame()
        
        # Aggregate sentiment by day and ticker
        daily_sentiment = self.aggregate_sentiment_by_time(filtered_news, freq='D')
        
        # Create market-wide features
        market_features = []
        
        for ticker in market_tickers:
            ticker_sentiment = daily_sentiment[daily_sentiment['ticker'] == ticker]
            
            if not ticker_sentiment.empty:
                # Create date range for the ticker
                date_range = pd.date_range(start=from_date, end=to_date, freq='D')
                
                # Reindex to ensure all dates are present
                ticker_sentiment = ticker_sentiment.set_index('published_utc').reindex(date_range)
                ticker_sentiment['ticker'] = ticker
                ticker_sentiment = ticker_sentiment.reset_index().rename(columns={'index': 'date'})
                
                # Fill missing values
                ticker_sentiment = ticker_sentiment.fillna({
                    'sentiment_score': 0.0,
                    'sentiment_std': 0.0,
                    'article_count': 0,
                    'positive_ratio': 0.33,  # Neutral default
                    'negative_ratio': 0.33,
                    'neutral_ratio': 0.33
                })
                
                market_features.append(ticker_sentiment)
        
        if market_features:
            return pd.concat(market_features, ignore_index=True)
        else:
            return pd.DataFrame()


# Example usage and testing
if __name__ == "__main__":
    # Initialize sentiment processor
    processor = NewsSentimentProcessor()
    
    # Example news data
    sample_news = pd.DataFrame({
        'ticker': ['AAPL', 'AAPL', 'AAPL'],
        'title': ['Apple reports strong earnings', 'Apple stock drops on news', 'Apple announces new product'],
        'description': ['Apple Inc. reported better than expected quarterly earnings...', 
                       'Apple shares fell after disappointing guidance...',
                       'Apple unveiled its latest iPhone with advanced features...'],
        'published_utc': ['2024-01-15', '2024-01-16', '2024-01-17'],
        'publisher': ['Reuters', 'Bloomberg', 'TechCrunch']
    })
    
    # Process sentiment
    print("Processing news sentiment...")
    processed_news = processor.process_news_batch(sample_news)
    print(processed_news[['title', 'sentiment_score', 'label']].head())
    
    # Aggregate by time
    print("\nAggregating sentiment by time...")
    aggregated = processor.aggregate_sentiment_by_time(processed_news)
    print(aggregated.head())
