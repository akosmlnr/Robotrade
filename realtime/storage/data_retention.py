"""
Data Retention Policies for Real-time LSTM Prediction System
Phase 3.2: Comprehensive data retention and lifecycle management
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

class RetentionPolicy(Enum):
    """Data retention policy types"""
    TIME_BASED = "time_based"
    SIZE_BASED = "size_based"
    PERFORMANCE_BASED = "performance_based"
    ARCHIVAL = "archival"
    COMPRESSION = "compression"

class DataCategory(Enum):
    """Data categories for retention policies"""
    MARKET_DATA = "market_data"
    PREDICTIONS = "predictions"
    VALIDATION_RESULTS = "validation_results"
    PERFORMANCE_METRICS = "performance_metrics"
    SYSTEM_LOGS = "system_logs"
    MODEL_ARTIFACTS = "model_artifacts"
    TRADE_RECOMMENDATIONS = "trade_recommendations"
    SYSTEM_ALERTS = "system_alerts"

@dataclass
class RetentionRule:
    """Data retention rule configuration"""
    category: DataCategory
    policy: RetentionPolicy
    retention_period_days: int
    max_size_mb: Optional[int] = None
    compression_enabled: bool = False
    archival_enabled: bool = False
    archival_location: Optional[str] = None
    performance_threshold: Optional[float] = None
    enabled: bool = True
    last_executed: Optional[datetime] = None

@dataclass
class RetentionResult:
    """Result of retention policy execution"""
    category: DataCategory
    policy: RetentionPolicy
    records_deleted: int
    records_archived: int
    space_freed_mb: float
    execution_time_seconds: float
    success: bool
    error_message: Optional[str] = None
    executed_at: datetime = None
    
    def __post_init__(self):
        if self.executed_at is None:
            self.executed_at = datetime.now()

class DataRetentionManager:
    """
    Comprehensive data retention and lifecycle management system
    """
    
    def __init__(self, data_storage, config: Dict[str, Any] = None):
        """
        Initialize data retention manager
        
        Args:
            data_storage: DataStorage instance
            config: Optional configuration dictionary
        """
        self.data_storage = data_storage
        
        # Default retention policies
        self.retention_rules = self._initialize_default_rules()
        
        # Load custom configuration if provided
        if config:
            self._load_configuration(config)
        
        # Retention statistics
        self.retention_stats = {
            'total_executions': 0,
            'total_records_deleted': 0,
            'total_records_archived': 0,
            'total_space_freed_mb': 0.0,
            'last_execution': None,
            'execution_history': []
        }
        
        # Archival settings
        self.archival_base_path = Path("archived_data")
        self.archival_base_path.mkdir(exist_ok=True)
        
        logger.info("DataRetentionManager initialized")
    
    def _initialize_default_rules(self) -> Dict[DataCategory, RetentionRule]:
        """Initialize default retention rules"""
        return {
            DataCategory.MARKET_DATA: RetentionRule(
                category=DataCategory.MARKET_DATA,
                policy=RetentionPolicy.TIME_BASED,
                retention_period_days=90,
                compression_enabled=True,
                archival_enabled=True
            ),
            DataCategory.PREDICTIONS: RetentionRule(
                category=DataCategory.PREDICTIONS,
                policy=RetentionPolicy.TIME_BASED,
                retention_period_days=60,
                compression_enabled=True,
                archival_enabled=True
            ),
            DataCategory.VALIDATION_RESULTS: RetentionRule(
                category=DataCategory.VALIDATION_RESULTS,
                policy=RetentionPolicy.TIME_BASED,
                retention_period_days=30,
                compression_enabled=False,
                archival_enabled=False
            ),
            DataCategory.PERFORMANCE_METRICS: RetentionRule(
                category=DataCategory.PERFORMANCE_METRICS,
                policy=RetentionPolicy.TIME_BASED,
                retention_period_days=180,
                compression_enabled=True,
                archival_enabled=True
            ),
            DataCategory.SYSTEM_LOGS: RetentionRule(
                category=DataCategory.SYSTEM_LOGS,
                policy=RetentionPolicy.SIZE_BASED,
                retention_period_days=30,
                max_size_mb=1000,
                compression_enabled=True,
                archival_enabled=True
            ),
            DataCategory.MODEL_ARTIFACTS: RetentionRule(
                category=DataCategory.MODEL_ARTIFACTS,
                policy=RetentionPolicy.PERFORMANCE_BASED,
                retention_period_days=365,
                performance_threshold=0.7,
                compression_enabled=True,
                archival_enabled=True
            ),
            DataCategory.TRADE_RECOMMENDATIONS: RetentionRule(
                category=DataCategory.TRADE_RECOMMENDATIONS,
                policy=RetentionPolicy.TIME_BASED,
                retention_period_days=90,
                compression_enabled=True,
                archival_enabled=True
            ),
            DataCategory.SYSTEM_ALERTS: RetentionRule(
                category=DataCategory.SYSTEM_ALERTS,
                policy=RetentionPolicy.TIME_BASED,
                retention_period_days=30,
                compression_enabled=False,
                archival_enabled=False
            )
        }
    
    def _load_configuration(self, config: Dict[str, Any]):
        """Load custom configuration"""
        try:
            for category_name, rule_config in config.get('retention_rules', {}).items():
                category = DataCategory(category_name)
                if category in self.retention_rules:
                    rule = self.retention_rules[category]
                    
                    # Update rule with custom configuration
                    if 'retention_period_days' in rule_config:
                        rule.retention_period_days = rule_config['retention_period_days']
                    if 'max_size_mb' in rule_config:
                        rule.max_size_mb = rule_config['max_size_mb']
                    if 'compression_enabled' in rule_config:
                        rule.compression_enabled = rule_config['compression_enabled']
                    if 'archival_enabled' in rule_config:
                        rule.archival_enabled = rule_config['archival_enabled']
                    if 'enabled' in rule_config:
                        rule.enabled = rule_config['enabled']
            
            logger.info("Custom retention configuration loaded")
            
        except Exception as e:
            logger.error(f"Error loading retention configuration: {e}")
    
    def execute_retention_policies(self, categories: List[DataCategory] = None) -> List[RetentionResult]:
        """
        Execute retention policies for specified categories
        
        Args:
            categories: List of categories to process (None for all)
            
        Returns:
            List of RetentionResult objects
        """
        try:
            if categories is None:
                categories = list(self.retention_rules.keys())
            
            results = []
            
            for category in categories:
                if category not in self.retention_rules:
                    continue
                
                rule = self.retention_rules[category]
                if not rule.enabled:
                    continue
                
                logger.info(f"Executing retention policy for {category.value}")
                
                start_time = datetime.now()
                result = self._execute_retention_rule(rule)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                result.execution_time_seconds = execution_time
                results.append(result)
                
                # Update rule last executed time
                rule.last_executed = datetime.now()
                
                # Update statistics
                self._update_retention_stats(result)
            
            self.retention_stats['last_execution'] = datetime.now()
            self.retention_stats['total_executions'] += 1
            
            logger.info(f"Retention policies executed for {len(results)} categories")
            return results
            
        except Exception as e:
            logger.error(f"Error executing retention policies: {e}")
            return []
    
    def _execute_retention_rule(self, rule: RetentionRule) -> RetentionResult:
        """Execute a specific retention rule"""
        try:
            if rule.policy == RetentionPolicy.TIME_BASED:
                return self._execute_time_based_retention(rule)
            elif rule.policy == RetentionPolicy.SIZE_BASED:
                return self._execute_size_based_retention(rule)
            elif rule.policy == RetentionPolicy.PERFORMANCE_BASED:
                return self._execute_performance_based_retention(rule)
            else:
                return RetentionResult(
                    category=rule.category,
                    policy=rule.policy,
                    records_deleted=0,
                    records_archived=0,
                    space_freed_mb=0.0,
                    execution_time_seconds=0.0,
                    success=False,
                    error_message=f"Unsupported policy: {rule.policy.value}"
                )
                
        except Exception as e:
            logger.error(f"Error executing retention rule for {rule.category.value}: {e}")
            return RetentionResult(
                category=rule.category,
                policy=rule.policy,
                records_deleted=0,
                records_archived=0,
                space_freed_mb=0.0,
                execution_time_seconds=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _execute_time_based_retention(self, rule: RetentionRule) -> RetentionResult:
        """Execute time-based retention policy"""
        try:
            cutoff_date = datetime.now() - timedelta(days=rule.retention_period_days)
            
            # Get table name for category
            table_name = self._get_table_name(rule.category)
            if not table_name:
                raise ValueError(f"No table mapping for category {rule.category.value}")
            
            # Count records to be deleted
            cursor = self.data_storage.connection.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE created_at < ?", (cutoff_date,))
            records_to_delete = cursor.fetchone()[0]
            
            if records_to_delete == 0:
                return RetentionResult(
                    category=rule.category,
                    policy=rule.policy,
                    records_deleted=0,
                    records_archived=0,
                    space_freed_mb=0.0,
                    execution_time_seconds=0.0,
                    success=True
                )
            
            # Archive data if enabled
            archived_count = 0
            if rule.archival_enabled:
                archived_count = self._archive_data(rule.category, cutoff_date)
            
            # Delete old records
            cursor.execute(f"DELETE FROM {table_name} WHERE created_at < ?", (cutoff_date,))
            deleted_count = cursor.rowcount
            
            self.data_storage.connection.commit()
            
            # Estimate space freed (rough calculation)
            space_freed = self._estimate_space_freed(table_name, deleted_count)
            
            logger.info(f"Time-based retention for {rule.category.value}: deleted {deleted_count} records, archived {archived_count}")
            
            return RetentionResult(
                category=rule.category,
                policy=rule.policy,
                records_deleted=deleted_count,
                records_archived=archived_count,
                space_freed_mb=space_freed,
                execution_time_seconds=0.0,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in time-based retention for {rule.category.value}: {e}")
            return RetentionResult(
                category=rule.category,
                policy=rule.policy,
                records_deleted=0,
                records_archived=0,
                space_freed_mb=0.0,
                execution_time_seconds=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _execute_size_based_retention(self, rule: RetentionRule) -> RetentionResult:
        """Execute size-based retention policy"""
        try:
            if not rule.max_size_mb:
                raise ValueError("Size-based retention requires max_size_mb to be set")
            
            # Get table name for category
            table_name = self._get_table_name(rule.category)
            if not table_name:
                raise ValueError(f"No table mapping for category {rule.category.value}")
            
            # Get current database size
            cursor = self.data_storage.connection.cursor()
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            current_size_bytes = cursor.fetchone()[0]
            current_size_mb = current_size_bytes / (1024 * 1024)
            
            if current_size_mb <= rule.max_size_mb:
                return RetentionResult(
                    category=rule.category,
                    policy=rule.policy,
                    records_deleted=0,
                    records_archived=0,
                    space_freed_mb=0.0,
                    execution_time_seconds=0.0,
                    success=True
                )
            
            # Calculate how much to delete
            excess_mb = current_size_mb - rule.max_size_mb
            target_deletion_ratio = excess_mb / current_size_mb
            
            # Get oldest records to delete
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_records = cursor.fetchone()[0]
            records_to_delete = int(total_records * target_deletion_ratio)
            
            if records_to_delete == 0:
                return RetentionResult(
                    category=rule.category,
                    policy=rule.policy,
                    records_deleted=0,
                    records_archived=0,
                    space_freed_mb=0.0,
                    execution_time_seconds=0.0,
                    success=True
                )
            
            # Archive data if enabled
            archived_count = 0
            if rule.archival_enabled:
                archived_count = self._archive_oldest_data(rule.category, records_to_delete)
            
            # Delete oldest records
            cursor.execute(f"""
                DELETE FROM {table_name} 
                WHERE id IN (
                    SELECT id FROM {table_name} 
                    ORDER BY created_at ASC 
                    LIMIT ?
                )
            """, (records_to_delete,))
            
            deleted_count = cursor.rowcount
            self.data_storage.connection.commit()
            
            # Estimate space freed
            space_freed = self._estimate_space_freed(table_name, deleted_count)
            
            logger.info(f"Size-based retention for {rule.category.value}: deleted {deleted_count} records, archived {archived_count}")
            
            return RetentionResult(
                category=rule.category,
                policy=rule.policy,
                records_deleted=deleted_count,
                records_archived=archived_count,
                space_freed_mb=space_freed,
                execution_time_seconds=0.0,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in size-based retention for {rule.category.value}: {e}")
            return RetentionResult(
                category=rule.category,
                policy=rule.policy,
                records_deleted=0,
                records_archived=0,
                space_freed_mb=0.0,
                execution_time_seconds=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _execute_performance_based_retention(self, rule: RetentionRule) -> RetentionResult:
        """Execute performance-based retention policy"""
        try:
            if not rule.performance_threshold:
                raise ValueError("Performance-based retention requires performance_threshold to be set")
            
            # This is a simplified implementation
            # In practice, you would analyze model performance metrics
            # and delete/archive underperforming models and their associated data
            
            logger.info(f"Performance-based retention for {rule.category.value} - threshold: {rule.performance_threshold}")
            
            # For now, return a placeholder result
            return RetentionResult(
                category=rule.category,
                policy=rule.policy,
                records_deleted=0,
                records_archived=0,
                space_freed_mb=0.0,
                execution_time_seconds=0.0,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in performance-based retention for {rule.category.value}: {e}")
            return RetentionResult(
                category=rule.category,
                policy=rule.policy,
                records_deleted=0,
                records_archived=0,
                space_freed_mb=0.0,
                execution_time_seconds=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _get_table_name(self, category: DataCategory) -> Optional[str]:
        """Get database table name for data category"""
        table_mapping = {
            DataCategory.MARKET_DATA: 'market_data',
            DataCategory.PREDICTIONS: 'predictions',
            DataCategory.VALIDATION_RESULTS: 'validation_results',
            DataCategory.PERFORMANCE_METRICS: 'performance_metrics',
            DataCategory.TRADE_RECOMMENDATIONS: 'trade_recommendations',
            DataCategory.SYSTEM_ALERTS: 'system_alerts'
        }
        return table_mapping.get(category)
    
    def _archive_data(self, category: DataCategory, cutoff_date: datetime) -> int:
        """Archive data before deletion"""
        try:
            table_name = self._get_table_name(category)
            if not table_name:
                return 0
            
            # Create archival directory
            archive_dir = self.archival_base_path / category.value
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Export data to CSV
            cursor = self.data_storage.connection.cursor()
            cursor.execute(f"SELECT * FROM {table_name} WHERE created_at < ?", (cutoff_date,))
            rows = cursor.fetchall()
            
            if not rows:
                return 0
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame([dict(row) for row in rows])
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_file = archive_dir / f"{table_name}_{timestamp}.csv"
            df.to_csv(archive_file, index=False)
            
            # Compress if enabled
            rule = self.retention_rules.get(category)
            if rule and rule.compression_enabled:
                import gzip
                compressed_file = archive_file.with_suffix('.csv.gz')
                with open(archive_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                archive_file.unlink()  # Remove uncompressed file
                archive_file = compressed_file
            
            logger.info(f"Archived {len(rows)} records from {table_name} to {archive_file}")
            return len(rows)
            
        except Exception as e:
            logger.error(f"Error archiving data for {category.value}: {e}")
            return 0
    
    def _archive_oldest_data(self, category: DataCategory, count: int) -> int:
        """Archive oldest data records"""
        try:
            table_name = self._get_table_name(category)
            if not table_name:
                return 0
            
            # Create archival directory
            archive_dir = self.archival_base_path / category.value
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Get oldest records
            cursor = self.data_storage.connection.cursor()
            cursor.execute(f"""
                SELECT * FROM {table_name} 
                ORDER BY created_at ASC 
                LIMIT ?
            """, (count,))
            rows = cursor.fetchall()
            
            if not rows:
                return 0
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame([dict(row) for row in rows])
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_file = archive_dir / f"{table_name}_oldest_{timestamp}.csv"
            df.to_csv(archive_file, index=False)
            
            # Compress if enabled
            rule = self.retention_rules.get(category)
            if rule and rule.compression_enabled:
                import gzip
                compressed_file = archive_file.with_suffix('.csv.gz')
                with open(archive_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                archive_file.unlink()  # Remove uncompressed file
                archive_file = compressed_file
            
            logger.info(f"Archived {len(rows)} oldest records from {table_name} to {archive_file}")
            return len(rows)
            
        except Exception as e:
            logger.error(f"Error archiving oldest data for {category.value}: {e}")
            return 0
    
    def _estimate_space_freed(self, table_name: str, record_count: int) -> float:
        """Estimate space freed by deletion (rough calculation)"""
        try:
            # This is a simplified estimation
            # In practice, you might want to calculate actual space usage
            estimated_bytes_per_record = 1024  # 1KB per record (rough estimate)
            space_freed_bytes = record_count * estimated_bytes_per_record
            return space_freed_bytes / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.error(f"Error estimating space freed: {e}")
            return 0.0
    
    def _update_retention_stats(self, result: RetentionResult):
        """Update retention statistics"""
        try:
            self.retention_stats['total_records_deleted'] += result.records_deleted
            self.retention_stats['total_records_archived'] += result.records_archived
            self.retention_stats['total_space_freed_mb'] += result.space_freed_mb
            
            # Add to execution history (keep last 100)
            self.retention_stats['execution_history'].append({
                'category': result.category.value,
                'policy': result.policy.value,
                'records_deleted': result.records_deleted,
                'records_archived': result.records_archived,
                'space_freed_mb': result.space_freed_mb,
                'execution_time_seconds': result.execution_time_seconds,
                'success': result.success,
                'executed_at': result.executed_at.isoformat()
            })
            
            if len(self.retention_stats['execution_history']) > 100:
                self.retention_stats['execution_history'] = self.retention_stats['execution_history'][-100:]
            
        except Exception as e:
            logger.error(f"Error updating retention stats: {e}")
    
    def get_retention_status(self) -> Dict[str, Any]:
        """Get current retention status and statistics"""
        try:
            status = {
                'retention_rules': {},
                'statistics': self.retention_stats.copy(),
                'archival_info': {
                    'archival_path': str(self.archival_base_path),
                    'archival_size_mb': self._get_archival_size()
                }
            }
            
            # Add rule information
            for category, rule in self.retention_rules.items():
                status['retention_rules'][category.value] = {
                    'policy': rule.policy.value,
                    'retention_period_days': rule.retention_period_days,
                    'max_size_mb': rule.max_size_mb,
                    'compression_enabled': rule.compression_enabled,
                    'archival_enabled': rule.archival_enabled,
                    'enabled': rule.enabled,
                    'last_executed': rule.last_executed.isoformat() if rule.last_executed else None
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting retention status: {e}")
            return {}
    
    def _get_archival_size(self) -> float:
        """Get total size of archived data in MB"""
        try:
            total_size = 0
            for file_path in self.archival_base_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.error(f"Error calculating archival size: {e}")
            return 0.0
    
    def cleanup_archives(self, days_back: int = 365) -> bool:
        """Clean up old archive files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            deleted_files = 0
            
            for file_path in self.archival_base_path.rglob('*'):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_path.unlink()
                        deleted_files += 1
            
            logger.info(f"Cleaned up {deleted_files} old archive files")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up archives: {e}")
            return False
    
    def update_retention_rule(self, category: DataCategory, **kwargs) -> bool:
        """Update a retention rule"""
        try:
            if category not in self.retention_rules:
                return False
            
            rule = self.retention_rules[category]
            
            # Update rule attributes
            for key, value in kwargs.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            
            logger.info(f"Updated retention rule for {category.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating retention rule for {category.value}: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("DataRetentionManager module loaded successfully")
    print("Use with DataStorage instance for full functionality")
