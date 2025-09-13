"""
Performance Dashboards for Real-time LSTM Prediction System
Phase 3.3: Comprehensive performance dashboards and visualization
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

class DashboardType(Enum):
    """Types of dashboards"""
    SYSTEM_OVERVIEW = "system_overview"
    PREDICTION_PERFORMANCE = "prediction_performance"
    MODEL_ANALYTICS = "model_analytics"
    BUSINESS_METRICS = "business_metrics"
    ALERT_MONITORING = "alert_monitoring"
    DATA_QUALITY = "data_quality"

class ChartType(Enum):
    """Types of charts"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    TABLE = "table"

@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    title: str
    chart_type: ChartType
    data_source: str
    refresh_interval_seconds: int = 60
    width: int = 6
    height: int = 4
    position_x: int = 0
    position_y: int = 0
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}

@dataclass
class Dashboard:
    """Dashboard configuration"""
    dashboard_id: str
    title: str
    dashboard_type: DashboardType
    widgets: List[DashboardWidget]
    refresh_interval_seconds: int = 60
    auto_refresh: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class PerformanceDashboard:
    """
    Comprehensive performance dashboard system with real-time visualizations
    """
    
    def __init__(self, data_storage, config: Dict[str, Any] = None):
        """
        Initialize performance dashboard
        
        Args:
            data_storage: DataStorage instance
            config: Optional configuration dictionary
        """
        self.data_storage = data_storage
        
        # Configuration
        self.config = self._load_default_config()
        if config:
            self.config.update(config)
        
        # Dashboard storage
        self.dashboards: Dict[str, Dashboard] = {}
        self.dashboard_templates: Dict[DashboardType, Dashboard] = {}
        
        # Output settings
        self.output_path = Path(self.config.get('output_path', 'dashboards'))
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize default dashboards
        self._create_default_dashboards()
        
        logger.info("PerformanceDashboard initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'output_path': 'dashboards',
            'default_refresh_interval': 60,
            'chart_theme': 'plotly_white',
            'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'figure_size': (12, 8),
            'dpi': 300,
            'auto_save': True,
            'export_formats': ['html', 'png', 'pdf'],
            'max_data_points': 1000
        }
    
    def _create_default_dashboards(self):
        """Create default dashboard templates"""
        try:
            # System Overview Dashboard
            system_widgets = [
                DashboardWidget(
                    widget_id="cpu_usage",
                    title="CPU Usage",
                    chart_type=ChartType.LINE,
                    data_source="performance_metrics",
                    config={'metric_name': 'cpu_usage_percent', 'unit': '%'}
                ),
                DashboardWidget(
                    widget_id="memory_usage",
                    title="Memory Usage",
                    chart_type=ChartType.LINE,
                    data_source="performance_metrics",
                    config={'metric_name': 'memory_usage_percent', 'unit': '%'}
                ),
                DashboardWidget(
                    widget_id="prediction_latency",
                    title="Prediction Latency",
                    chart_type=ChartType.LINE,
                    data_source="performance_metrics",
                    config={'metric_name': 'prediction_latency_ms', 'unit': 'ms'}
                ),
                DashboardWidget(
                    widget_id="system_health",
                    title="System Health",
                    chart_type=ChartType.GAUGE,
                    data_source="system_health",
                    config={'thresholds': [70, 85, 95]}
                )
            ]
            
            system_dashboard = Dashboard(
                dashboard_id="system_overview",
                title="System Overview",
                dashboard_type=DashboardType.SYSTEM_OVERVIEW,
                widgets=system_widgets
            )
            
            self.dashboard_templates[DashboardType.SYSTEM_OVERVIEW] = system_dashboard
            
            # Prediction Performance Dashboard
            prediction_widgets = [
                DashboardWidget(
                    widget_id="accuracy_trend",
                    title="Prediction Accuracy Trend",
                    chart_type=ChartType.LINE,
                    data_source="prediction_accuracy",
                    config={'metric': 'hit_rate'}
                ),
                DashboardWidget(
                    widget_id="accuracy_distribution",
                    title="Accuracy Distribution",
                    chart_type=ChartType.BAR,
                    data_source="accuracy_distribution",
                    config={}
                ),
                DashboardWidget(
                    widget_id="prediction_volume",
                    title="Prediction Volume",
                    chart_type=ChartType.BAR,
                    data_source="prediction_volume",
                    config={'timeframe': 'hourly'}
                ),
                DashboardWidget(
                    widget_id="confidence_scores",
                    title="Confidence Scores",
                    chart_type=ChartType.SCATTER,
                    data_source="confidence_scores",
                    config={}
                )
            ]
            
            prediction_dashboard = Dashboard(
                dashboard_id="prediction_performance",
                title="Prediction Performance",
                dashboard_type=DashboardType.PREDICTION_PERFORMANCE,
                widgets=prediction_widgets
            )
            
            self.dashboard_templates[DashboardType.PREDICTION_PERFORMANCE] = prediction_dashboard
            
            # Model Analytics Dashboard
            model_widgets = [
                DashboardWidget(
                    widget_id="model_performance",
                    title="Model Performance Metrics",
                    chart_type=ChartType.TABLE,
                    data_source="model_performance",
                    config={}
                ),
                DashboardWidget(
                    widget_id="feature_importance",
                    title="Feature Importance",
                    chart_type=ChartType.BAR,
                    data_source="feature_importance",
                    config={}
                ),
                DashboardWidget(
                    widget_id="model_drift",
                    title="Model Drift Detection",
                    chart_type=ChartType.LINE,
                    data_source="model_drift",
                    config={}
                ),
                DashboardWidget(
                    widget_id="prediction_errors",
                    title="Prediction Errors",
                    chart_type=ChartType.HEATMAP,
                    data_source="prediction_errors",
                    config={}
                )
            ]
            
            model_dashboard = Dashboard(
                dashboard_id="model_analytics",
                title="Model Analytics",
                dashboard_type=DashboardType.MODEL_ANALYTICS,
                widgets=model_widgets
            )
            
            self.dashboard_templates[DashboardType.MODEL_ANALYTICS] = model_dashboard
            
            # Alert Monitoring Dashboard
            alert_widgets = [
                DashboardWidget(
                    widget_id="active_alerts",
                    title="Active Alerts",
                    chart_type=ChartType.TABLE,
                    data_source="active_alerts",
                    config={}
                ),
                DashboardWidget(
                    widget_id="alert_trends",
                    title="Alert Trends",
                    chart_type=ChartType.LINE,
                    data_source="alert_trends",
                    config={}
                ),
                DashboardWidget(
                    widget_id="alert_severity",
                    title="Alert Severity Distribution",
                    chart_type=ChartType.PIE,
                    data_source="alert_severity",
                    config={}
                ),
                DashboardWidget(
                    widget_id="alert_resolution",
                    title="Alert Resolution Time",
                    chart_type=ChartType.BAR,
                    data_source="alert_resolution",
                    config={}
                )
            ]
            
            alert_dashboard = Dashboard(
                dashboard_id="alert_monitoring",
                title="Alert Monitoring",
                dashboard_type=DashboardType.ALERT_MONITORING,
                widgets=alert_widgets
            )
            
            self.dashboard_templates[DashboardType.ALERT_MONITORING] = alert_dashboard
            
            logger.info(f"Created {len(self.dashboard_templates)} default dashboard templates")
            
        except Exception as e:
            logger.error(f"Error creating default dashboards: {e}")
    
    def create_dashboard(self, dashboard_type: DashboardType, 
                        custom_title: str = None) -> str:
        """
        Create a new dashboard instance
        
        Args:
            dashboard_type: Type of dashboard to create
            custom_title: Custom title for the dashboard
            
        Returns:
            Dashboard ID
        """
        try:
            if dashboard_type not in self.dashboard_templates:
                logger.error(f"Dashboard template for {dashboard_type.value} not found")
                return ""
            
            template = self.dashboard_templates[dashboard_type]
            dashboard_id = f"{dashboard_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create dashboard instance
            dashboard = Dashboard(
                dashboard_id=dashboard_id,
                title=custom_title or template.title,
                dashboard_type=dashboard_type,
                widgets=template.widgets.copy(),
                refresh_interval_seconds=template.refresh_interval_seconds,
                auto_refresh=template.auto_refresh
            )
            
            self.dashboards[dashboard_id] = dashboard
            
            logger.info(f"Created dashboard: {dashboard_id}")
            return dashboard_id
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            return ""
    
    def generate_dashboard(self, dashboard_id: str, 
                          output_format: str = 'html') -> str:
        """
        Generate dashboard visualization
        
        Args:
            dashboard_id: Dashboard ID
            output_format: Output format (html, png, pdf)
            
        Returns:
            Path to generated dashboard file
        """
        try:
            if dashboard_id not in self.dashboards:
                logger.error(f"Dashboard {dashboard_id} not found")
                return ""
            
            dashboard = self.dashboards[dashboard_id]
            
            # Create subplots layout
            widget_count = len(dashboard.widgets)
            rows = (widget_count + 1) // 2  # 2 widgets per row
            cols = 2
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[widget.title for widget in dashboard.widgets],
                specs=[[{"type": "scatter"} for _ in range(cols)] for _ in range(rows)]
            )
            
            # Generate each widget
            for i, widget in enumerate(dashboard.widgets):
                row = (i // cols) + 1
                col = (i % cols) + 1
                
                try:
                    chart_data = self._generate_widget_data(widget)
                    chart_fig = self._create_widget_chart(widget, chart_data)
                    
                    # Add traces to main figure
                    for trace in chart_fig.data:
                        fig.add_trace(trace, row=row, col=col)
                    
                except Exception as e:
                    logger.error(f"Error generating widget {widget.widget_id}: {e}")
                    continue
            
            # Update layout
            fig.update_layout(
                title=f"{dashboard.title} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                showlegend=False,
                height=400 * rows,
                template=self.config.get('chart_theme', 'plotly_white')
            )
            
            # Generate output file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{dashboard_id}_{timestamp}.{output_format}"
            output_path = self.output_path / filename
            
            if output_format == 'html':
                fig.write_html(str(output_path))
            elif output_format == 'png':
                fig.write_image(str(output_path), width=1200, height=400*rows, scale=2)
            elif output_format == 'pdf':
                fig.write_image(str(output_path), format='pdf', width=1200, height=400*rows)
            else:
                logger.error(f"Unsupported output format: {output_format}")
                return ""
            
            logger.info(f"Generated dashboard: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating dashboard {dashboard_id}: {e}")
            return ""
    
    def _generate_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Generate data for a widget"""
        try:
            data_source = widget.data_source
            
            if data_source == "performance_metrics":
                return self._get_performance_metrics_data(widget.config)
            elif data_source == "prediction_accuracy":
                return self._get_prediction_accuracy_data(widget.config)
            elif data_source == "system_health":
                return self._get_system_health_data(widget.config)
            elif data_source == "active_alerts":
                return self._get_active_alerts_data(widget.config)
            elif data_source == "alert_trends":
                return self._get_alert_trends_data(widget.config)
            elif data_source == "model_performance":
                return self._get_model_performance_data(widget.config)
            else:
                logger.warning(f"Unknown data source: {data_source}")
                return {}
                
        except Exception as e:
            logger.error(f"Error generating widget data: {e}")
            return {}
    
    def _get_performance_metrics_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get performance metrics data"""
        try:
            metric_name = config.get('metric_name', 'cpu_usage_percent')
            hours_back = config.get('hours_back', 24)
            
            # Get metrics from database
            metrics_df = self.data_storage.get_performance_metrics(
                metric_name=metric_name,
                days_back=hours_back // 24 + 1
            )
            
            if metrics_df.empty:
                return {'x': [], 'y': []}
            
            # Filter recent data
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            metrics_df['metric_timestamp'] = pd.to_datetime(metrics_df['metric_timestamp'])
            recent_metrics = metrics_df[metrics_df['metric_timestamp'] >= cutoff_time]
            
            return {
                'x': recent_metrics['metric_timestamp'].tolist(),
                'y': recent_metrics['metric_value'].tolist(),
                'name': metric_name,
                'unit': config.get('unit', '')
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics data: {e}")
            return {'x': [], 'y': []}
    
    def _get_prediction_accuracy_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction accuracy data"""
        try:
            metric = config.get('metric', 'hit_rate')
            days_back = config.get('days_back', 7)
            
            # Get accuracy data from database
            cursor = self.data_storage.connection.cursor()
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            cursor.execute("""
                SELECT prediction_timestamp, {} as value
                FROM prediction_accuracy 
                WHERE prediction_timestamp >= ?
                ORDER BY prediction_timestamp
            """.format(metric), (cutoff_date,))
            
            rows = cursor.fetchall()
            
            if not rows:
                return {'x': [], 'y': []}
            
            timestamps = [row[0] for row in rows]
            values = [row[1] for row in rows]
            
            return {
                'x': timestamps,
                'y': values,
                'name': metric,
                'unit': '%' if 'rate' in metric else ''
            }
            
        except Exception as e:
            logger.error(f"Error getting prediction accuracy data: {e}")
            return {'x': [], 'y': []}
    
    def _get_system_health_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get system health data"""
        try:
            # Calculate overall system health score
            # This is a simplified calculation
            health_score = 85.0  # Placeholder
            
            return {
                'value': health_score,
                'thresholds': config.get('thresholds', [70, 85, 95]),
                'unit': '%'
            }
            
        except Exception as e:
            logger.error(f"Error getting system health data: {e}")
            return {'value': 0}
    
    def _get_active_alerts_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get active alerts data"""
        try:
            # Get active alerts from database
            alerts_df = self.data_storage.get_system_alerts(resolved=False, days_back=1)
            
            if alerts_df.empty:
                return {'data': []}
            
            # Format for table display
            table_data = []
            for _, row in alerts_df.iterrows():
                table_data.append({
                    'Alert ID': row['id'],
                    'Type': row['alert_type'],
                    'Severity': row['severity'],
                    'Message': row['message'][:50] + '...' if len(row['message']) > 50 else row['message'],
                    'Created': row['created_at']
                })
            
            return {'data': table_data}
            
        except Exception as e:
            logger.error(f"Error getting active alerts data: {e}")
            return {'data': []}
    
    def _get_alert_trends_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get alert trends data"""
        try:
            hours_back = config.get('hours_back', 24)
            
            # Get alerts from database
            alerts_df = self.data_storage.get_system_alerts(days_back=hours_back // 24 + 1)
            
            if alerts_df.empty:
                return {'x': [], 'y': []}
            
            # Group by hour
            alerts_df['created_at'] = pd.to_datetime(alerts_df['created_at'])
            alerts_df['hour'] = alerts_df['created_at'].dt.floor('H')
            
            hourly_counts = alerts_df.groupby('hour').size().reset_index(name='count')
            
            return {
                'x': hourly_counts['hour'].tolist(),
                'y': hourly_counts['count'].tolist(),
                'name': 'Alerts per Hour'
            }
            
        except Exception as e:
            logger.error(f"Error getting alert trends data: {e}")
            return {'x': [], 'y': []}
    
    def _get_model_performance_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get model performance data"""
        try:
            # Get model performance metrics
            cursor = self.data_storage.connection.cursor()
            cursor.execute("""
                SELECT symbol, accuracy, rmse, mae, total_predictions, correct_predictions
                FROM model_performance 
                ORDER BY timestamp DESC
                LIMIT 10
            """)
            
            rows = cursor.fetchall()
            
            if not rows:
                return {'data': []}
            
            # Format for table display
            table_data = []
            for row in rows:
                table_data.append({
                    'Symbol': row[0],
                    'Accuracy': f"{row[1]:.3f}" if row[1] else 'N/A',
                    'RMSE': f"{row[2]:.4f}" if row[2] else 'N/A',
                    'MAE': f"{row[3]:.4f}" if row[3] else 'N/A',
                    'Total Predictions': row[4] or 0,
                    'Correct Predictions': row[5] or 0
                })
            
            return {'data': table_data}
            
        except Exception as e:
            logger.error(f"Error getting model performance data: {e}")
            return {'data': []}
    
    def _create_widget_chart(self, widget: DashboardWidget, data: Dict[str, Any]) -> go.Figure:
        """Create chart for a widget"""
        try:
            chart_type = widget.chart_type
            
            if chart_type == ChartType.LINE:
                return self._create_line_chart(data)
            elif chart_type == ChartType.BAR:
                return self._create_bar_chart(data)
            elif chart_type == ChartType.PIE:
                return self._create_pie_chart(data)
            elif chart_type == ChartType.SCATTER:
                return self._create_scatter_chart(data)
            elif chart_type == ChartType.HEATMAP:
                return self._create_heatmap_chart(data)
            elif chart_type == ChartType.GAUGE:
                return self._create_gauge_chart(data)
            elif chart_type == ChartType.TABLE:
                return self._create_table_chart(data)
            else:
                logger.warning(f"Unsupported chart type: {chart_type}")
                return go.Figure()
                
        except Exception as e:
            logger.error(f"Error creating widget chart: {e}")
            return go.Figure()
    
    def _create_line_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create line chart"""
        fig = go.Figure()
        
        if 'x' in data and 'y' in data:
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='lines+markers',
                name=data.get('name', 'Data'),
                line=dict(color=self.config['color_palette'][0])
            ))
        
        return fig
    
    def _create_bar_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create bar chart"""
        fig = go.Figure()
        
        if 'x' in data and 'y' in data:
            fig.add_trace(go.Bar(
                x=data['x'],
                y=data['y'],
                name=data.get('name', 'Data'),
                marker_color=self.config['color_palette'][1]
            ))
        
        return fig
    
    def _create_pie_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create pie chart"""
        fig = go.Figure()
        
        if 'labels' in data and 'values' in data:
            fig.add_trace(go.Pie(
                labels=data['labels'],
                values=data['values'],
                marker_colors=self.config['color_palette']
            ))
        
        return fig
    
    def _create_scatter_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create scatter chart"""
        fig = go.Figure()
        
        if 'x' in data and 'y' in data:
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='markers',
                name=data.get('name', 'Data'),
                marker=dict(color=self.config['color_palette'][2])
            ))
        
        return fig
    
    def _create_heatmap_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create heatmap chart"""
        fig = go.Figure()
        
        if 'z' in data:
            fig.add_trace(go.Heatmap(
                z=data['z'],
                colorscale='Viridis'
            ))
        
        return fig
    
    def _create_gauge_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create gauge chart"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=data.get('value', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "System Health"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        return fig
    
    def _create_table_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create table chart"""
        fig = go.Figure()
        
        if 'data' in data and data['data']:
            # Convert data to table format
            table_data = data['data']
            if table_data:
                headers = list(table_data[0].keys())
                values = [list(row.values()) for row in table_data]
                
                fig.add_trace(go.Table(
                    header=dict(values=headers, fill_color='paleturquoise'),
                    cells=dict(values=list(zip(*values)), fill_color='lavender')
                ))
        
        return fig
    
    def generate_dashboard_summary(self) -> Dict[str, Any]:
        """Generate dashboard summary"""
        try:
            return {
                'total_dashboards': len(self.dashboards),
                'available_templates': len(self.dashboard_templates),
                'dashboard_types': [dt.value for dt in DashboardType],
                'output_path': str(self.output_path),
                'config': self.config,
                'recent_dashboards': [
                    {
                        'dashboard_id': dashboard.dashboard_id,
                        'title': dashboard.title,
                        'type': dashboard.dashboard_type.value,
                        'widgets_count': len(dashboard.widgets),
                        'created_at': dashboard.created_at.isoformat()
                    }
                    for dashboard in list(self.dashboards.values())[-5:]  # Last 5 dashboards
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating dashboard summary: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("PerformanceDashboard module loaded successfully")
    print("Use with DataStorage instance for full functionality")
