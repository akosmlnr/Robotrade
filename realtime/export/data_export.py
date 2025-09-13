"""
Data Export Functionality for Real-time LSTM Prediction System
Phase 3.2: Comprehensive data export with multiple formats and filtering options
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import json
import csv
import sqlite3
from pathlib import Path
import zipfile
import io
import base64
import shutil
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib

logger = logging.getLogger(__name__)

class ExportFormat(Enum):
    """Supported export formats"""
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    PARQUET = "parquet"
    SQL = "sql"
    XML = "xml"

class ExportType(Enum):
    """Types of data exports"""
    MARKET_DATA = "market_data"
    PREDICTIONS = "predictions"
    VALIDATION_RESULTS = "validation_results"
    PERFORMANCE_METRICS = "performance_metrics"
    TRADE_RECOMMENDATIONS = "trade_recommendations"
    SYSTEM_ALERTS = "system_alerts"
    COMPREHENSIVE = "comprehensive"

class ExportStatus(Enum):
    """Export status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ExportRequest:
    """Export request configuration"""
    export_id: str
    export_type: ExportType
    export_format: ExportFormat
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    filters: Dict[str, Any]
    output_path: str
    compression_enabled: bool = False
    email_enabled: bool = False
    email_recipients: List[str] = None
    status: ExportStatus = ExportStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    file_size_bytes: int = 0
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.email_recipients is None:
            self.email_recipients = []

class DataExporter:
    """
    Comprehensive data export system with multiple formats and delivery options
    """
    
    def __init__(self, data_storage, config: Dict[str, Any] = None):
        """
        Initialize data exporter
        
        Args:
            data_storage: DataStorage instance
            config: Optional configuration dictionary
        """
        self.data_storage = data_storage
        
        # Configuration
        self.config = self._load_default_config()
        if config:
            self.config.update(config)
        
        # Export settings
        self.export_base_path = Path(self.config.get('export_path', 'exports'))
        self.export_base_path.mkdir(parents=True, exist_ok=True)
        
        # Export history and statistics
        self.export_history: List[ExportRequest] = []
        self.export_metadata_file = self.export_base_path / 'export_metadata.json'
        
        # Initialize export statistics first
        self.export_stats = {
            'total_exports': 0,
            'successful_exports': 0,
            'failed_exports': 0,
            'total_data_exported_mb': 0.0,
            'last_export': None
        }
        
        # Load export metadata (this may update export_stats)
        self._load_export_metadata()
        
        logger.info("DataExporter initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'export_path': 'exports',
            'compression_enabled': True,
            'email_enabled': False,
            'email_smtp_server': 'smtp.gmail.com',
            'email_smtp_port': 587,
            'email_username': '',
            'email_password': '',
            'email_from': '',
            'max_export_size_mb': 100,
            'export_timeout_minutes': 30,
            'default_formats': [ExportFormat.CSV, ExportFormat.JSON],
            'table_mappings': {
                ExportType.MARKET_DATA: 'market_data',
                ExportType.PREDICTIONS: 'predictions',
                ExportType.VALIDATION_RESULTS: 'validation_results',
                ExportType.PERFORMANCE_METRICS: 'performance_metrics',
                ExportType.TRADE_RECOMMENDATIONS: 'trade_recommendations',
                ExportType.SYSTEM_ALERTS: 'system_alerts'
            }
        }
    
    def _load_export_metadata(self):
        """Load export metadata from file"""
        try:
            if self.export_metadata_file.exists():
                with open(self.export_metadata_file, 'r') as f:
                    data = json.load(f)
                    self.export_history = [
                        ExportRequest(**export) for export in data.get('export_history', [])
                    ]
                    self.export_stats = data.get('export_stats', self.export_stats)
                
                logger.info(f"Loaded export metadata: {len(self.export_history)} exports")
            
        except Exception as e:
            logger.error(f"Error loading export metadata: {e}")
    
    def _save_export_metadata(self):
        """Save export metadata to file"""
        try:
            data = {
                'export_history': [
                    {
                        'export_id': export.export_id,
                        'export_type': export.export_type.value,
                        'export_format': export.export_format.value,
                        'symbols': export.symbols,
                        'start_date': export.start_date.isoformat(),
                        'end_date': export.end_date.isoformat(),
                        'filters': export.filters,
                        'output_path': export.output_path,
                        'compression_enabled': export.compression_enabled,
                        'email_enabled': export.email_enabled,
                        'email_recipients': export.email_recipients,
                        'status': export.status.value,
                        'created_at': export.created_at.isoformat(),
                        'started_at': export.started_at.isoformat() if export.started_at else None,
                        'completed_at': export.completed_at.isoformat() if export.completed_at else None,
                        'file_size_bytes': export.file_size_bytes,
                        'error_message': export.error_message
                    }
                    for export in self.export_history
                ],
                'export_stats': self.export_stats
            }
            
            with open(self.export_metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving export metadata: {e}")
    
    def create_export(self, export_type: ExportType, export_format: ExportFormat,
                     symbols: List[str], start_date: datetime, end_date: datetime,
                     filters: Dict[str, Any] = None, compression_enabled: bool = None,
                     email_enabled: bool = None, email_recipients: List[str] = None) -> str:
        """
        Create a data export request
        
        Args:
            export_type: Type of data to export
            export_format: Export format
            symbols: List of symbols to export
            start_date: Start date for data
            end_date: End date for data
            filters: Optional filters to apply
            compression_enabled: Enable compression
            email_enabled: Enable email delivery
            email_recipients: Email recipients
            
        Returns:
            Export ID
        """
        try:
            export_id = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Determine output path
            filename = f"{export_id}_{export_type.value}.{export_format.value}"
            if compression_enabled or (compression_enabled is None and self.config.get('compression_enabled', True)):
                filename += '.zip'
            
            output_path = str(self.export_base_path / filename)
            
            # Create export request
            export_request = ExportRequest(
                export_id=export_id,
                export_type=export_type,
                export_format=export_format,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                filters=filters or {},
                output_path=output_path,
                compression_enabled=compression_enabled if compression_enabled is not None else self.config.get('compression_enabled', True),
                email_enabled=email_enabled if email_enabled is not None else self.config.get('email_enabled', False),
                email_recipients=email_recipients or []
            )
            
            self.export_history.append(export_request)
            self._save_export_metadata()
            
            logger.info(f"Created export request: {export_id}")
            return export_id
            
        except Exception as e:
            logger.error(f"Error creating export request: {e}")
            return ""
    
    def execute_export(self, export_id: str) -> bool:
        """
        Execute an export request
        
        Args:
            export_id: Export ID to execute
            
        Returns:
            True if successful
        """
        try:
            # Find export request
            export_request = None
            for export in self.export_history:
                if export.export_id == export_id:
                    export_request = export
                    break
            
            if not export_request:
                logger.error(f"Export request {export_id} not found")
                return False
            
            if export_request.status != ExportStatus.PENDING:
                logger.error(f"Export request {export_id} is not in pending status")
                return False
            
            # Update status
            export_request.status = ExportStatus.IN_PROGRESS
            export_request.started_at = datetime.now()
            self._save_export_metadata()
            
            logger.info(f"Executing export: {export_id}")
            
            # Execute export based on type
            success = False
            if export_request.export_type == ExportType.COMPREHENSIVE:
                success = self._execute_comprehensive_export(export_request)
            else:
                success = self._execute_single_type_export(export_request)
            
            # Update status
            export_request.completed_at = datetime.now()
            if success:
                export_request.status = ExportStatus.COMPLETED
                
                # Calculate file size
                if Path(export_request.output_path).exists():
                    export_request.file_size_bytes = Path(export_request.output_path).stat().st_size
                
                # Update statistics
                self.export_stats['total_exports'] += 1
                self.export_stats['successful_exports'] += 1
                self.export_stats['total_data_exported_mb'] += export_request.file_size_bytes / (1024 * 1024)
                self.export_stats['last_export'] = export_request.completed_at
                
                # Send email if enabled
                if export_request.email_enabled and export_request.email_recipients:
                    self._send_export_email(export_request)
                
                logger.info(f"Export completed successfully: {export_id}")
            else:
                export_request.status = ExportStatus.FAILED
                self.export_stats['total_exports'] += 1
                self.export_stats['failed_exports'] += 1
                logger.error(f"Export failed: {export_id}")
            
            self._save_export_metadata()
            return success
            
        except Exception as e:
            logger.error(f"Error executing export {export_id}: {e}")
            
            # Update export request with error
            for export in self.export_history:
                if export.export_id == export_id:
                    export.status = ExportStatus.FAILED
                    export.error_message = str(e)
                    export.completed_at = datetime.now()
                    break
            
            self.export_stats['total_exports'] += 1
            self.export_stats['failed_exports'] += 1
            self._save_export_metadata()
            return False
    
    def _execute_single_type_export(self, export_request: ExportRequest) -> bool:
        """Execute export for a single data type"""
        try:
            # Get table name
            table_name = self.config['table_mappings'].get(export_request.export_type)
            if not table_name:
                raise ValueError(f"No table mapping for export type {export_request.export_type.value}")
            
            # Query data
            data = self._query_export_data(
                table_name, export_request.symbols, 
                export_request.start_date, export_request.end_date,
                export_request.filters
            )
            
            if data.empty:
                logger.warning(f"No data found for export {export_request.export_id}")
                return True  # Consider empty export as successful
            
            # Export data
            return self._export_data_to_file(data, export_request)
            
        except Exception as e:
            logger.error(f"Error in single type export: {e}")
            return False
    
    def _execute_comprehensive_export(self, export_request: ExportRequest) -> bool:
        """Execute comprehensive export (all data types)"""
        try:
            all_data = {}
            
            # Export each data type
            for export_type, table_name in self.config['table_mappings'].items():
                try:
                    data = self._query_export_data(
                        table_name, export_request.symbols,
                        export_request.start_date, export_request.end_date,
                        export_request.filters
                    )
                    
                    if not data.empty:
                        all_data[export_type.value] = data
                    
                except Exception as e:
                    logger.warning(f"Error exporting {export_type.value}: {e}")
                    continue
            
            if not all_data:
                logger.warning(f"No data found for comprehensive export {export_request.export_id}")
                return True
            
            # Export all data
            return self._export_comprehensive_data(all_data, export_request)
            
        except Exception as e:
            logger.error(f"Error in comprehensive export: {e}")
            return False
    
    def _query_export_data(self, table_name: str, symbols: List[str],
                          start_date: datetime, end_date: datetime,
                          filters: Dict[str, Any]) -> pd.DataFrame:
        """Query data for export"""
        try:
            cursor = self.data_storage.connection.cursor()
            
            # Build query
            query = f"SELECT * FROM {table_name}"
            conditions = []
            params = []
            
            # Add date range condition
            if 'created_at' in [col[1] for col in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()]:
                conditions.append("created_at BETWEEN ? AND ?")
                params.extend([start_date, end_date])
            elif 'timestamp' in [col[1] for col in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()]:
                conditions.append("timestamp BETWEEN ? AND ?")
                params.extend([start_date, end_date])
            
            # Add symbol condition if applicable
            if symbols and 'symbol' in [col[1] for col in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()]:
                placeholders = ', '.join(['?' for _ in symbols])
                conditions.append(f"symbol IN ({placeholders})")
                params.extend(symbols)
            
            # Add custom filters
            for key, value in filters.items():
                if key in [col[1] for col in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()]:
                    conditions.append(f"{key} = ?")
                    params.append(value)
            
            # Execute query
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at DESC" if 'created_at' in [col[1] for col in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()] else ""
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                return pd.DataFrame()
            
            # Convert to DataFrame
            columns = [description[0] for description in cursor.description]
            data = []
            for row in rows:
                data.append(dict(zip(columns, row)))
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error querying export data from {table_name}: {e}")
            return pd.DataFrame()
    
    def _export_data_to_file(self, data: pd.DataFrame, export_request: ExportRequest) -> bool:
        """Export data to file in specified format"""
        try:
            output_path = Path(export_request.output_path)
            
            # Remove compression extension for actual file
            if export_request.compression_enabled and output_path.suffix == '.zip':
                actual_path = output_path.with_suffix('')
            else:
                actual_path = output_path
            
            # Export based on format
            if export_request.export_format == ExportFormat.CSV:
                data.to_csv(actual_path, index=False)
            elif export_request.export_format == ExportFormat.JSON:
                data.to_json(actual_path, orient='records', indent=2)
            elif export_request.export_format == ExportFormat.EXCEL:
                data.to_excel(actual_path, index=False)
            elif export_request.export_format == ExportFormat.PARQUET:
                data.to_parquet(actual_path, index=False)
            elif export_request.export_format == ExportFormat.XML:
                data.to_xml(actual_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {export_request.export_format.value}")
            
            # Compress if enabled
            if export_request.compression_enabled:
                self._compress_export_file(actual_path, output_path)
                actual_path.unlink()  # Remove uncompressed file
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data to file: {e}")
            return False
    
    def _export_comprehensive_data(self, all_data: Dict[str, pd.DataFrame], 
                                 export_request: ExportRequest) -> bool:
        """Export comprehensive data (multiple data types)"""
        try:
            if export_request.export_format == ExportFormat.EXCEL:
                # Export to Excel with multiple sheets
                output_path = Path(export_request.output_path)
                if export_request.compression_enabled and output_path.suffix == '.zip':
                    actual_path = output_path.with_suffix('.xlsx')
                else:
                    actual_path = output_path
                
                with pd.ExcelWriter(actual_path, engine='openpyxl') as writer:
                    for data_type, data in all_data.items():
                        data.to_excel(writer, sheet_name=data_type, index=False)
                
                # Compress if enabled
                if export_request.compression_enabled:
                    self._compress_export_file(actual_path, output_path)
                    actual_path.unlink()
                
            else:
                # Export each data type to separate files and create archive
                temp_dir = Path(export_request.output_path).parent / f"temp_{export_request.export_id}"
                temp_dir.mkdir(exist_ok=True)
                
                try:
                    for data_type, data in all_data.items():
                        file_path = temp_dir / f"{data_type}.{export_request.export_format.value}"
                        self._export_data_to_file(data, ExportRequest(
                            export_id=export_request.export_id,
                            export_type=export_request.export_type,
                            export_format=export_request.export_format,
                            symbols=export_request.symbols,
                            start_date=export_request.start_date,
                            end_date=export_request.end_date,
                            filters=export_request.filters,
                            output_path=str(file_path),
                            compression_enabled=False
                        ))
                    
                    # Create archive
                    if export_request.compression_enabled:
                        with zipfile.ZipFile(export_request.output_path, 'w') as zipf:
                            for file_path in temp_dir.rglob('*'):
                                if file_path.is_file():
                                    zipf.write(file_path, file_path.name)
                    else:
                        # Move files to output directory
                        output_dir = Path(export_request.output_path)
                        output_dir.mkdir(exist_ok=True)
                        for file_path in temp_dir.rglob('*'):
                            if file_path.is_file():
                                shutil.move(str(file_path), str(output_dir / file_path.name))
                
                finally:
                    # Clean up temp directory
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting comprehensive data: {e}")
            return False
    
    def _compress_export_file(self, source_path: Path, target_path: Path):
        """Compress export file"""
        try:
            with zipfile.ZipFile(target_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(source_path, source_path.name)
            
        except Exception as e:
            logger.error(f"Error compressing export file: {e}")
            raise
    
    def _send_export_email(self, export_request: ExportRequest):
        """Send export via email"""
        try:
            if not self.config.get('email_enabled', False):
                return
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.config.get('email_from', '')
            msg['To'] = ', '.join(export_request.email_recipients)
            msg['Subject'] = f"Data Export: {export_request.export_id}"
            
            # Email body
            body = f"""
            Data Export Completed
            
            Export ID: {export_request.export_id}
            Export Type: {export_request.export_type.value}
            Export Format: {export_request.export_format.value}
            Symbols: {', '.join(export_request.symbols)}
            Date Range: {export_request.start_date.date()} to {export_request.end_date.date()}
            File Size: {export_request.file_size_bytes / (1024 * 1024):.2f} MB
            Completed: {export_request.completed_at}
            
            The export file is attached to this email.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach file
            if Path(export_request.output_path).exists():
                with open(export_request.output_path, 'rb') as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {Path(export_request.output_path).name}'
                )
                msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.config.get('email_smtp_server', ''), 
                                self.config.get('email_smtp_port', 587))
            server.starttls()
            server.login(self.config.get('email_username', ''), 
                        self.config.get('email_password', ''))
            text = msg.as_string()
            server.sendmail(self.config.get('email_from', ''), 
                          export_request.email_recipients, text)
            server.quit()
            
            logger.info(f"Export email sent for {export_request.export_id}")
            
        except Exception as e:
            logger.error(f"Error sending export email: {e}")
    
    def get_export_status(self, export_id: str) -> Dict[str, Any]:
        """Get export status"""
        try:
            for export in self.export_history:
                if export.export_id == export_id:
                    return {
                        'export_id': export.export_id,
                        'export_type': export.export_type.value,
                        'export_format': export.export_format.value,
                        'symbols': export.symbols,
                        'start_date': export.start_date.isoformat(),
                        'end_date': export.end_date.isoformat(),
                        'status': export.status.value,
                        'created_at': export.created_at.isoformat(),
                        'started_at': export.started_at.isoformat() if export.started_at else None,
                        'completed_at': export.completed_at.isoformat() if export.completed_at else None,
                        'file_size_bytes': export.file_size_bytes,
                        'file_size_mb': export.file_size_bytes / (1024 * 1024) if export.file_size_bytes > 0 else 0,
                        'output_path': export.output_path,
                        'file_exists': Path(export.output_path).exists(),
                        'error_message': export.error_message
                    }
            
            return {'error': f'Export {export_id} not found'}
            
        except Exception as e:
            logger.error(f"Error getting export status: {e}")
            return {'error': str(e)}
    
    def list_exports(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent exports"""
        try:
            exports = []
            for export in self.export_history[-limit:]:
                exports.append({
                    'export_id': export.export_id,
                    'export_type': export.export_type.value,
                    'export_format': export.export_format.value,
                    'symbols': export.symbols,
                    'status': export.status.value,
                    'created_at': export.created_at.isoformat(),
                    'completed_at': export.completed_at.isoformat() if export.completed_at else None,
                    'file_size_mb': export.file_size_bytes / (1024 * 1024) if export.file_size_bytes > 0 else 0,
                    'file_exists': Path(export.output_path).exists()
                })
            
            return sorted(exports, key=lambda x: x['created_at'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing exports: {e}")
            return []
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """Get export statistics"""
        try:
            return {
                'export_stats': self.export_stats,
                'recent_exports': self.list_exports(10),
                'export_path': str(self.export_base_path),
                'total_exports': len(self.export_history),
                'success_rate': (self.export_stats['successful_exports'] / 
                               max(1, self.export_stats['total_exports'])) * 100
            }
            
        except Exception as e:
            logger.error(f"Error getting export statistics: {e}")
            return {}
    
    def cleanup_old_exports(self, days_back: int = 30) -> bool:
        """Clean up old export files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            deleted_count = 0
            
            for export in self.export_history:
                if export.created_at < cutoff_date:
                    if Path(export.output_path).exists():
                        Path(export.output_path).unlink()
                        deleted_count += 1
            
            # Remove from history
            self.export_history = [
                export for export in self.export_history
                if export.created_at >= cutoff_date
            ]
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old export files")
                self._save_export_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up old exports: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("DataExporter module loaded successfully")
    print("Use with DataStorage instance for full functionality")
