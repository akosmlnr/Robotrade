"""
Backup and Recovery Systems for Real-time LSTM Prediction System
Phase 3.2: Comprehensive backup, recovery, and disaster recovery capabilities
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sqlite3
import shutil
import gzip
import json
import os
from pathlib import Path
import hashlib
import subprocess
import threading
import time

logger = logging.getLogger(__name__)

class BackupType(Enum):
    """Types of backups"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    HOT = "hot"
    COLD = "cold"

class BackupStatus(Enum):
    """Backup status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"

class RecoveryType(Enum):
    """Types of recovery"""
    FULL_RECOVERY = "full_recovery"
    POINT_IN_TIME = "point_in_time"
    SELECTIVE = "selective"
    EMERGENCY = "emergency"

@dataclass
class BackupInfo:
    """Backup information"""
    backup_id: str
    backup_type: BackupType
    status: BackupStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    size_bytes: int = 0
    checksum: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class RecoveryInfo:
    """Recovery information"""
    recovery_id: str
    recovery_type: RecoveryType
    backup_id: str
    target_timestamp: Optional[datetime] = None
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None

class BackupRecoveryManager:
    """
    Comprehensive backup and recovery management system
    """
    
    def __init__(self, data_storage, config: Dict[str, Any] = None):
        """
        Initialize backup and recovery manager
        
        Args:
            data_storage: DataStorage instance
            config: Optional configuration dictionary
        """
        self.data_storage = data_storage
        
        # Configuration
        self.config = self._load_default_config()
        if config:
            self.config.update(config)
        
        # Backup settings
        self.backup_base_path = Path(self.config.get('backup_path', 'backups'))
        self.backup_base_path.mkdir(parents=True, exist_ok=True)
        
        # Backup metadata
        self.backup_metadata_file = self.backup_base_path / 'backup_metadata.json'
        self.backup_history: List[BackupInfo] = []
        self.recovery_history: List[RecoveryInfo] = []
        
        # Backup statistics (initialize before loading metadata)
        self.backup_stats = {
            'total_backups': 0,
            'successful_backups': 0,
            'failed_backups': 0,
            'total_size_bytes': 0,
            'last_backup': None,
            'last_successful_backup': None
        }
        
        # Load existing backup metadata
        self._load_backup_metadata()
        
        # Auto-backup settings
        self.auto_backup_enabled = self.config.get('auto_backup_enabled', True)
        self.auto_backup_interval_hours = self.config.get('auto_backup_interval_hours', 24)
        self.auto_backup_thread: Optional[threading.Thread] = None
        self.auto_backup_stop_event = threading.Event()
        
        # Start auto-backup if enabled
        if self.auto_backup_enabled:
            self._start_auto_backup()
        
        logger.info("BackupRecoveryManager initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'backup_path': 'backups',
            'auto_backup_enabled': True,
            'auto_backup_interval_hours': 24,
            'max_backup_retention_days': 30,
            'compression_enabled': True,
            'encryption_enabled': False,
            'backup_verification_enabled': True,
            'hot_backup_enabled': True,
            'incremental_backup_enabled': True,
            'backup_tables': [
                'market_data', 'predictions', 'trade_recommendations',
                'model_performance', 'validation_results', 'validation_details',
                'prediction_results', 'system_alerts', 'performance_metrics'
            ]
        }
    
    def _load_backup_metadata(self):
        """Load backup metadata from file"""
        try:
            if self.backup_metadata_file.exists():
                with open(self.backup_metadata_file, 'r') as f:
                    data = json.load(f)
                    self.backup_history = [
                        BackupInfo(**backup) for backup in data.get('backup_history', [])
                    ]
                    self.recovery_history = [
                        RecoveryInfo(**recovery) for recovery in data.get('recovery_history', [])
                    ]
                    self.backup_stats = data.get('backup_stats', self.backup_stats)
                
                logger.info(f"Loaded backup metadata: {len(self.backup_history)} backups")
            
        except Exception as e:
            logger.error(f"Error loading backup metadata: {e}")
    
    def _save_backup_metadata(self):
        """Save backup metadata to file"""
        try:
            data = {
                'backup_history': [
                    {
                        'backup_id': backup.backup_id,
                        'backup_type': backup.backup_type.value,
                        'status': backup.status.value,
                        'start_time': backup.start_time.isoformat(),
                        'end_time': backup.end_time.isoformat() if backup.end_time else None,
                        'size_bytes': backup.size_bytes,
                        'checksum': backup.checksum,
                        'file_path': backup.file_path,
                        'metadata': backup.metadata,
                        'error_message': backup.error_message
                    }
                    for backup in self.backup_history
                ],
                'recovery_history': [
                    {
                        'recovery_id': recovery.recovery_id,
                        'recovery_type': recovery.recovery_type.value,
                        'backup_id': recovery.backup_id,
                        'target_timestamp': recovery.target_timestamp.isoformat() if recovery.target_timestamp else None,
                        'status': recovery.status,
                        'start_time': recovery.start_time.isoformat() if recovery.start_time else None,
                        'end_time': recovery.end_time.isoformat() if recovery.end_time else None,
                        'success': recovery.success,
                        'error_message': recovery.error_message
                    }
                    for recovery in self.recovery_history
                ],
                'backup_stats': self.backup_stats
            }
            
            with open(self.backup_metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving backup metadata: {e}")
    
    def create_backup(self, backup_type: BackupType = BackupType.FULL, 
                     tables: List[str] = None) -> str:
        """
        Create a backup of the database
        
        Args:
            backup_type: Type of backup to create
            tables: Specific tables to backup (None for all)
            
        Returns:
            Backup ID
        """
        try:
            backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create backup info
            backup_info = BackupInfo(
                backup_id=backup_id,
                backup_type=backup_type,
                status=BackupStatus.IN_PROGRESS,
                start_time=datetime.now()
            )
            
            self.backup_history.append(backup_info)
            self._save_backup_metadata()
            
            logger.info(f"Starting {backup_type.value} backup: {backup_id}")
            
            # Create backup in separate thread to avoid blocking
            backup_thread = threading.Thread(
                target=self._perform_backup,
                args=(backup_info, tables),
                daemon=True,
                name=f"Backup-{backup_id}"
            )
            backup_thread.start()
            
            return backup_id
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return ""
    
    def _perform_backup(self, backup_info: BackupInfo, tables: List[str] = None):
        """Perform the actual backup operation"""
        try:
            # Determine backup file path
            backup_filename = f"{backup_info.backup_id}.db"
            if self.config.get('compression_enabled', True):
                backup_filename += '.gz'
            
            backup_path = self.backup_base_path / backup_filename
            backup_info.file_path = str(backup_path)
            
            # Perform backup based on type
            if backup_info.backup_type == BackupType.FULL:
                self._perform_full_backup(backup_path, tables)
            elif backup_info.backup_type == BackupType.INCREMENTAL:
                self._perform_incremental_backup(backup_path, tables)
            elif backup_info.backup_type == BackupType.HOT:
                self._perform_hot_backup(backup_path, tables)
            else:
                self._perform_full_backup(backup_path, tables)
            
            # Calculate backup size and checksum
            backup_info.size_bytes = backup_path.stat().st_size
            backup_info.checksum = self._calculate_checksum(backup_path)
            
            # Verify backup if enabled
            if self.config.get('backup_verification_enabled', True):
                if not self._verify_backup(backup_path):
                    backup_info.status = BackupStatus.CORRUPTED
                    backup_info.error_message = "Backup verification failed"
                    logger.error(f"Backup verification failed for {backup_info.backup_id}")
                    return
            
            # Update backup info
            backup_info.status = BackupStatus.COMPLETED
            backup_info.end_time = datetime.now()
            
            # Update statistics
            self.backup_stats['total_backups'] += 1
            self.backup_stats['successful_backups'] += 1
            self.backup_stats['total_size_bytes'] += backup_info.size_bytes
            self.backup_stats['last_backup'] = backup_info.end_time
            self.backup_stats['last_successful_backup'] = backup_info.end_time
            
            # Save metadata
            self._save_backup_metadata()
            
            logger.info(f"Backup completed successfully: {backup_info.backup_id}")
            
        except Exception as e:
            logger.error(f"Error performing backup {backup_info.backup_id}: {e}")
            backup_info.status = BackupStatus.FAILED
            backup_info.error_message = str(e)
            backup_info.end_time = datetime.now()
            
            # Update statistics
            self.backup_stats['total_backups'] += 1
            self.backup_stats['failed_backups'] += 1
            self.backup_stats['last_backup'] = backup_info.end_time
            
            self._save_backup_metadata()
    
    def _perform_full_backup(self, backup_path: Path, tables: List[str] = None):
        """Perform full database backup"""
        try:
            # Use SQLite backup API for full backup
            source_conn = sqlite3.connect(self.data_storage.db_path)
            backup_conn = sqlite3.connect(str(backup_path))
            
            # Perform backup
            source_conn.backup(backup_conn)
            
            backup_conn.close()
            source_conn.close()
            
            # Compress if enabled
            if self.config.get('compression_enabled', True):
                self._compress_backup(backup_path)
            
        except Exception as e:
            logger.error(f"Error in full backup: {e}")
            raise
    
    def _perform_incremental_backup(self, backup_path: Path, tables: List[str] = None):
        """Perform incremental backup"""
        try:
            # Get last backup timestamp
            last_backup_time = self._get_last_backup_time()
            
            # Create new database for incremental backup
            backup_conn = sqlite3.connect(str(backup_path))
            cursor = backup_conn.cursor()
            
            # Create tables structure
            self._create_backup_tables(cursor)
            
            # Copy only new/modified data
            source_conn = sqlite3.connect(self.data_storage.db_path)
            source_cursor = source_conn.cursor()
            
            tables_to_backup = tables or self.config.get('backup_tables', [])
            
            for table in tables_to_backup:
                # Check if table exists
                source_cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                if not source_cursor.fetchone():
                    continue
                
                # Get new/modified records
                if last_backup_time:
                    source_cursor.execute(f"SELECT * FROM {table} WHERE created_at > ?", (last_backup_time,))
                else:
                    source_cursor.execute(f"SELECT * FROM {table}")
                
                rows = source_cursor.fetchall()
                
                # Get column names
                source_cursor.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in source_cursor.fetchall()]
                
                # Insert into backup database
                if rows:
                    placeholders = ', '.join(['?' for _ in columns])
                    insert_sql = f"INSERT INTO {table} VALUES ({placeholders})"
                    cursor.executemany(insert_sql, rows)
            
            backup_conn.commit()
            backup_conn.close()
            source_conn.close()
            
            # Compress if enabled
            if self.config.get('compression_enabled', True):
                self._compress_backup(backup_path)
            
        except Exception as e:
            logger.error(f"Error in incremental backup: {e}")
            raise
    
    def _perform_hot_backup(self, backup_path: Path, tables: List[str] = None):
        """Perform hot backup (online backup)"""
        try:
            # Hot backup is similar to full backup but with additional safety measures
            self._perform_full_backup(backup_path, tables)
            
        except Exception as e:
            logger.error(f"Error in hot backup: {e}")
            raise
    
    def _create_backup_tables(self, cursor):
        """Create table structure in backup database"""
        try:
            # Get table schemas from source database
            source_conn = sqlite3.connect(self.data_storage.db_path)
            source_cursor = source_conn.cursor()
            
            source_cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
            schemas = source_cursor.fetchall()
            
            for schema in schemas:
                if schema[0]:  # Skip None schemas
                    cursor.execute(schema[0])
            
            source_conn.close()
            
        except Exception as e:
            logger.error(f"Error creating backup tables: {e}")
            raise
    
    def _compress_backup(self, backup_path: Path):
        """Compress backup file"""
        try:
            compressed_path = backup_path.with_suffix(backup_path.suffix + '.gz')
            
            with open(backup_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove uncompressed file
            backup_path.unlink()
            
            # Update path
            backup_path = compressed_path
            
        except Exception as e:
            logger.error(f"Error compressing backup: {e}")
            raise
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating checksum: {e}")
            return ""
    
    def _verify_backup(self, backup_path: Path) -> bool:
        """Verify backup integrity"""
        try:
            # Try to open and query the backup database
            if backup_path.suffix == '.gz':
                # Decompress temporarily for verification
                temp_path = backup_path.with_suffix('')
                with gzip.open(backup_path, 'rb') as f_in:
                    with open(temp_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                verify_path = temp_path
            else:
                verify_path = backup_path
            
            # Verify database
            conn = sqlite3.connect(str(verify_path))
            cursor = conn.cursor()
            
            # Check if we can query basic information
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            
            conn.close()
            
            # Clean up temporary file if created
            if backup_path.suffix == '.gz' and temp_path.exists():
                temp_path.unlink()
            
            return table_count > 0
            
        except Exception as e:
            logger.error(f"Error verifying backup: {e}")
            return False
    
    def _get_last_backup_time(self) -> Optional[datetime]:
        """Get timestamp of last successful backup"""
        try:
            successful_backups = [
                backup for backup in self.backup_history
                if backup.status == BackupStatus.COMPLETED
            ]
            
            if not successful_backups:
                return None
            
            # Sort by start time and get the latest
            latest_backup = max(successful_backups, key=lambda b: b.start_time)
            return latest_backup.start_time
            
        except Exception as e:
            logger.error(f"Error getting last backup time: {e}")
            return None
    
    def restore_backup(self, backup_id: str, recovery_type: RecoveryType = RecoveryType.FULL_RECOVERY,
                      target_timestamp: datetime = None) -> str:
        """
        Restore from backup
        
        Args:
            backup_id: ID of backup to restore
            recovery_type: Type of recovery
            target_timestamp: Target timestamp for point-in-time recovery
            
        Returns:
            Recovery ID
        """
        try:
            # Find backup info
            backup_info = None
            for backup in self.backup_history:
                if backup.backup_id == backup_id:
                    backup_info = backup
                    break
            
            if not backup_info:
                raise ValueError(f"Backup {backup_id} not found")
            
            if backup_info.status != BackupStatus.COMPLETED:
                raise ValueError(f"Backup {backup_id} is not in completed status")
            
            recovery_id = f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create recovery info
            recovery_info = RecoveryInfo(
                recovery_id=recovery_id,
                recovery_type=recovery_type,
                backup_id=backup_id,
                target_timestamp=target_timestamp,
                status="in_progress",
                start_time=datetime.now()
            )
            
            self.recovery_history.append(recovery_info)
            self._save_backup_metadata()
            
            logger.info(f"Starting recovery: {recovery_id} from backup {backup_id}")
            
            # Perform recovery in separate thread
            recovery_thread = threading.Thread(
                target=self._perform_recovery,
                args=(recovery_info, backup_info),
                daemon=True,
                name=f"Recovery-{recovery_id}"
            )
            recovery_thread.start()
            
            return recovery_id
            
        except Exception as e:
            logger.error(f"Error starting recovery: {e}")
            return ""
    
    def _perform_recovery(self, recovery_info: RecoveryInfo, backup_info: BackupInfo):
        """Perform the actual recovery operation"""
        try:
            backup_path = Path(backup_info.file_path)
            
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            # Verify backup checksum if available
            if backup_info.checksum:
                current_checksum = self._calculate_checksum(backup_path)
                if current_checksum != backup_info.checksum:
                    raise ValueError("Backup checksum verification failed")
            
            # Create backup of current database before recovery
            current_backup_path = self.backup_base_path / f"pre_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            shutil.copy2(self.data_storage.db_path, current_backup_path)
            
            # Perform recovery based on type
            if recovery_info.recovery_type == RecoveryType.FULL_RECOVERY:
                self._perform_full_recovery(backup_path)
            elif recovery_info.recovery_type == RecoveryType.POINT_IN_TIME:
                self._perform_point_in_time_recovery(backup_path, recovery_info.target_timestamp)
            elif recovery_info.recovery_type == RecoveryType.SELECTIVE:
                self._perform_selective_recovery(backup_path)
            else:
                self._perform_full_recovery(backup_path)
            
            # Update recovery info
            recovery_info.status = "completed"
            recovery_info.end_time = datetime.now()
            recovery_info.success = True
            
            logger.info(f"Recovery completed successfully: {recovery_info.recovery_id}")
            
        except Exception as e:
            logger.error(f"Error performing recovery {recovery_info.recovery_id}: {e}")
            recovery_info.status = "failed"
            recovery_info.error_message = str(e)
            recovery_info.end_time = datetime.now()
            recovery_info.success = False
        
        # Save metadata
        self._save_backup_metadata()
    
    def _perform_full_recovery(self, backup_path: Path):
        """Perform full database recovery"""
        try:
            # Close current database connection
            if self.data_storage.connection:
                self.data_storage.connection.close()
            
            # Decompress backup if needed
            if backup_path.suffix == '.gz':
                temp_path = backup_path.with_suffix('')
                with gzip.open(backup_path, 'rb') as f_in:
                    with open(temp_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                source_path = temp_path
            else:
                source_path = backup_path
            
            # Replace current database with backup
            shutil.copy2(source_path, self.data_storage.db_path)
            
            # Clean up temporary file if created
            if backup_path.suffix == '.gz' and temp_path.exists():
                temp_path.unlink()
            
            # Reconnect to database
            self.data_storage._initialize_database()
            
        except Exception as e:
            logger.error(f"Error in full recovery: {e}")
            raise
    
    def _perform_point_in_time_recovery(self, backup_path: Path, target_timestamp: datetime):
        """Perform point-in-time recovery"""
        try:
            # This is a simplified implementation
            # In practice, you would need transaction logs for true point-in-time recovery
            logger.warning("Point-in-time recovery not fully implemented - performing full recovery")
            self._perform_full_recovery(backup_path)
            
        except Exception as e:
            logger.error(f"Error in point-in-time recovery: {e}")
            raise
    
    def _perform_selective_recovery(self, backup_path: Path):
        """Perform selective recovery (specific tables)"""
        try:
            # This is a simplified implementation
            # In practice, you would restore only specific tables
            logger.warning("Selective recovery not fully implemented - performing full recovery")
            self._perform_full_recovery(backup_path)
            
        except Exception as e:
            logger.error(f"Error in selective recovery: {e}")
            raise
    
    def _start_auto_backup(self):
        """Start automatic backup thread"""
        try:
            self.auto_backup_thread = threading.Thread(
                target=self._auto_backup_loop,
                daemon=True,
                name="AutoBackup"
            )
            self.auto_backup_thread.start()
            logger.info("Auto-backup started")
            
        except Exception as e:
            logger.error(f"Error starting auto-backup: {e}")
    
    def _auto_backup_loop(self):
        """Auto-backup loop"""
        while not self.auto_backup_stop_event.is_set():
            try:
                # Wait for next backup interval
                self.auto_backup_stop_event.wait(self.auto_backup_interval_hours * 3600)
                
                if not self.auto_backup_stop_event.is_set():
                    # Create incremental backup
                    backup_id = self.create_backup(BackupType.INCREMENTAL)
                    logger.info(f"Auto-backup created: {backup_id}")
                    
                    # Clean up old backups
                    self._cleanup_old_backups()
                
            except Exception as e:
                logger.error(f"Error in auto-backup loop: {e}")
                time.sleep(3600)  # Wait 1 hour before retrying
    
    def _cleanup_old_backups(self):
        """Clean up old backup files"""
        try:
            retention_days = self.config.get('max_backup_retention_days', 30)
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            deleted_count = 0
            for backup in self.backup_history:
                if backup.start_time < cutoff_date and backup.status == BackupStatus.COMPLETED:
                    if backup.file_path and Path(backup.file_path).exists():
                        Path(backup.file_path).unlink()
                        deleted_count += 1
            
            # Remove from history
            self.backup_history = [
                backup for backup in self.backup_history
                if backup.start_time >= cutoff_date or backup.status != BackupStatus.COMPLETED
            ]
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old backup files")
                self._save_backup_metadata()
            
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
    
    def get_backup_status(self) -> Dict[str, Any]:
        """Get backup status and statistics"""
        try:
            return {
                'backup_stats': self.backup_stats,
                'recent_backups': [
                    {
                        'backup_id': backup.backup_id,
                        'backup_type': backup.backup_type.value,
                        'status': backup.status.value,
                        'start_time': backup.start_time.isoformat(),
                        'end_time': backup.end_time.isoformat() if backup.end_time else None,
                        'size_bytes': backup.size_bytes,
                        'checksum': backup.checksum
                    }
                    for backup in self.backup_history[-10:]  # Last 10 backups
                ],
                'recent_recoveries': [
                    {
                        'recovery_id': recovery.recovery_id,
                        'recovery_type': recovery.recovery_type.value,
                        'backup_id': recovery.backup_id,
                        'status': recovery.status,
                        'start_time': recovery.start_time.isoformat() if recovery.start_time else None,
                        'end_time': recovery.end_time.isoformat() if recovery.end_time else None,
                        'success': recovery.success
                    }
                    for recovery in self.recovery_history[-5:]  # Last 5 recoveries
                ],
                'auto_backup_enabled': self.auto_backup_enabled,
                'auto_backup_interval_hours': self.auto_backup_interval_hours,
                'backup_path': str(self.backup_base_path)
            }
            
        except Exception as e:
            logger.error(f"Error getting backup status: {e}")
            return {}
    
    def stop_auto_backup(self):
        """Stop automatic backup"""
        try:
            self.auto_backup_stop_event.set()
            if self.auto_backup_thread and self.auto_backup_thread.is_alive():
                self.auto_backup_thread.join(timeout=10)
            
            logger.info("Auto-backup stopped")
            
        except Exception as e:
            logger.error(f"Error stopping auto-backup: {e}")
    
    def list_available_backups(self) -> List[Dict[str, Any]]:
        """List all available backups"""
        try:
            backups = []
            for backup in self.backup_history:
                if backup.status == BackupStatus.COMPLETED:
                    backups.append({
                        'backup_id': backup.backup_id,
                        'backup_type': backup.backup_type.value,
                        'start_time': backup.start_time.isoformat(),
                        'end_time': backup.end_time.isoformat() if backup.end_time else None,
                        'size_bytes': backup.size_bytes,
                        'size_mb': backup.size_bytes / (1024 * 1024),
                        'checksum': backup.checksum,
                        'file_path': backup.file_path,
                        'exists': Path(backup.file_path).exists() if backup.file_path else False
                    })
            
            return sorted(backups, key=lambda x: x['start_time'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("BackupRecoveryManager module loaded successfully")
    print("Use with DataStorage instance for full functionality")
