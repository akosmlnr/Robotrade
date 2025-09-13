"""
Update Scheduler for Real-time LSTM Prediction System
Phase 3.1: 15-minute update scheduler with prediction validation workflow
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
import threading
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class UpdateStatus(Enum):
    """Status of update operations"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class UpdateTask:
    """Represents a scheduled update task"""
    task_id: str
    symbol: str
    scheduled_time: datetime
    status: UpdateStatus
    priority: int = 1  # 1=high, 2=medium, 3=low
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class UpdateScheduler:
    """
    15-minute update scheduler for real-time prediction system
    """
    
    def __init__(self, prediction_engine, data_storage, validation_workflow):
        """
        Initialize the update scheduler
        
        Args:
            prediction_engine: PredictionEngine instance
            data_storage: DataStorage instance
            validation_workflow: PredictionValidationWorkflow instance
        """
        self.prediction_engine = prediction_engine
        self.data_storage = data_storage
        self.validation_workflow = validation_workflow
        
        # Scheduler configuration
        self.update_interval_minutes = 15
        self.max_concurrent_tasks = 5
        self.task_timeout_seconds = 300  # 5 minutes
        
        # Task management
        self.scheduled_tasks: Dict[str, UpdateTask] = {}
        self.running_tasks: Dict[str, UpdateTask] = {}
        self.completed_tasks: List[UpdateTask] = []
        self.failed_tasks: List[UpdateTask] = []
        
        # Threading and async
        self.scheduler_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.stop_event = threading.Event()
        
        # Callbacks
        self.on_task_completed: Optional[Callable] = None
        self.on_task_failed: Optional[Callable] = None
        self.on_system_alert: Optional[Callable] = None
        
        # Performance tracking
        self.total_tasks_executed = 0
        self.successful_tasks = 0
        self.failed_tasks_count = 0
        self.average_execution_time = 0.0
        
        logger.info("UpdateScheduler initialized")
    
    def start_scheduler(self, symbols: List[str]) -> bool:
        """
        Start the 15-minute update scheduler
        
        Args:
            symbols: List of stock symbols to schedule updates for
            
        Returns:
            True if scheduler started successfully
        """
        try:
            if self.is_running:
                logger.warning("Scheduler is already running")
                return False
            
            self.is_running = True
            self.stop_event.clear()
            
            # Schedule initial tasks for all symbols
            for symbol in symbols:
                self.schedule_next_update(symbol)
            
            # Start scheduler thread
            self.scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                daemon=True,
                name="UpdateScheduler"
            )
            self.scheduler_thread.start()
            
            logger.info(f"Update scheduler started for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            self.is_running = False
            return False
    
    def stop_scheduler(self) -> bool:
        """
        Stop the update scheduler
        
        Returns:
            True if scheduler stopped successfully
        """
        try:
            if not self.is_running:
                logger.warning("Scheduler is not running")
                return False
            
            self.is_running = False
            self.stop_event.set()
            
            # Wait for scheduler thread to finish
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=10)
            
            logger.info("Update scheduler stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop scheduler: {e}")
            return False
    
    def schedule_next_update(self, symbol: str, priority: int = 1) -> str:
        """
        Schedule the next update for a symbol
        
        Args:
            symbol: Stock symbol
            priority: Task priority (1=high, 2=medium, 3=low)
            
        Returns:
            Task ID of the scheduled update
        """
        try:
            # Calculate next update time (next 15-minute interval)
            now = datetime.now()
            minutes_since_hour = now.minute
            next_interval = ((minutes_since_hour // self.update_interval_minutes) + 1) * self.update_interval_minutes
            
            if next_interval >= 60:
                # Next hour
                scheduled_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            else:
                # Same hour
                scheduled_time = now.replace(minute=next_interval, second=0, microsecond=0)
            
            # Create task
            task_id = f"{symbol}_{scheduled_time.strftime('%Y%m%d_%H%M')}"
            task = UpdateTask(
                task_id=task_id,
                symbol=symbol,
                scheduled_time=scheduled_time,
                status=UpdateStatus.PENDING,
                priority=priority
            )
            
            self.scheduled_tasks[task_id] = task
            
            logger.info(f"Scheduled update for {symbol} at {scheduled_time}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to schedule update for {symbol}: {e}")
            return ""
    
    def _scheduler_loop(self):
        """Main scheduler loop running in separate thread"""
        logger.info("Scheduler loop started")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                current_time = datetime.now()
                
                # Check for tasks ready to execute
                ready_tasks = self._get_ready_tasks(current_time)
                
                # Execute ready tasks (respecting concurrency limit)
                for task in ready_tasks:
                    if len(self.running_tasks) < self.max_concurrent_tasks:
                        self._execute_task(task)
                
                # Check for completed/failed tasks
                self._check_running_tasks()
                
                # Clean up old completed tasks
                self._cleanup_old_tasks()
                
                # Sleep for 30 seconds before next check
                self.stop_event.wait(30)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                self.stop_event.wait(60)  # Wait longer on error
        
        logger.info("Scheduler loop stopped")
    
    def _get_ready_tasks(self, current_time: datetime) -> List[UpdateTask]:
        """Get tasks that are ready to execute"""
        ready_tasks = []
        
        for task in list(self.scheduled_tasks.values()):
            if (task.status == UpdateStatus.PENDING and 
                task.scheduled_time <= current_time):
                ready_tasks.append(task)
        
        # Sort by priority and scheduled time
        ready_tasks.sort(key=lambda t: (t.priority, t.scheduled_time))
        
        return ready_tasks
    
    def _execute_task(self, task: UpdateTask):
        """Execute a scheduled task"""
        try:
            task.status = UpdateStatus.RUNNING
            task.started_at = datetime.now()
            self.running_tasks[task.task_id] = task
            
            # Remove from scheduled tasks
            if task.task_id in self.scheduled_tasks:
                del self.scheduled_tasks[task.task_id]
            
            # Execute in separate thread to avoid blocking
            execution_thread = threading.Thread(
                target=self._run_prediction_task,
                args=(task,),
                daemon=True,
                name=f"Task-{task.symbol}"
            )
            execution_thread.start()
            
            logger.info(f"Started execution of task {task.task_id}")
            
        except Exception as e:
            logger.error(f"Failed to execute task {task.task_id}: {e}")
            task.status = UpdateStatus.FAILED
            task.error_message = str(e)
            self.failed_tasks.append(task)
    
    def _run_prediction_task(self, task: UpdateTask):
        """Run the actual prediction task"""
        try:
            start_time = time.time()
            
            # Get model data for symbol
            model_data = self.prediction_engine.model_manager.get_model_data(task.symbol)
            if not model_data:
                raise Exception(f"No model data available for {task.symbol}")
            
            # Generate prediction
            prediction_result = self.prediction_engine.generate_weekly_prediction(
                task.symbol, model_data
            )
            
            if not prediction_result:
                raise Exception(f"Failed to generate prediction for {task.symbol}")
            
            # Validate prediction
            validation_result = self.validation_workflow.validate_prediction(
                task.symbol, prediction_result
            )
            
            # Store results
            self.data_storage.store_prediction_result(
                symbol=task.symbol,
                prediction_result=prediction_result,
                validation_result=validation_result,
                task_id=task.task_id
            )
            
            # Update task status
            task.status = UpdateStatus.COMPLETED
            task.completed_at = datetime.now()
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self.total_tasks_executed += 1
            self.successful_tasks += 1
            self._update_average_execution_time(execution_time)
            
            # Move to completed tasks
            self.completed_tasks.append(task)
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            
            # Schedule next update
            self.schedule_next_update(task.symbol)
            
            # Call completion callback
            if self.on_task_completed:
                self.on_task_completed(task, prediction_result, validation_result)
            
            logger.info(f"Completed task {task.task_id} in {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            self._handle_task_failure(task, str(e))
    
    def _handle_task_failure(self, task: UpdateTask, error_message: str):
        """Handle task failure with retry logic"""
        task.error_message = error_message
        task.retry_count += 1
        
        if task.retry_count <= task.max_retries:
            # Retry the task
            task.status = UpdateStatus.PENDING
            task.scheduled_time = datetime.now() + timedelta(minutes=5)  # Retry in 5 minutes
            self.scheduled_tasks[task.task_id] = task
            
            logger.info(f"Scheduling retry {task.retry_count}/{task.max_retries} for task {task.task_id}")
        else:
            # Max retries exceeded
            task.status = UpdateStatus.FAILED
            self.failed_tasks.append(task)
            self.failed_tasks_count += 1
            
            # Schedule next update despite failure
            self.schedule_next_update(task.symbol)
            
            # Call failure callback
            if self.on_task_failed:
                self.on_task_failed(task, error_message)
            
            # Send alert
            if self.on_system_alert:
                self.on_system_alert(
                    f"Task {task.task_id} failed after {task.max_retries} retries: {error_message}",
                    "TASK_FAILURE"
                )
        
        # Remove from running tasks
        if task.task_id in self.running_tasks:
            del self.running_tasks[task.task_id]
    
    def _check_running_tasks(self):
        """Check for timeout in running tasks"""
        current_time = datetime.now()
        timed_out_tasks = []
        
        for task in list(self.running_tasks.values()):
            if task.started_at:
                elapsed = (current_time - task.started_at).total_seconds()
                if elapsed > self.task_timeout_seconds:
                    timed_out_tasks.append(task)
        
        for task in timed_out_tasks:
            logger.warning(f"Task {task.task_id} timed out after {self.task_timeout_seconds}s")
            self._handle_task_failure(task, "Task execution timeout")
    
    def _cleanup_old_tasks(self):
        """Clean up old completed and failed tasks"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Clean completed tasks
        self.completed_tasks = [
            task for task in self.completed_tasks 
            if task.completed_at and task.completed_at > cutoff_time
        ]
        
        # Clean failed tasks
        self.failed_tasks = [
            task for task in self.failed_tasks 
            if task.created_at and task.created_at > cutoff_time
        ]
    
    def _update_average_execution_time(self, execution_time: float):
        """Update average execution time"""
        if self.total_tasks_executed == 1:
            self.average_execution_time = execution_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_execution_time = (alpha * execution_time + 
                                         (1 - alpha) * self.average_execution_time)
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        return {
            'is_running': self.is_running,
            'scheduled_tasks': len(self.scheduled_tasks),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'total_executed': self.total_tasks_executed,
            'successful_tasks': self.successful_tasks,
            'failed_tasks_count': self.failed_tasks_count,
            'success_rate': (self.successful_tasks / max(1, self.total_tasks_executed)) * 100,
            'average_execution_time': self.average_execution_time,
            'next_scheduled_tasks': [
                {
                    'task_id': task.task_id,
                    'symbol': task.symbol,
                    'scheduled_time': task.scheduled_time.isoformat(),
                    'priority': task.priority
                }
                for task in sorted(self.scheduled_tasks.values(), 
                                 key=lambda t: t.scheduled_time)[:5]
            ]
        }
    
    def force_update(self, symbol: str, priority: int = 1) -> str:
        """
        Force an immediate update for a symbol
        
        Args:
            symbol: Stock symbol
            priority: Task priority
            
        Returns:
            Task ID of the forced update
        """
        try:
            task_id = f"{symbol}_forced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            task = UpdateTask(
                task_id=task_id,
                symbol=symbol,
                scheduled_time=datetime.now(),
                status=UpdateStatus.PENDING,
                priority=priority
            )
            
            self.scheduled_tasks[task_id] = task
            
            logger.info(f"Forced update scheduled for {symbol}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to force update for {symbol}: {e}")
            return ""
    
    def pause_symbol(self, symbol: str) -> bool:
        """Pause updates for a specific symbol"""
        try:
            # Remove scheduled tasks for symbol
            tasks_to_remove = [
                task_id for task_id, task in self.scheduled_tasks.items()
                if task.symbol == symbol
            ]
            
            for task_id in tasks_to_remove:
                del self.scheduled_tasks[task_id]
            
            logger.info(f"Paused updates for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause updates for {symbol}: {e}")
            return False
    
    def resume_symbol(self, symbol: str) -> bool:
        """Resume updates for a specific symbol"""
        try:
            # Schedule next update
            self.schedule_next_update(symbol)
            logger.info(f"Resumed updates for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume updates for {symbol}: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("UpdateScheduler module loaded successfully")
    print("Use with PredictionEngine, DataStorage, and ValidationSystem instances for full functionality")
