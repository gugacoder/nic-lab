"""
Index Update Scheduler

This module provides scheduling functionality for automatic index updates,
including incremental updates and full rebuilds.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import signal

from .indexer import SearchIndexer, IndexConfig
from .storage.index_store import IndexStore, StorageConfig, IndexMetadata

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of scheduled tasks"""
    INCREMENTAL_UPDATE = "incremental_update"
    FULL_REBUILD = "full_rebuild"
    OPTIMIZATION = "optimization"
    BACKUP = "backup"


@dataclass
class ScheduleConfig:
    """Configuration for index scheduling"""
    # Update intervals (in seconds)
    incremental_interval: int = 300  # 5 minutes
    full_rebuild_interval: int = 86400  # 24 hours
    optimization_interval: int = 604800  # 7 days
    backup_interval: int = 3600  # 1 hour
    
    # Time windows for intensive operations
    full_rebuild_hour: int = 2  # 2 AM
    optimization_hour: int = 3  # 3 AM
    
    # Feature flags
    enable_incremental: bool = True
    enable_full_rebuild: bool = True
    enable_optimization: bool = True
    enable_auto_backup: bool = True
    
    # Performance settings
    max_concurrent_tasks: int = 1
    retry_on_failure: bool = True
    max_retries: int = 3
    retry_delay: int = 60  # seconds


@dataclass
class ScheduledTask:
    """Represents a scheduled task"""
    task_type: ScheduleType
    next_run: datetime
    last_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    is_running: bool = False


class IndexScheduler:
    """
    Schedules and manages automatic index updates.
    
    Features:
    - Periodic incremental updates
    - Scheduled full rebuilds
    - Index optimization
    - Automatic backups
    - Error handling and retries
    """
    
    def __init__(
        self,
        indexer: SearchIndexer,
        store: IndexStore,
        config: Optional[ScheduleConfig] = None
    ):
        """
        Initialize the scheduler.
        
        Args:
            indexer: Search indexer instance
            store: Index store instance
            config: Scheduler configuration
        """
        self.indexer = indexer
        self.store = store
        self.config = config or ScheduleConfig()
        
        self._running = False
        self._tasks: Dict[ScheduleType, ScheduledTask] = {}
        self._task_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        
        # Initialize scheduled tasks
        self._init_tasks()
        
        # Task callbacks
        self._task_callbacks: Dict[ScheduleType, Callable] = {
            ScheduleType.INCREMENTAL_UPDATE: self._run_incremental_update,
            ScheduleType.FULL_REBUILD: self._run_full_rebuild,
            ScheduleType.OPTIMIZATION: self._run_optimization,
            ScheduleType.BACKUP: self._run_backup
        }
    
    def _init_tasks(self):
        """Initialize scheduled tasks based on configuration"""
        now = datetime.now()
        
        if self.config.enable_incremental:
            self._tasks[ScheduleType.INCREMENTAL_UPDATE] = ScheduledTask(
                task_type=ScheduleType.INCREMENTAL_UPDATE,
                next_run=now + timedelta(seconds=self.config.incremental_interval)
            )
        
        if self.config.enable_full_rebuild:
            # Schedule for next occurrence of specified hour
            next_rebuild = self._get_next_scheduled_time(self.config.full_rebuild_hour)
            self._tasks[ScheduleType.FULL_REBUILD] = ScheduledTask(
                task_type=ScheduleType.FULL_REBUILD,
                next_run=next_rebuild
            )
        
        if self.config.enable_optimization:
            # Schedule for next occurrence of specified hour
            next_optimization = self._get_next_scheduled_time(
                self.config.optimization_hour,
                days_interval=7
            )
            self._tasks[ScheduleType.OPTIMIZATION] = ScheduledTask(
                task_type=ScheduleType.OPTIMIZATION,
                next_run=next_optimization
            )
        
        if self.config.enable_auto_backup:
            self._tasks[ScheduleType.BACKUP] = ScheduledTask(
                task_type=ScheduleType.BACKUP,
                next_run=now + timedelta(seconds=self.config.backup_interval)
            )
    
    def _get_next_scheduled_time(self, hour: int, days_interval: int = 1) -> datetime:
        """Get next occurrence of specified hour"""
        now = datetime.now()
        next_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        
        # If the hour has passed today, schedule for tomorrow
        if next_time <= now:
            next_time += timedelta(days=1)
        
        # Apply days interval for weekly tasks
        if days_interval > 1:
            days_until = days_interval - (next_time.date() - now.date()).days % days_interval
            next_time += timedelta(days=days_until)
        
        return next_time
    
    async def start(self):
        """Start the scheduler"""
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        self._running = True
        logger.info("Starting index scheduler")
        
        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown)
        
        # Start scheduler loop
        try:
            await self._scheduler_loop()
        except asyncio.CancelledError:
            logger.info("Scheduler cancelled")
        finally:
            self._running = False
            logger.info("Scheduler stopped")
    
    def _handle_shutdown(self):
        """Handle shutdown signal"""
        logger.info("Received shutdown signal")
        self._shutdown_event.set()
    
    async def stop(self):
        """Stop the scheduler gracefully"""
        logger.info("Stopping scheduler...")
        self._shutdown_event.set()
        
        # Wait for running tasks to complete
        async with self._task_lock:
            running_tasks = [
                task for task in self._tasks.values()
                if task.is_running
            ]
        
        if running_tasks:
            logger.info(f"Waiting for {len(running_tasks)} tasks to complete...")
            await asyncio.sleep(5)  # Give tasks time to complete
        
        self._running = False
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self._running and not self._shutdown_event.is_set():
            try:
                # Check for tasks to run
                await self._check_and_run_tasks()
                
                # Sleep for a short interval
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                # Normal timeout, continue loop
                pass
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(60)  # Sleep longer on error
    
    async def _check_and_run_tasks(self):
        """Check for and run due tasks"""
        now = datetime.now()
        
        async with self._task_lock:
            due_tasks = [
                task for task in self._tasks.values()
                if task.next_run <= now and not task.is_running
            ]
        
        # Run due tasks concurrently (up to max_concurrent_tasks)
        if due_tasks:
            semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
            
            async def run_with_semaphore(task: ScheduledTask):
                async with semaphore:
                    await self._run_task(task)
            
            await asyncio.gather(
                *[run_with_semaphore(task) for task in due_tasks],
                return_exceptions=True
            )
    
    async def _run_task(self, task: ScheduledTask):
        """Run a scheduled task"""
        async with self._task_lock:
            task.is_running = True
            task.last_run = datetime.now()
        
        logger.info(f"Running scheduled task: {task.task_type.value}")
        
        try:
            # Get task callback
            callback = self._task_callbacks.get(task.task_type)
            if not callback:
                raise ValueError(f"No callback for task type: {task.task_type}")
            
            # Run the task
            await callback()
            
            # Update task on success
            async with self._task_lock:
                task.run_count += 1
                task.error_count = 0
                task.last_error = None
                task.is_running = False
                
                # Schedule next run
                task.next_run = self._calculate_next_run(task)
            
            logger.info(f"Task completed successfully: {task.task_type.value}")
            
        except Exception as e:
            logger.error(f"Task failed: {task.task_type.value} - {e}")
            
            async with self._task_lock:
                task.error_count += 1
                task.last_error = str(e)
                task.is_running = False
                
                # Retry logic
                if self.config.retry_on_failure and task.error_count < self.config.max_retries:
                    task.next_run = datetime.now() + timedelta(seconds=self.config.retry_delay)
                    logger.info(f"Retrying task in {self.config.retry_delay} seconds")
                else:
                    # Reset error count and schedule normally
                    task.error_count = 0
                    task.next_run = self._calculate_next_run(task)
    
    def _calculate_next_run(self, task: ScheduledTask) -> datetime:
        """Calculate next run time for a task"""
        now = datetime.now()
        
        if task.task_type == ScheduleType.INCREMENTAL_UPDATE:
            return now + timedelta(seconds=self.config.incremental_interval)
        
        elif task.task_type == ScheduleType.FULL_REBUILD:
            return self._get_next_scheduled_time(self.config.full_rebuild_hour)
        
        elif task.task_type == ScheduleType.OPTIMIZATION:
            return self._get_next_scheduled_time(self.config.optimization_hour, days_interval=7)
        
        elif task.task_type == ScheduleType.BACKUP:
            return now + timedelta(seconds=self.config.backup_interval)
        
        else:
            # Default to 1 hour
            return now + timedelta(hours=1)
    
    async def _run_incremental_update(self):
        """Run incremental index update"""
        stats = await self.indexer.update_incremental()
        
        # Save updated metadata
        metadata = await self.store.load_metadata() or IndexMetadata()
        metadata.total_documents = stats.total_documents
        metadata.indexed_projects = list(stats.indexed_projects)
        await self.store.save_metadata(metadata)
        
        # Save index state
        if hasattr(self.indexer, '_indexed_content'):
            await self.store.save_state(self.indexer._indexed_content)
    
    async def _run_full_rebuild(self):
        """Run full index rebuild"""
        # Backup current index first
        if self.config.enable_auto_backup:
            await self.store.backup_index()
        
        # Run full rebuild
        stats = await self.indexer.build_index(force_rebuild=True)
        
        # Save metadata
        metadata = IndexMetadata(
            total_documents=stats.total_documents,
            indexed_projects=list(stats.indexed_projects),
            index_size_bytes=int(stats.index_size_mb * 1024 * 1024)
        )
        await self.store.save_metadata(metadata)
        
        # Save state
        if hasattr(self.indexer, '_indexed_content'):
            await self.store.save_state(self.indexer._indexed_content)
    
    async def _run_optimization(self):
        """Run index optimization"""
        await self.indexer.optimize_index()
        
        # Update metadata
        metadata = await self.store.load_metadata()
        if metadata:
            metadata.last_modified = datetime.now()
            await self.store.save_metadata(metadata)
    
    async def _run_backup(self):
        """Run index backup"""
        backup_path = await self.store.backup_index()
        
        if backup_path:
            logger.info(f"Created scheduled backup: {backup_path}")
        else:
            raise Exception("Backup creation failed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status and task information"""
        status = {
            'running': self._running,
            'tasks': {}
        }
        
        for task_type, task in self._tasks.items():
            status['tasks'][task_type.value] = {
                'next_run': task.next_run.isoformat(),
                'last_run': task.last_run.isoformat() if task.last_run else None,
                'run_count': task.run_count,
                'error_count': task.error_count,
                'last_error': task.last_error,
                'is_running': task.is_running
            }
        
        return status
    
    async def trigger_task(self, task_type: ScheduleType) -> bool:
        """
        Manually trigger a scheduled task.
        
        Args:
            task_type: Type of task to trigger
            
        Returns:
            True if task was triggered, False otherwise
        """
        task = self._tasks.get(task_type)
        if not task:
            logger.warning(f"Task type not found: {task_type}")
            return False
        
        if task.is_running:
            logger.warning(f"Task already running: {task_type}")
            return False
        
        # Run task immediately
        logger.info(f"Manually triggering task: {task_type.value}")
        asyncio.create_task(self._run_task(task))
        
        return True


async def test_scheduler():
    """Test scheduler functionality"""
    # Create test instances
    indexer = SearchIndexer(IndexConfig(index_dir="test_indexes"))
    store = IndexStore(StorageConfig(base_dir="test_indexes"))
    
    # Create scheduler with short intervals for testing
    config = ScheduleConfig(
        incremental_interval=30,  # 30 seconds
        backup_interval=60,  # 1 minute
        enable_full_rebuild=False,  # Disable for testing
        enable_optimization=False  # Disable for testing
    )
    
    scheduler = IndexScheduler(indexer, store, config)
    
    # Start scheduler in background
    scheduler_task = asyncio.create_task(scheduler.start())
    
    try:
        # Let it run for a bit
        print("Scheduler running... (press Ctrl+C to stop)")
        await asyncio.sleep(120)  # Run for 2 minutes
        
    except KeyboardInterrupt:
        print("\nStopping scheduler...")
    
    finally:
        # Stop scheduler
        await scheduler.stop()
        scheduler_task.cancel()
        
        # Show status
        status = scheduler.get_status()
        print(f"\nScheduler status: {status}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(test_scheduler())
    else:
        print("Usage: python -m src.indexing.scheduler test")