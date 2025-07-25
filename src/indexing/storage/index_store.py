"""
Index Storage and Persistence Management

This module handles the storage, backup, and recovery of search indexes,
ensuring data persistence and efficient loading.
"""

import os
import json
import shutil
import logging
import gzip
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
import asyncio
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Configuration for index storage"""
    base_dir: str = "indexes"
    backup_dir: str = "index_backups"
    max_backups: int = 5
    compress_backups: bool = True
    metadata_file: str = "index_metadata.json"
    state_file: str = "index_state.pkl"
    enable_auto_backup: bool = True
    backup_interval_hours: int = 24


@dataclass
class IndexMetadata:
    """Metadata about the stored index"""
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    total_documents: int = 0
    indexed_projects: List[int] = field(default_factory=list)
    index_size_bytes: int = 0
    schema_version: str = "1.0"
    config: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO format
        data['created_at'] = self.created_at.isoformat()
        data['last_modified'] = self.last_modified.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexMetadata':
        """Create from dictionary"""
        # Convert ISO format back to datetime
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'last_modified' in data:
            data['last_modified'] = datetime.fromisoformat(data['last_modified'])
        return cls(**data)


class IndexStore:
    """
    Manages index storage, backup, and recovery.
    
    Features:
    - Atomic index updates
    - Automatic backups
    - Compression support
    - Metadata tracking
    - State persistence
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Initialize index store.
        
        Args:
            config: Storage configuration
        """
        self.config = config or StorageConfig()
        self._ensure_directories()
        self._lock = asyncio.Lock()
        
    def _ensure_directories(self):
        """Ensure required directories exist"""
        Path(self.config.base_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.backup_dir).mkdir(parents=True, exist_ok=True)
    
    async def save_metadata(self, metadata: IndexMetadata):
        """
        Save index metadata to disk.
        
        Args:
            metadata: Index metadata to save
        """
        metadata_path = os.path.join(self.config.base_dir, self.config.metadata_file)
        
        async with self._lock:
            try:
                # Update last modified time
                metadata.last_modified = datetime.now()
                
                # Save to temporary file first for atomic update
                temp_path = f"{metadata_path}.tmp"
                
                async with aiofiles.open(temp_path, 'w') as f:
                    await f.write(json.dumps(metadata.to_dict(), indent=2))
                
                # Atomic rename
                os.replace(temp_path, metadata_path)
                
                logger.info(f"Saved index metadata: {metadata.total_documents} documents")
                
            except Exception as e:
                logger.error(f"Failed to save metadata: {e}")
                # Clean up temp file if it exists
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise
    
    async def load_metadata(self) -> Optional[IndexMetadata]:
        """
        Load index metadata from disk.
        
        Returns:
            Index metadata or None if not found
        """
        metadata_path = os.path.join(self.config.base_dir, self.config.metadata_file)
        
        if not os.path.exists(metadata_path):
            logger.info("No metadata file found")
            return None
        
        try:
            async with aiofiles.open(metadata_path, 'r') as f:
                data = json.loads(await f.read())
            
            metadata = IndexMetadata.from_dict(data)
            logger.info(f"Loaded index metadata: {metadata.total_documents} documents")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return None
    
    async def save_state(self, state: Dict[str, Any]):
        """
        Save index state (tracked content, etc).
        
        Args:
            state: State dictionary to save
        """
        state_path = os.path.join(self.config.base_dir, self.config.state_file)
        
        async with self._lock:
            try:
                # Save to temporary file first
                temp_path = f"{state_path}.tmp"
                
                # Use gzip compression for state files
                if self.config.compress_backups:
                    with gzip.open(temp_path, 'wb') as f:
                        pickle.dump(state, f)
                else:
                    with open(temp_path, 'wb') as f:
                        pickle.dump(state, f)
                
                # Atomic rename
                os.replace(temp_path, state_path)
                
                logger.info(f"Saved index state: {len(state)} entries")
                
            except Exception as e:
                logger.error(f"Failed to save state: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise
    
    async def load_state(self) -> Optional[Dict[str, Any]]:
        """
        Load index state from disk.
        
        Returns:
            State dictionary or None if not found
        """
        state_path = os.path.join(self.config.base_dir, self.config.state_file)
        
        if not os.path.exists(state_path):
            logger.info("No state file found")
            return None
        
        try:
            # Check if file is gzipped
            if state_path.endswith('.gz') or self._is_gzipped(state_path):
                with gzip.open(state_path, 'rb') as f:
                    state = pickle.load(f)
            else:
                with open(state_path, 'rb') as f:
                    state = pickle.load(f)
            
            logger.info(f"Loaded index state: {len(state)} entries")
            return state
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None
    
    def _is_gzipped(self, filepath: str) -> bool:
        """Check if file is gzipped"""
        with open(filepath, 'rb') as f:
            return f.read(2) == b'\x1f\x8b'
    
    async def backup_index(self, index_dir: str = "main") -> Optional[str]:
        """
        Create a backup of the index.
        
        Args:
            index_dir: Name of the index directory to backup
            
        Returns:
            Path to backup file or None on failure
        """
        index_path = os.path.join(self.config.base_dir, index_dir)
        
        if not os.path.exists(index_path):
            logger.warning(f"Index directory not found: {index_path}")
            return None
        
        async with self._lock:
            try:
                # Generate backup filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"index_backup_{timestamp}"
                
                if self.config.compress_backups:
                    backup_path = os.path.join(self.config.backup_dir, f"{backup_name}.tar.gz")
                    
                    # Create compressed archive
                    await self._create_compressed_backup(index_path, backup_path)
                else:
                    backup_path = os.path.join(self.config.backup_dir, backup_name)
                    
                    # Copy directory
                    await self._create_directory_backup(index_path, backup_path)
                
                # Also backup metadata and state
                await self._backup_metadata_and_state(backup_name)
                
                # Clean up old backups
                self._cleanup_old_backups()
                
                logger.info(f"Created index backup: {backup_path}")
                return backup_path
                
            except Exception as e:
                logger.error(f"Failed to create backup: {e}")
                return None
    
    async def _create_compressed_backup(self, source_dir: str, backup_path: str):
        """Create compressed backup using tar.gz"""
        import tarfile
        
        def create_archive():
            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(source_dir, arcname=os.path.basename(source_dir))
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, create_archive)
    
    async def _create_directory_backup(self, source_dir: str, backup_path: str):
        """Create directory backup"""
        def copy_directory():
            shutil.copytree(source_dir, backup_path)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, copy_directory)
    
    async def _backup_metadata_and_state(self, backup_name: str):
        """Backup metadata and state files"""
        metadata_src = os.path.join(self.config.base_dir, self.config.metadata_file)
        state_src = os.path.join(self.config.base_dir, self.config.state_file)
        
        if os.path.exists(metadata_src):
            metadata_dst = os.path.join(self.config.backup_dir, f"{backup_name}_metadata.json")
            shutil.copy2(metadata_src, metadata_dst)
        
        if os.path.exists(state_src):
            state_dst = os.path.join(self.config.backup_dir, f"{backup_name}_state.pkl")
            shutil.copy2(state_src, state_dst)
    
    def _cleanup_old_backups(self):
        """Remove old backups exceeding max_backups limit"""
        if self.config.max_backups <= 0:
            return
        
        # Get all backup files
        backup_files = []
        for file in os.listdir(self.config.backup_dir):
            if file.startswith("index_backup_"):
                file_path = os.path.join(self.config.backup_dir, file)
                backup_files.append((file_path, os.path.getmtime(file_path)))
        
        # Sort by modification time (oldest first)
        backup_files.sort(key=lambda x: x[1])
        
        # Remove excess backups
        while len(backup_files) > self.config.max_backups:
            old_backup, _ = backup_files.pop(0)
            
            # Remove backup and associated files
            if os.path.isdir(old_backup):
                shutil.rmtree(old_backup)
            else:
                os.remove(old_backup)
            
            # Remove associated metadata and state files
            backup_name = os.path.basename(old_backup).split('.')[0]
            for suffix in ['_metadata.json', '_state.pkl']:
                associated_file = os.path.join(self.config.backup_dir, f"{backup_name}{suffix}")
                if os.path.exists(associated_file):
                    os.remove(associated_file)
            
            logger.info(f"Removed old backup: {old_backup}")
    
    async def restore_index(self, backup_path: str, index_dir: str = "main") -> bool:
        """
        Restore index from backup.
        
        Args:
            backup_path: Path to backup file or directory
            index_dir: Target index directory name
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(backup_path):
            logger.error(f"Backup not found: {backup_path}")
            return False
        
        index_path = os.path.join(self.config.base_dir, index_dir)
        
        async with self._lock:
            try:
                # Backup current index if it exists
                if os.path.exists(index_path):
                    temp_backup = f"{index_path}_temp"
                    shutil.move(index_path, temp_backup)
                
                try:
                    if backup_path.endswith('.tar.gz'):
                        # Extract compressed backup
                        await self._extract_compressed_backup(backup_path, self.config.base_dir)
                    else:
                        # Copy directory backup
                        shutil.copytree(backup_path, index_path)
                    
                    # Restore metadata and state
                    await self._restore_metadata_and_state(backup_path)
                    
                    # Remove temporary backup
                    if os.path.exists(temp_backup):
                        shutil.rmtree(temp_backup)
                    
                    logger.info(f"Restored index from: {backup_path}")
                    return True
                    
                except Exception as e:
                    # Restore original index on failure
                    if os.path.exists(temp_backup):
                        if os.path.exists(index_path):
                            shutil.rmtree(index_path)
                        shutil.move(temp_backup, index_path)
                    raise
                
            except Exception as e:
                logger.error(f"Failed to restore index: {e}")
                return False
    
    async def _extract_compressed_backup(self, backup_path: str, target_dir: str):
        """Extract compressed backup"""
        import tarfile
        
        def extract_archive():
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(target_dir)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, extract_archive)
    
    async def _restore_metadata_and_state(self, backup_path: str):
        """Restore metadata and state files from backup"""
        backup_name = os.path.basename(backup_path).split('.')[0]
        
        metadata_src = os.path.join(self.config.backup_dir, f"{backup_name}_metadata.json")
        state_src = os.path.join(self.config.backup_dir, f"{backup_name}_state.pkl")
        
        if os.path.exists(metadata_src):
            metadata_dst = os.path.join(self.config.base_dir, self.config.metadata_file)
            shutil.copy2(metadata_src, metadata_dst)
        
        if os.path.exists(state_src):
            state_dst = os.path.join(self.config.base_dir, self.config.state_file)
            shutil.copy2(state_src, state_dst)
    
    async def get_index_info(self, index_dir: str = "main") -> Dict[str, Any]:
        """
        Get information about stored index.
        
        Args:
            index_dir: Index directory name
            
        Returns:
            Dictionary with index information
        """
        index_path = os.path.join(self.config.base_dir, index_dir)
        
        info = {
            'exists': os.path.exists(index_path),
            'path': index_path,
            'size_bytes': 0,
            'file_count': 0,
            'metadata': None,
            'backups': []
        }
        
        if info['exists']:
            # Calculate size
            total_size = 0
            file_count = 0
            for root, dirs, files in os.walk(index_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
                    file_count += 1
            
            info['size_bytes'] = total_size
            info['file_count'] = file_count
        
        # Load metadata
        metadata = await self.load_metadata()
        if metadata:
            info['metadata'] = metadata.to_dict()
        
        # List available backups
        for file in os.listdir(self.config.backup_dir):
            if file.startswith("index_backup_"):
                backup_path = os.path.join(self.config.backup_dir, file)
                info['backups'].append({
                    'name': file,
                    'path': backup_path,
                    'size_bytes': os.path.getsize(backup_path) if os.path.isfile(backup_path) else 0,
                    'created': datetime.fromtimestamp(os.path.getmtime(backup_path)).isoformat()
                })
        
        return info
    
    async def validate_index(self, index_dir: str = "main") -> Tuple[bool, List[str]]:
        """
        Validate index integrity.
        
        Args:
            index_dir: Index directory name
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        index_path = os.path.join(self.config.base_dir, index_dir)
        issues = []
        
        # Check if index exists
        if not os.path.exists(index_path):
            issues.append("Index directory does not exist")
            return False, issues
        
        # Check for required Whoosh files
        required_files = ['_MAIN_1.toc']  # At least one segment file
        for file in required_files:
            if not any(f.endswith('.toc') for f in os.listdir(index_path)):
                issues.append("No index segment files found")
                break
        
        # Check metadata
        metadata = await self.load_metadata()
        if not metadata:
            issues.append("Metadata file missing or corrupted")
        
        # Check state file
        state = await self.load_state()
        if state is None:
            issues.append("State file missing (not critical)")
        
        # Try to open the index with Whoosh
        try:
            from whoosh import index
            idx = index.open_dir(index_path)
            with idx.searcher() as searcher:
                doc_count = searcher.doc_count()
                if metadata and doc_count != metadata.total_documents:
                    issues.append(f"Document count mismatch: index has {doc_count}, metadata shows {metadata.total_documents}")
        except Exception as e:
            issues.append(f"Failed to open index: {str(e)}")
        
        return len(issues) == 0, issues


async def test_index_store():
    """Test index storage functionality"""
    config = StorageConfig(base_dir="test_indexes", backup_dir="test_backups")
    store = IndexStore(config)
    
    # Test metadata
    print("Testing metadata storage...")
    metadata = IndexMetadata(
        total_documents=1000,
        indexed_projects=[1, 2, 3],
        index_size_bytes=1024 * 1024,
        statistics={'avg_doc_size': 512}
    )
    
    await store.save_metadata(metadata)
    loaded_metadata = await store.load_metadata()
    print(f"Loaded metadata: {loaded_metadata.total_documents} documents")
    
    # Test state
    print("\nTesting state storage...")
    state = {
        'doc1': {'project_id': 1, 'file_path': 'test.py'},
        'doc2': {'project_id': 2, 'file_path': 'readme.md'}
    }
    
    await store.save_state(state)
    loaded_state = await store.load_state()
    print(f"Loaded state: {len(loaded_state)} entries")
    
    # Test backup
    print("\nTesting backup creation...")
    # Create dummy index directory
    os.makedirs("test_indexes/main", exist_ok=True)
    with open("test_indexes/main/test.txt", "w") as f:
        f.write("test content")
    
    backup_path = await store.backup_index()
    print(f"Created backup: {backup_path}")
    
    # Test index info
    print("\nTesting index info...")
    info = await store.get_index_info()
    print(f"Index info: {json.dumps(info, indent=2)}")
    
    # Clean up
    shutil.rmtree("test_indexes", ignore_errors=True)
    shutil.rmtree("test_backups", ignore_errors=True)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        async def show_stats():
            store = IndexStore()
            info = await store.get_index_info()
            print(json.dumps(info, indent=2))
        
        asyncio.run(show_stats())
    else:
        asyncio.run(test_index_store())