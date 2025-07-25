"""
State Recovery System

This module provides comprehensive state persistence and recovery capabilities
for the NIC Chat system, ensuring user data is preserved during errors and
can be restored after recovery actions.
"""

import json
import pickle
import logging
from typing import Any, Dict, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import streamlit as st
import tempfile
import os


@dataclass
class StateSnapshot:
    """Represents a saved state snapshot"""
    snapshot_id: str
    timestamp: datetime
    state_data: Dict[str, Any]
    metadata: Dict[str, Any]
    recovery_points: List[str]


class StateRecoveryManager:
    """Manages state persistence and recovery operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.snapshots: Dict[str, StateSnapshot] = {}
        self.recovery_handlers: Dict[str, Callable] = {}
        self.critical_keys: Set[str] = {
            "messages", "chat_settings", "session_id", "user_preferences"
        }
        self.temp_dir = Path(tempfile.gettempdir()) / "nic_chat_state"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Auto-cleanup old snapshots
        self._cleanup_old_snapshots()
    
    def create_snapshot(
        self, 
        snapshot_id: str,
        include_keys: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StateSnapshot:
        """
        Create a state snapshot for recovery
        
        Args:
            snapshot_id: Unique identifier for the snapshot
            include_keys: Specific keys to include (if None, includes all)
            exclude_keys: Keys to exclude from snapshot
            metadata: Additional metadata to store with snapshot
            
        Returns:
            StateSnapshot object
        """
        try:
            # Determine which keys to include
            if include_keys is None:
                # Include all critical keys and current session state
                include_keys = list(self.critical_keys)
                include_keys.extend([
                    key for key in st.session_state.keys()
                    if not key.startswith('_') and key not in self.critical_keys
                ])
            
            # Apply exclusions
            if exclude_keys:
                include_keys = [key for key in include_keys if key not in exclude_keys]
            
            # Extract state data
            state_data = {}
            for key in include_keys:
                if key in st.session_state:
                    try:
                        # Serialize complex objects carefully
                        value = st.session_state[key]
                        state_data[key] = self._serialize_value(value)
                    except Exception as e:
                        self.logger.warning(f"Failed to serialize key '{key}': {e}")
                        # Store a placeholder indicating serialization failure
                        state_data[key] = {"__serialization_error__": str(e)}
            
            # Create snapshot
            snapshot = StateSnapshot(
                snapshot_id=snapshot_id,
                timestamp=datetime.now(),
                state_data=state_data,
                metadata=metadata or {},
                recovery_points=[self._create_recovery_point()]
            )
            
            # Store snapshot
            self.snapshots[snapshot_id] = snapshot
            self._persist_snapshot(snapshot)
            
            self.logger.info(f"Created state snapshot '{snapshot_id}' with {len(state_data)} keys")
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to create snapshot '{snapshot_id}': {e}")
            raise
    
    def restore_snapshot(
        self, 
        snapshot_id: str,
        partial_restore: bool = False,
        restore_keys: Optional[List[str]] = None
    ) -> bool:
        """
        Restore state from a snapshot
        
        Args:
            snapshot_id: ID of snapshot to restore
            partial_restore: If True, only restore non-conflicting keys
            restore_keys: Specific keys to restore (if None, restores all)
            
        Returns:
            True if restoration was successful
        """
        try:
            # Get snapshot
            snapshot = self.snapshots.get(snapshot_id)
            if not snapshot:
                # Try loading from disk
                snapshot = self._load_snapshot(snapshot_id)
            
            if not snapshot:
                self.logger.error(f"Snapshot '{snapshot_id}' not found")
                return False
            
            # Determine which keys to restore
            keys_to_restore = restore_keys or list(snapshot.state_data.keys())
            
            # Restore state
            restored_count = 0
            failed_keys = []
            
            for key in keys_to_restore:
                if key not in snapshot.state_data:
                    continue
                
                try:
                    value = snapshot.state_data[key]
                    
                    # Skip serialization errors unless explicitly requested
                    if isinstance(value, dict) and "__serialization_error__" in value:
                        self.logger.warning(f"Skipping key '{key}' due to serialization error")
                        continue
                    
                    # Check for conflicts in partial restore mode
                    if partial_restore and key in st.session_state:
                        # Only restore if current value seems invalid/empty
                        current_value = st.session_state[key]
                        if self._is_valid_state_value(current_value):
                            continue
                    
                    # Deserialize and restore
                    restored_value = self._deserialize_value(value)
                    st.session_state[key] = restored_value
                    restored_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to restore key '{key}': {e}")
                    failed_keys.append(key)
            
            self.logger.info(
                f"Restored {restored_count} keys from snapshot '{snapshot_id}'. "
                f"Failed keys: {failed_keys}"
            )
            
            # Update recovery metadata
            self._update_recovery_metadata(snapshot_id, restored_count, failed_keys)
            
            return restored_count > 0
            
        except Exception as e:
            self.logger.error(f"Failed to restore snapshot '{snapshot_id}': {e}")
            return False
    
    def create_auto_snapshot(self, trigger: str = "auto") -> Optional[str]:
        """
        Create an automatic snapshot with timestamp-based ID
        
        Args:
            trigger: What triggered the snapshot creation
            
        Returns:
            Snapshot ID if successful, None otherwise
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_id = f"auto_{trigger}_{timestamp}"
            
            snapshot = self.create_snapshot(
                snapshot_id=snapshot_id,
                metadata={
                    "trigger": trigger,
                    "auto_created": True,
                    "session_id": st.session_state.get("session_id")
                }
            )
            
            return snapshot_id
            
        except Exception as e:
            self.logger.error(f"Failed to create auto snapshot: {e}")
            return None
    
    def get_latest_snapshot(self, max_age_hours: int = 24) -> Optional[StateSnapshot]:
        """Get the most recent snapshot within the specified age limit"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        valid_snapshots = [
            snapshot for snapshot in self.snapshots.values()
            if snapshot.timestamp > cutoff_time
        ]
        
        if not valid_snapshots:
            return None
        
        return max(valid_snapshots, key=lambda s: s.timestamp)
    
    def recover_critical_state(self) -> bool:
        """
        Attempt to recover critical application state
        
        Returns:
            True if recovery was successful
        """
        try:
            # First, try to recover from the latest auto snapshot
            latest = self.get_latest_snapshot()
            if latest and latest.metadata.get("auto_created"):
                self.logger.info("Attempting recovery from latest auto snapshot")
                if self.restore_snapshot(latest.snapshot_id, partial_restore=True):
                    return True
            
            # Try to recover critical keys from any available snapshot
            for snapshot in sorted(self.snapshots.values(), key=lambda s: s.timestamp, reverse=True):
                critical_keys_available = [
                    key for key in self.critical_keys 
                    if key in snapshot.state_data
                ]
                
                if critical_keys_available:
                    self.logger.info(f"Attempting critical recovery from snapshot '{snapshot.snapshot_id}'")
                    if self.restore_snapshot(
                        snapshot.snapshot_id, 
                        partial_restore=True,
                        restore_keys=critical_keys_available
                    ):
                        return True
            
            self.logger.warning("No suitable snapshots found for critical state recovery")
            return False
            
        except Exception as e:
            self.logger.error(f"Critical state recovery failed: {e}")
            return False
    
    def register_recovery_handler(self, state_key: str, handler: Callable[[Any], Any]) -> None:
        """Register a custom recovery handler for a specific state key"""
        self.recovery_handlers[state_key] = handler
    
    def cleanup_snapshots(self, max_age_hours: int = 48, max_count: int = 50) -> int:
        """
        Clean up old snapshots
        
        Args:
            max_age_hours: Maximum age for snapshots in hours
            max_count: Maximum number of snapshots to keep
            
        Returns:
            Number of snapshots cleaned up
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            # Remove old snapshots
            old_snapshots = [
                snapshot_id for snapshot_id, snapshot in self.snapshots.items()
                if snapshot.timestamp < cutoff_time
            ]
            
            # Remove excess snapshots (keep most recent)
            if len(self.snapshots) > max_count:
                sorted_snapshots = sorted(
                    self.snapshots.items(),
                    key=lambda x: x[1].timestamp,
                    reverse=True
                )
                
                excess_snapshots = [
                    snapshot_id for snapshot_id, _ in sorted_snapshots[max_count:]
                ]
                old_snapshots.extend(excess_snapshots)
            
            # Remove snapshots
            cleaned_count = 0
            for snapshot_id in old_snapshots:
                if self._remove_snapshot(snapshot_id):
                    cleaned_count += 1
            
            self.logger.info(f"Cleaned up {cleaned_count} old snapshots")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Snapshot cleanup failed: {e}")
            return 0
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """Get current recovery system status"""
        try:
            latest_snapshot = self.get_latest_snapshot()
            
            return {
                "total_snapshots": len(self.snapshots),
                "latest_snapshot": {
                    "id": latest_snapshot.snapshot_id,
                    "timestamp": latest_snapshot.timestamp.isoformat(),
                    "keys_count": len(latest_snapshot.state_data)
                } if latest_snapshot else None,
                "critical_keys_backed_up": [
                    key for key in self.critical_keys
                    if latest_snapshot and key in latest_snapshot.state_data
                ] if latest_snapshot else [],
                "temp_dir_exists": self.temp_dir.exists(),
                "temp_dir_size": sum(
                    f.stat().st_size for f in self.temp_dir.glob("*.json")
                    if f.is_file()
                ) if self.temp_dir.exists() else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get recovery status: {e}")
            return {"error": str(e)}
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for storage"""
        try:
            # Handle common Streamlit objects
            if hasattr(value, '__dict__') and hasattr(value, '__class__'):
                # Custom objects - try to extract relevant data
                if hasattr(value, 'to_dict'):
                    return {"__type__": "custom_object", "data": value.to_dict()}
                elif hasattr(value, '__dict__'):
                    return {"__type__": "object_dict", "data": value.__dict__}
            
            # Handle standard types
            if isinstance(value, (str, int, float, bool, type(None))):
                return value
            elif isinstance(value, (list, tuple)):
                return [self._serialize_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: self._serialize_value(v) for k, v in value.items()}
            elif isinstance(value, datetime):
                return {"__type__": "datetime", "data": value.isoformat()}
            else:
                # Fall back to string representation
                return {"__type__": "string_repr", "data": str(value)}
            
        except Exception as e:
            raise ValueError(f"Cannot serialize value of type {type(value)}: {e}")
    
    def _deserialize_value(self, value: Any) -> Any:
        """Deserialize a stored value"""
        try:
            if isinstance(value, dict) and "__type__" in value:
                value_type = value["__type__"]
                data = value["data"]
                
                if value_type == "datetime":
                    return datetime.fromisoformat(data)
                elif value_type in ["custom_object", "object_dict"]:
                    # Return as dict - custom reconstruction would require class info
                    return data
                elif value_type == "string_repr":
                    return data
            
            # Handle nested structures
            if isinstance(value, list):
                return [self._deserialize_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: self._deserialize_value(v) for k, v in value.items()}
            
            return value
            
        except Exception as e:
            self.logger.warning(f"Deserialization failed, returning raw value: {e}")
            return value
    
    def _is_valid_state_value(self, value: Any) -> bool:
        """Check if a state value appears to be valid"""
        if value is None:
            return False
        if isinstance(value, (list, dict)) and len(value) == 0:
            return False
        if isinstance(value, str) and value.strip() == "":
            return False
        return True
    
    def _create_recovery_point(self) -> str:
        """Create a recovery point identifier"""
        return f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _persist_snapshot(self, snapshot: StateSnapshot) -> None:
        """Persist snapshot to disk"""
        try:
            file_path = self.temp_dir / f"{snapshot.snapshot_id}.json"
            
            snapshot_data = {
                "snapshot_id": snapshot.snapshot_id,
                "timestamp": snapshot.timestamp.isoformat(),
                "state_data": snapshot.state_data,
                "metadata": snapshot.metadata,
                "recovery_points": snapshot.recovery_points
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.warning(f"Failed to persist snapshot to disk: {e}")
    
    def _load_snapshot(self, snapshot_id: str) -> Optional[StateSnapshot]:
        """Load snapshot from disk"""
        try:
            file_path = self.temp_dir / f"{snapshot_id}.json"
            
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            snapshot = StateSnapshot(
                snapshot_id=data["snapshot_id"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                state_data=data["state_data"],
                metadata=data["metadata"],
                recovery_points=data["recovery_points"]
            )
            
            # Add to memory cache
            self.snapshots[snapshot_id] = snapshot
            return snapshot
            
        except Exception as e:
            self.logger.warning(f"Failed to load snapshot from disk: {e}")
            return None
    
    def _remove_snapshot(self, snapshot_id: str) -> bool:
        """Remove snapshot from memory and disk"""
        try:
            # Remove from memory
            if snapshot_id in self.snapshots:
                del self.snapshots[snapshot_id]
            
            # Remove from disk
            file_path = self.temp_dir / f"{snapshot_id}.json"
            if file_path.exists():
                file_path.unlink()
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to remove snapshot '{snapshot_id}': {e}")
            return False
    
    def _cleanup_old_snapshots(self) -> None:
        """Clean up old snapshots on initialization"""
        try:
            # Load existing snapshots from disk
            for file_path in self.temp_dir.glob("*.json"):
                try:
                    snapshot_id = file_path.stem
                    self._load_snapshot(snapshot_id)
                except Exception as e:
                    self.logger.warning(f"Failed to load snapshot from {file_path}: {e}")
            
            # Clean up old snapshots
            self.cleanup_snapshots()
            
        except Exception as e:
            self.logger.warning(f"Failed to cleanup old snapshots: {e}")
    
    def _update_recovery_metadata(
        self, 
        snapshot_id: str, 
        restored_count: int, 
        failed_keys: List[str]
    ) -> None:
        """Update recovery metadata for tracking"""
        if "state_recovery" not in st.session_state:
            st.session_state["state_recovery"] = {}
        
        st.session_state["state_recovery"]["last_recovery"] = {
            "snapshot_id": snapshot_id,
            "timestamp": datetime.now().isoformat(),
            "restored_count": restored_count,
            "failed_keys": failed_keys
        }


# Global state recovery manager instance
_global_recovery_manager = None


def get_recovery_manager() -> StateRecoveryManager:
    """Get the global state recovery manager instance"""
    global _global_recovery_manager
    if _global_recovery_manager is None:
        _global_recovery_manager = StateRecoveryManager()
    return _global_recovery_manager


def create_error_snapshot(error_context: str = "error") -> Optional[str]:
    """Create a snapshot before handling an error"""
    try:
        recovery_manager = get_recovery_manager()
        return recovery_manager.create_auto_snapshot(f"error_{error_context}")
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to create error snapshot: {e}")
        return None


def attempt_state_recovery() -> bool:
    """Attempt to recover from the most recent valid state"""
    try:
        recovery_manager = get_recovery_manager()
        return recovery_manager.recover_critical_state()
    except Exception as e:
        logging.getLogger(__name__).error(f"State recovery attempt failed: {e}")
        return False