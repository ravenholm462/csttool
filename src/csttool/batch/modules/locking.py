import fcntl
import os
import json
import socket
import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

class LockError(Exception):
    """Raised when a lock cannot be acquired."""
    pass

@dataclass
class Lock:
    """Represents an acquired filesystem lock."""
    lock_file: Path
    fd: int

def _acquire_lock(lock_path: Path, name: str = "Batch") -> Lock:
    """
    Acquire an advisory filesystem lock using fcntl.flock().
    
    The flock itself is the source of truth for ownership.
    Metadata in the file is informational only.
    """
    # Ensure parent directory exists
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open file descriptor
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    
    try:
        # Attempt non-blocking exclusive lock
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        
        # Construct error message from metadata if possible
        try:
            metadata = json.loads(lock_path.read_text())
            info = (f"held by PID {metadata.get('pid')} "
                    f"on {metadata.get('hostname')} "
                    f"since {metadata.get('started_at')}")
        except Exception:
            info = "held by another process"
            
        raise LockError(f"{name} lock {lock_path} {info}")
    
    # Write informational metadata for humans
    try:
        metadata = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": datetime.now().isoformat()
        }
        # Seek and truncate to overwrite
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(metadata, indent=2).encode('utf-8'))
    except Exception as e:
        # Failure to write metadata is not fatal as long as we have the flock
        logger.warning(f"Failed to write metadata to lock file {lock_path}: {e}")
        
    return Lock(lock_file=lock_path, fd=fd)

def acquire_batch_lock(output_dir: Path) -> Lock:
    """
    Acquire global batch lock in output root.
    Prevents multiple batch runs in the same output directory.
    """
    return _acquire_lock(output_dir / "batch.lock", name="Batch")

def acquire_subject_lock(subject_dir: Path) -> Lock:
    """
    Acquire per-subject lock in subject output directory.
    Protects against concurrent processing of the same subject.
    """
    return _acquire_lock(subject_dir / ".lock", name="Subject")

def release_lock(lock: Optional[Lock]) -> None:
    """
    Release flock and close file descriptor. 
    Deletes the lock file to keep things clean.
    """
    if lock is None:
        return
        
    try:
        # Release the flock
        fcntl.flock(lock.fd, fcntl.LOCK_UN)
        # Close the FD
        os.close(lock.fd)
        # Remove the file (optional, but cleaner)
        lock.lock_file.unlink(missing_ok=True)
    except Exception as e:
        logger.error(f"Error releasing lock {lock.lock_file}: {e}")
