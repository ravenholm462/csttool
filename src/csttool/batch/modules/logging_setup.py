import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

class JsonLinesFormatter(logging.Formatter):
    """Formats log records as JSON objects, one per line."""
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        # Add extra attributes if passed via extra={}
        # We look for attributes that aren't standard LogRecord attributes
        standard_attrs = {
            'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
            'funcName', 'levelname', 'levelno', 'lineno', 'module',
            'msecs', 'message', 'msg', 'name', 'pathname', 'process',
            'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName'
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs:
                log_entry[key] = value
                
        return json.dumps(log_entry)

def setup_subject_logger(
    subject_id: str,
    session_id: Optional[str],
    log_path: Path,
    level: str = "INFO",
    verbose: bool = False
) -> logging.Logger:
    """
    Sets up an isolated logger for a specific subject/session attempt.
    
    Outputs structured JSON Lines to the log_path and plain text to stdout if verbose.
    
    Args:
        subject_id: Identifier for the subject
        session_id: Optional identifier for the session
        log_path: Full path to the log file to create
        level: Logging level (default: INFO)
        verbose: If True, also output to stdout
        
    Returns:
        A configured logging.Logger instance
    """
    # Create parent directories for the log file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a unique logger name for this subject run
    logger_name = f"csttool.batch.{subject_id}"
    if session_id:
        logger_name += f".{session_id}"
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Prevent propagation to root logger to maintain isolation
    logger.propagate = False
    
    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # 1. File Handler (JSON Lines)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(JsonLinesFormatter())
    logger.addHandler(file_handler)
    
    # 2. Console Handler (Plain text, optional)
    if verbose:
        console_handler = logging.StreamHandler(sys.stdout)
        fmt = f"[%(levelname)s] {subject_id}"
        if session_id:
            fmt += f"/{session_id}"
        fmt += ": %(message)s"
        console_handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(console_handler)
        
    return logger
