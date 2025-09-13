"""Memory configuration for CSV Agent."""

import os
from pathlib import Path

# Memory database configuration
MEMORY_DB_DIR = Path("./memory")
MEMORY_DB_NAME = "csv_agent_memory.db"
MEMORY_DB_PATH = MEMORY_DB_DIR / MEMORY_DB_NAME

def setup_memory_db():
    """Create memory directory if it doesn't exist."""
    MEMORY_DB_DIR.mkdir(exist_ok=True)
    return str(MEMORY_DB_PATH)

def get_session_id():
    """Generate a simple session ID."""
    import time
    return f"csv_session_{int(time.time())}"
