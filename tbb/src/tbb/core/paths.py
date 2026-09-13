"""
Core path constants for TBB (Trading Bot Backbone).

All file paths and database locations are defined here.
Never hardcode paths elsewhere in the codebase.
"""

from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # /workspace/tbb

# Source directory
SRC_DIR = BASE_DIR / "src" / "tbb"

# Data directory - databases and stored data
DATA_DIR = BASE_DIR / "data"

# Database paths
LIVE_STATE_DB = DATA_DIR / "live_state.db"      # Live candle state, indicators, regime
TRADES_DB = DATA_DIR / "trades.db"              # Trade execution logs
ANALYSIS_RUNS_DB = DATA_DIR / "analysis_runs.db"  # Backtest/analysis results
TRACKING_DB = DATA_DIR / "tracking.db"          # Signal tracking and resolution

def get_tracking_db_path() -> str:
    """Get path to tracking database."""
    return str(TRACKING_DB)

# Log directory
LOG_DIR = BASE_DIR / "logs"

# Log file paths
LOG_FILE_MAIN = LOG_DIR / "tbb_main.log"        # Main application log
LOG_FILE_TRADES = LOG_DIR / "tbb_trades.log"    # Trade execution log (JSON lines)
LOG_FILE_ALERTS = LOG_DIR / "tbb_alerts.log"    # Alert trigger log

# Data cache directory (for historical OHLCV caching)
CACHE_DIR = BASE_DIR / "cache"
OHLCV_CACHE_DIR = CACHE_DIR / "ohlcv"

# Ensure directories exist
def ensure_directories() -> None:
    """Create all required directories if they don't exist."""
    for directory in [DATA_DIR, LOG_DIR, CACHE_DIR, OHLCV_CACHE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
