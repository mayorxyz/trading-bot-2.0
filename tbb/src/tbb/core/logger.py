"""
Structured logging setup for TBB.

Provides configured loggers for main application, trade logs (JSON lines),
and alert logging.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .paths import LOG_FILE_MAIN, LOG_FILE_TRADES, LOG_FILE_ALERTS, ensure_directories
from .config import settings


class JsonTradeFormatter(logging.Formatter):
    """
    JSON line formatter for trade logs.
    Each log record is output as a single JSON line for easy parsing.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        import json
        
        # Build JSON-compatible dict
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, "trade_id"):
            log_entry["trade_id"] = record.trade_id
        if hasattr(record, "symbol"):
            log_entry["symbol"] = record.symbol
        if hasattr(record, "side"):
            log_entry["side"] = record.side
        if hasattr(record, "pnl_r"):
            log_entry["pnl_r"] = record.pnl_r
        if hasattr(record, "exit_reason"):
            log_entry["exit_reason"] = record.exit_reason
        
        return json.dumps(log_entry)


class ColoredConsoleFormatter(logging.Formatter):
    """
    Colored formatter for console output.
    Different colors for different log levels.
    """
    
    # ANSI color codes
    COLORS = {
        "DEBUG": "\x1b[36m",      # Cyan
        "INFO": "\x1b[32m",       # Green
        "WARNING": "\x1b[33m",    # Yellow
        "ERROR": "\x1b[31m",      # Red
        "CRITICAL": "\x1b[35m",   # Magenta
    }
    RESET = "\x1b[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        levelname_colored = f"{color}{record.levelname}{self.RESET}"
        record.levelname = levelname_colored
        return super().format(record)


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: Optional[str] = None,
    json_format: bool = False,
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level (default from settings)
        json_format: Use JSON formatting (for trade logs)
    
    Returns:
        Configured logger instance
    """
    # Ensure directories exist
    ensure_directories()
    
    # Get logging level
    log_level = getattr(logging, (level or settings.LOG_LEVEL).upper())
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    if json_format:
        console_formatter = logging.Formatter(settings.LOG_FORMAT)
    else:
        console_formatter = ColoredConsoleFormatter(settings.LOG_FORMAT)
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        
        if json_format:
            file_formatter = JsonTradeFormatter()
        else:
            file_formatter = logging.Formatter(settings.LOG_FORMAT)
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


# Pre-configured loggers
main_logger = setup_logger("tbb.main", LOG_FILE_MAIN)
trade_logger = setup_logger("tbb.trades", LOG_FILE_TRADES, json_format=True)
alert_logger = setup_logger("tbb.alerts", LOG_FILE_ALERTS)


def get_main_logger() -> logging.Logger:
    """Get the main application logger."""
    return main_logger


def get_trade_logger() -> logging.Logger:
    """Get the trade execution logger (JSON format)."""
    return trade_logger


def get_alert_logger() -> logging.Logger:
    """Get the alert logger."""
    return alert_logger


# Alias for convenience - used by backtest and other modules
def get_logger(name: str = "tbb.main") -> logging.Logger:
    """Get a logger by name, or the main logger."""
    if name == "tbb.main":
        return main_logger
    return setup_logger(name)
