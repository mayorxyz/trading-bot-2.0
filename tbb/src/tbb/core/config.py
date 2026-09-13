"""
Configuration management for TBB.

Loads settings from .env file via python-dotenv.
All configuration values are accessed through the Settings class.
"""

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

# Load .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # === Exchange Configuration ===
    BYBIT_API_KEY: str = Field(default="test_api_key", description="Bybit API key")
    BYBIT_API_SECRET: str = Field(default="test_api_secret", description="Bybit API secret")
    BYBIT_TESTNET: bool = Field(default=True, description="Use Bybit testnet (True) or mainnet (False)")
    
    # === Trading Configuration ===
    DEFAULT_LEVERAGE: int = Field(default=10, ge=1, le=100, description="Default leverage for positions")
    ACCOUNT_RISK_PCT: float = Field(default=0.01, description="Risk per trade as % of account (0.01 = 1%)")
    MAX_CONCURRENT_POSITIONS: int = Field(default=4, description="Maximum concurrent open positions")
    MAX_PORTFOLIO_HEAT: float = Field(default=0.06, description="Maximum total open risk (6%)")
    
    # === Symbols to Trade ===
    TRADING_SYMBOLS: str = Field(
        default="BTCUSDT,ETHUSDT,SOLUSDT",
        description="Comma-separated list of symbols to trade"
    )
    
    # === Timeframes ===
    HTF_TIMEFRAME: str = Field(default="4h", description="Higher timeframe for regime and OB/FVG")
    LTF_TIMEFRAME: str = Field(default="15m", description="Lower timeframe for entry signals")
    
    # === Database ===
    DB_PATH: str = Field(default="./data/live_state.db", description="Path to SQLite database")
    
    # === Telegram Alerts ===
    TELEGRAM_BOT_TOKEN: Optional[str] = Field(default=None, description="Telegram bot token for alerts")
    TELEGRAM_CHAT_ID: Optional[str] = Field(default=None, description="Telegram chat ID for alerts")
    TELEGRAM_ALERT_LEVEL: str = Field(
        default="WARNING",
        description="Minimum alert level: CRITICAL, WARNING, INFO"
    )
    
    # === Logging ===
    LOG_LEVEL: str = Field(default="INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    LOG_FORMAT: str = Field(
        default="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        description="Log format string"
    )
    
    # === Risk Management ===
    DAILY_DD_LIMIT: float = Field(default=-0.03, description="Daily drawdown limit (-3%)")
    CONSECUTIVE_LOSS_LIMIT: int = Field(default=8, description="Consecutive losses before pause")
    PORTFOLIO_DD_LIMIT: float = Field(default=-0.15, description="Portfolio drawdown halt limit (-15%)")
    SLIPPAGE_THRESHOLD: float = Field(default=0.30, description="Slippage threshold (30% worse than expected)")
    
    # === Signal Thresholds ===
    MIN_CONFLUENCE_SCORE: float = Field(default=0.6, description="Minimum confluence score to enter (0.6)")
    MIN_RISK_REWARD: float = Field(default=1.5, description="Minimum R:R ratio to enter (1.5)")
    FRESH_FVG_MAX_AGE_BARS_1H: int = Field(default=72, description="Max age for FVG on 1H (72 bars)")
    FRESH_FVG_MAX_AGE_BARS_15M: int = Field(default=48, description="Max age for FVG on 15m (48 bars)")
    MAX_OB_TOUCHES: int = Field(default=3, description="Max touches before OB invalidated")
    
    # === EXECUTION MODE ===
    AUTO_EXECUTE: bool = Field(
        default=False,
        description="If True, bot auto-places orders on valid signals. If False, signals only."
    )
    
    # === Performance Monitoring ===
    METRICS_ROLLING_WINDOW: int = Field(default=100, description="Rolling window for metrics (100 trades)")
    SHARPE_WINDOW_DAYS: int = Field(default=60, description="Sharpe ratio calculation window (60 days)")
    CORRELATION_WINDOW_DAYS: int = Field(default=90, description="Correlation matrix window (90 days)")
    
    # === Macro Event Gate ===
    MACRO_EVENT_BLACKOUT_MIN_BEFORE: int = Field(default=30, description="Minutes before macro event to block entries")
    MACRO_EVENT_BLACKOUT_MIN_AFTER: int = Field(default=10, description="Minutes after macro event to block entries")
    
    # === API Rate Limits ===
    API_ERROR_RATE_LIMIT: float = Field(default=0.03, description="API error rate threshold (3%)")
    FILL_RATE_MIN: float = Field(default=0.80, description="Minimum fill rate (80%)")
    
    @property
    def trading_symbols(self) -> list[str]:
        """Parse trading symbols from comma-separated string."""
        return [s.strip() for s in self.TRADING_SYMBOLS.split(",")]
    
    @property
    def bybit_base_url(self) -> str:
        """Get Bybit API base URL based on testnet setting."""
        if self.BYBIT_TESTNET:
            return "https://api-testnet.bybit.com"
        return "https://api.bybit.com"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience access
settings = get_settings()
