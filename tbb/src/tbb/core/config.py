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
    
    # === Market Structure Parameters ===
    FRACTAL_STRENGTH: int = Field(default=2, description="Fractal strength for swing detection (2=5-bar, 1=3-bar)")
    
    # FVG Settings
    FVG_GAP_ATR_MULT: float = Field(default=0.5, description="FVG gap size must be >= 0.5 * ATR")
    FVG_BODY_ATR_MULT: float = Field(default=0.5, description="FVG middle candle body must be >= 0.5 * ATR")
    
    # Sweep Settings
    SWEEP_WICK_RATIO: float = Field(default=2.0, description="Wick must be 2x body for sweep")
    
    # BOS Settings
    BOS_VOLUME_MULT: float = Field(default=1.5, description="Volume > 1.5x avg for BOS confirmation")
    
    # OB Settings
    OB_SCAN_BACK: int = Field(default=25, description="Bars to scan back for Order Block")
    
    # === EXECUTION MODE ===
    AUTO_EXECUTE: bool = Field(
        default=False,
        description="If True, bot auto-places orders on valid signals. If False, signals only."
    )
    IS_TESTNET: bool = Field(
        default=True,
        description="If True, use exchange testnet endpoints. If False, use mainnet."
    )
    EXECUTION_MODE: str = Field(
        default="paper",  # "paper" | "live"
        description="If 'paper', log intended API calls without sending. If 'live', send real orders."
    )
    KILL_SWITCH: bool = Field(
        default=False,
        description="If True, cancel all open orders and flatten positions immediately."
    )
    
    # === POSITION LIMITS ===
    MAX_OPEN_POSITIONS: int = Field(
        default=4,
        description="Maximum concurrent open positions"
    )
    MAX_DAILY_LOSS_PCT: float = Field(
        default=0.03,
        description="Maximum daily loss percentage (3%) before blocking new entries"
    )
    
    # === RISK PER TRADE ===
    RISK_PER_TRADE_PCT: float = Field(
        default=0.01,
        description="Risk per trade as % of account equity (1%)"
    )
    
    # === PAPER TRACKING CONFIG ===
    MOVE_SL_TO_BE_ON_TP1: bool = Field(
        default=False,
        description="If True, move stop loss to breakeven when TP1 is hit"
    )
    RESOLVE_FULLY_AT_TP1: bool = Field(
        default=False,
        description="If True, fully resolve signal at TP1 (instead of partial)"
    )
    
    # === ADVANCED TRADE MANAGEMENT (2024-2026 Research) ===
    TRAIL_AFTER_TP1: bool = Field(
        default=True,
        description="If True, trail stop after TP1 hit (avoids BE wick-outs)"
    )
    BE_OFFSET_R: float = Field(
        default=0.5,
        description="Lock in +0.5R profit when trailing (avoids exact BE hunt zones)"
    )
    TRAIL_ATR_MULTIPLIER: float = Field(
        default=3.0,
        description="Trailing stop distance as multiple of ATR"
    )
    MAE_HARD_EXIT_R: float = Field(
        default=1.0,
        description="Early exit if MAE > this value and PnL < 0 (93-95% loss probability)"
    )
    MAE_STALL_EXIT_R: float = Field(
        default=0.7,
        description="Early exit if MAE > this and no +0.5R profit after 6 candles"
    )
    MAX_BARS_WITHOUT_PROFIT: int = Field(
        default=6,
        description="Max bars without +0.5R profit before stall exit"
    )
    
    # === DYNAMIC EXPIRY BY TIMEFRAME ===
    SIGNAL_EXPIRY_1M_5M_HOURS: int = Field(default=4, description="Expiry for 1m-5m signals")
    SIGNAL_EXPIRY_15M_1H_HOURS: int = Field(default=24, description="Expiry for 15m-1H signals")
    SIGNAL_EXPIRY_4H_HOURS: int = Field(default=48, description="Expiry for 4H signals")
    SIGNAL_EXPIRY_DAILY_HOURS: int = Field(default=168, description="Expiry for Daily signals (7 days)")
    
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
