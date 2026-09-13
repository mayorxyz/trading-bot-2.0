"""Survivorship Bias Handling - Point-in-Time Universe Management.

Handles delisted assets and ensures backtests only include assets that were
tradable at each point in time.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import pandas as pd


@dataclass
class AssetInfo:
    """Information about a tradable asset."""
    symbol: str
    name: str
    listed_date: datetime
    delisted_date: Optional[datetime] = None
    is_delisted: bool = False
    delisting_reason: Optional[str] = None  # e.g., "bankruptcy", "merger", "regulatory"
    terminal_value_pct: float = 0.0  # Recovery value if delisted (0.0 = -100%, 0.1 = -90%)


@dataclass
class UniverseState:
    """Current state of the tradable universe."""
    active_assets: list[str] = field(default_factory=list)
    delisted_assets: list[str] = field(default_factory=list)
    asset_info: dict[str, AssetInfo] = field(default_factory=dict)
    
    def get_tradable_symbols(self, at_time: datetime) -> list[str]:
        """Get list of symbols that were tradable at a specific time.
        
        Args:
            at_time: Point in time to check
            
        Returns:
            List of symbols that were listed and not yet delisted
        """
        tradable = []
        for symbol, info in self.asset_info.items():
            if info.listed_date <= at_time:
                if info.delisted_date is None or at_time < info.delisted_date:
                    tradable.append(symbol)
        return tradable
    
    def is_tradable(self, symbol: str, at_time: datetime) -> bool:
        """Check if a symbol was tradable at a specific time."""
        if symbol not in self.asset_info:
            return False
        info = self.asset_info[symbol]
        if info.listed_date > at_time:
            return False
        if info.delisted_date is not None and at_time >= info.delisted_date:
            return False
        return True
    
    def get_delisted_assets(self, from_date: datetime, to_date: datetime) -> list[AssetInfo]:
        """Get assets that delisted within a date range."""
        delisted = []
        for info in self.asset_info.values():
            if info.delisted_date and from_date <= info.delisted_date <= to_date:
                delisted.append(info)
        return delisted


class UniverseManager:
    """Manages point-in-time universe composition.
    
    Prevents survivorship bias by:
    1. Tracking when assets were listed/delisted
    2. Removing delisted assets from the tradable universe
    3. Applying terminal values to forced position closures
    """
    
    def __init__(self):
        self.state = UniverseState()
    
    def add_asset(
        self,
        symbol: str,
        name: str,
        listed_date: datetime,
        delisted_date: Optional[datetime] = None,
        delisting_reason: Optional[str] = None,
        terminal_value_pct: float = 0.0,
    ):
        """Add an asset to the universe.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            name: Human-readable name
            listed_date: When the asset became tradable
            delisted_date: When the asset stopped being tradable (if applicable)
            delisting_reason: Reason for delisting
            terminal_value_pct: Recovery percentage (0.0 = total loss)
        """
        info = AssetInfo(
            symbol=symbol,
            name=name,
            listed_date=listed_date,
            delisted_date=delisted_date,
            is_delisted=delisted_date is not None,
            delisting_reason=delisting_reason,
            terminal_value_pct=terminal_value_pct,
        )
        self.state.asset_info[symbol] = info
        
        # Update active/delisted lists
        if info.is_delisted:
            if symbol not in self.state.delisted_assets:
                self.state.delisted_assets.append(symbol)
        else:
            if symbol not in self.state.active_assets:
                self.state.active_assets.append(symbol)
    
    def load_universe_from_csv(self, filepath: str):
        """Load universe composition from CSV file.
        
        CSV format:
        symbol,name,listed_date,delisted_date,delisting_reason,terminal_value_pct
        BTCUSDT,Bitcoin USDT,2020-01-01,,,0.0
        LUNAUSDT,Terra Luna,2020-01-01,2022-05-13,bankruptcy,0.0
        
        Args:
            filepath: Path to CSV file
        """
        df = pd.read_csv(filepath, parse_dates=['listed_date', 'delisted_date'])
        for _, row in df.iterrows():
            self.add_asset(
                symbol=row['symbol'],
                name=row['name'],
                listed_date=row['listed_date'],
                delisted_date=row.get('delisted_date'),
                delisting_reason=row.get('delisting_reason'),
                terminal_value_pct=row.get('terminal_value_pct', 0.0),
            )
    
    def check_delistings(self, current_time: datetime) -> list[AssetInfo]:
        """Check for assets that delisted at current time.
        
        Args:
            current_time: Current backtest timestamp
            
        Returns:
            List of assets delisting at this time
        """
        delisting_now = []
        for info in self.state.asset_info.values():
            if info.delisted_date and info.delisted_date == current_time:
                delisting_now.append(info)
        return delisting_now
    
    def get_terminal_value(self, symbol: str) -> float:
        """Get terminal value percentage for a delisted asset.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Terminal value as percentage (0.0 = -100%, 1.0 = full value)
        """
        if symbol not in self.state.asset_info:
            return 0.0
        return self.state.asset_info[symbol].terminal_value_pct
    
    def get_universe_snapshot(self, at_time: datetime) -> dict:
        """Get universe composition at a point in time.
        
        Args:
            at_time: Point in time
            
        Returns:
            Dict with active_symbols, delisted_before, delisted_after counts
        """
        tradable = self.state.get_tradable_symbols(at_time)
        
        delisted_before = sum(
            1 for info in self.state.asset_info.values()
            if info.delisted_date and info.delisted_date < at_time
        )
        
        delisted_after = sum(
            1 for info in self.state.asset_info.values()
            if info.delisted_date and info.delisted_date >= at_time
        )
        
        return {
            "timestamp": at_time.isoformat(),
            "active_symbols": tradable,
            "active_count": len(tradable),
            "delisted_before": delisted_before,
            "delisted_after": delisted_after,
            "total_universe_size": len(self.state.asset_info),
        }
