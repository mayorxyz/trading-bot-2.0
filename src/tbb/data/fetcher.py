"""
Historical OHLCV Fetcher - Bybit REST API
Fetches historical candle data for backtesting and initialization
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import pandas as pd

from tbb.core.config import settings
from tbb.core.logger import get_logger
from tbb.data.store import DataStore

logger = get_logger(__name__)


class BybitFetcher:
    """Fetch historical OHLCV data from Bybit REST API"""
    
    # Bybit v5 API endpoints
    BASE_URL = "https://api.bybit.com"
    
    # Rate limiting: 10 requests per second max
    RATE_LIMIT_CALLS = 10
    RATE_LIMIT_PERIOD = 1.0
    
    # Interval mappings (Bybit expects specific format)
    INTERVAL_MAP = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "6h": "360",
        "12h": "720",
        "1d": "D",
        "1w": "W",
        "1M": "M",
    }
    
    def __init__(self):
        self._store = DataStore()
        self._rate_limiter = asyncio.Semaphore(self.RATE_LIMIT_CALLS)
        self._last_request_time = 0.0
        
    async def _respect_rate_limit(self) -> None:
        """Ensure we don't exceed rate limits"""
        async with self._rate_limiter:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self.RATE_LIMIT_PERIOD / self.RATE_LIMIT_CALLS:
                await asyncio.sleep(self.RATE_LIMIT_PERIOD / self.RATE_LIMIT_CALLS - elapsed)
            self._last_request_time = time.time()
    
    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """
        Fetch historical kline/candlestick data
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            start_time: Start datetime (UTC). If None, fetches most recent candles
            end_time: End datetime (UTC). If None, uses current time
            limit: Max candles to fetch per request (max 200 for Bybit)
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if interval not in self.INTERVAL_MAP:
            raise ValueError(f"Invalid interval: {interval}. Must be one of {list(self.INTERVAL_MAP.keys())}")
        
        category = "linear"  # Perpetual futures
        bybit_interval = self.INTERVAL_MAP[interval]
        
        # Convert times to milliseconds
        start_ms = int(start_time.timestamp() * 1000) if start_time else None
        end_ms = int(end_time.timestamp() * 1000) if end_time else int(datetime.now(timezone.utc).timestamp() * 1000)
        
        all_candles = []
        
        # Paginate through results if more than 'limit' candles needed
        current_end = end_ms
        
        while True:
            await self._respect_rate_limit()
            
            params = {
                "category": category,
                "symbol": symbol,
                "interval": bybit_interval,
                "limit": min(limit, 200),
            }
            
            if start_ms:
                params["start"] = start_ms
            if current_end:
                params["end"] = current_end
            
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(
                        f"{self.BASE_URL}/v5/market/kline",
                        params=params,
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                if data.get("retCode") != 0:
                    logger.error(f"Bybit API error: {data.get('retMsg', 'Unknown error')}")
                    break
                
                candles = data.get("result", {}).get("list", [])
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Check if we have enough data or reached the start
                if len(all_candles) >= limit:
                    break
                
                # Get the oldest candle timestamp to paginate backwards
                oldest_timestamp = int(candles[-1][0])
                if oldest_timestamp <= start_ms if start_ms else 0:
                    break
                
                # Set end to just before the oldest candle we received
                current_end = oldest_timestamp - 1
                
                # Small delay between paginated requests
                await asyncio.sleep(0.1)
                
            except httpx.HTTPError as e:
                logger.error(f"HTTP error fetching klines: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error fetching klines: {e}")
                break
        
        return self._parse_candles(all_candles, symbol, interval)
    
    def _parse_candles(self, candles: list, symbol: str, interval: str) -> pd.DataFrame:
        """
        Parse raw Bybit candle data into DataFrame
        
        Bybit format: [startTime, open, high, low, close, volume, turnover]
        """
        if not candles:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        df = pd.DataFrame(candles, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        
        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        
        # Keep only required columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        
        # Sort by timestamp ascending
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Add metadata
        df["symbol"] = symbol
        df["interval"] = interval
        
        logger.info(f"Parsed {len(df)} candles for {symbol} {interval}")
        return df
    
    async def fetch_and_store(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
    ) -> int:
        """
        Fetch historical data and store in database
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            start_time: Start datetime
            end_time: End datetime (default: now)
        
        Returns:
            Number of candles stored
        """
        logger.info(f"Fetching {interval} data for {symbol} from {start_time} to {end_time or 'now'}")
        
        df = await self.fetch_klines(symbol, interval, start_time, end_time, limit=10000)
        
        if df.empty:
            logger.warning(f"No data fetched for {symbol} {interval}")
            return 0
        
        # Store in database
        stored = await self._store.store_candles(df, symbol, interval)
        
        logger.info(f"Stored {stored} candles for {symbol} {interval} in database")
        return stored
    
    async def get_latest_timestamp(
        self,
        symbol: str,
        interval: str,
    ) -> Optional[datetime]:
        """Get the latest timestamp we have in storage for a symbol/interval"""
        return await self._store.get_latest_candle_timestamp(symbol, interval)
    
    async def fetch_missing_data(
        self,
        symbol: str,
        interval: str,
        lookback_days: int = 30,
    ) -> int:
        """
        Fetch missing data to fill gaps in storage
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            lookback_days: How many days to look back if no data exists
        
        Returns:
            Number of candles fetched
        """
        # Get latest timestamp in storage
        latest_ts = await self.get_latest_timestamp(symbol, interval)
        
        if latest_ts:
            # Fetch from latest timestamp to now
            start_time = latest_ts
            logger.info(f"Fetching missing data for {symbol} {interval} from {start_time}")
        else:
            # No data exists, fetch lookback period
            start_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            start_time = datetime(start_time.year, start_time.month, start_time.day - lookback_days, tzinfo=timezone.utc)
            logger.info(f"No existing data for {symbol} {interval}, fetching from {start_time}")
        
        return await self.fetch_and_store(symbol, interval, start_time)
    
    async def initialize_historical_data(
        self,
        symbols: list[str],
        intervals: list[str],
        lookback_days: int = 100,
    ) -> dict[str, dict[str, int]]:
        """
        Initialize historical data for multiple symbols and intervals
        
        Args:
            symbols: List of symbols to fetch
            intervals: List of intervals to fetch
            lookback_days: Days of historical data to fetch
        
        Returns:
            Dict of {symbol: {interval: count}} with number of candles fetched
        """
        results = {}
        start_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start_time = datetime(start_time.year, start_time.month, start_time.day - lookback_days, tzinfo=timezone.utc)
        
        for symbol in symbols:
            results[symbol] = {}
            for interval in intervals:
                count = await self.fetch_and_store(symbol, interval, start_time)
                results[symbol][interval] = count
                logger.info(f"Initialized {count} {interval} candles for {symbol}")
                await asyncio.sleep(0.2)  # Rate limiting
        
        return results


# Convenience function for quick fetch
async def fetch_historical_ohlcv(
    symbol: str,
    interval: str,
    bars: int = 1000,
) -> pd.DataFrame:
    """
    Quick helper to fetch historical OHLCV data
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Candle interval (1m, 5m, 15m, 1h, 4h, etc.)
        bars: Number of bars to fetch
    
    Returns:
        DataFrame with OHLCV data
    """
    fetcher = BybitFetcher()
    
    # Calculate start time based on interval
    interval_seconds = {
        "1m": 60,
        "3m": 180,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "2h": 7200,
        "4h": 14400,
        "6h": 21600,
        "12h": 43200,
        "1d": 86400,
    }
    
    seconds_per_bar = interval_seconds.get(interval, 3600)
    start_time = datetime.now(timezone.utc)
    start_time = datetime.fromtimestamp(start_time.timestamp() - (bars * seconds_per_bar), tz=timezone.utc)
    
    df = await fetcher.fetch_klines(symbol, interval, start_time=start_time, limit=bars)
    return df


if __name__ == "__main__":
    # Example usage
    async def main():
        fetcher = BybitFetcher()
        
        # Fetch last 200 1-hour candles for BTCUSDT
        df = await fetcher.fetch_klines("BTCUSDT", "1h", limit=200)
        print(f"Fetched {len(df)} candles")
        print(df.tail())
        
        # Fetch and store 4h data for backtesting
        start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        count = await fetcher.fetch_and_store("BTCUSDT", "4h", start)
        print(f"Stored {count} candles")
    
    # asyncio.run(main())
