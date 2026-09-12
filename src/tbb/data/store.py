"""
SQLite storage layer for TBB.
Tables: candles, live_state, trades, signals, metrics_snapshots

Schema designed to migrate to TimescaleDB (hypertables on timestamp).
Uses aiosqlite for async operations.
"""

import aiosqlite
import json
from datetime import datetime, timezone
from typing import Optional, Any
from pathlib import Path

from tbb.core.paths import DB_PATHS
from tbb.core.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

CANDLES_TABLE = """
CREATE TABLE IF NOT EXISTS candles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp INTEGER NOT NULL,  -- Unix timestamp (ms)
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    turnover REAL DEFAULT 0,
    trades_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timeframe, timestamp)
);
"""

CANDLES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf_ts 
ON candles(symbol, timeframe, timestamp);
"""

LIVE_STATE_TABLE = """
CREATE TABLE IF NOT EXISTS live_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    last_candle_ts INTEGER,
    last_1m_ts INTEGER,
    last_15m_ts INTEGER,
    last_1h_ts INTEGER,
    last_4h_ts INTEGER,
    current_regime TEXT,  -- TRENDING_UP | TRENDING_DOWN | RANGING | HIGH_VOLATILITY | LOW_VOLATILITY
    adx_value REAL,
    hurst_value REAL,
    atr_percentile REAL,
    funding_rate REAL,
    open_interest REAL,
    cvd_value REAL,
    obi_value REAL,
    active_fvgs TEXT,  -- JSON array of FVG dicts
    active_obs TEXT,   -- JSON array of OB dicts
    circuit_breaker_status TEXT DEFAULT 'ACTIVE',  -- ACTIVE | PAUSED | HALTED
    pause_reason TEXT,
    pause_until INTEGER,  -- Unix timestamp when pause expires
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT NOT NULL UNIQUE,  -- UUID
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,  -- LONG | SHORT
    status TEXT NOT NULL,  -- OPEN | CLOSED | CANCELLED
    entry_price_expected REAL,
    entry_price_actual REAL,
    exit_price_actual REAL,
    position_size_notional REAL,
    position_size_margin REAL,
    leverage INTEGER,
    stop_loss_price REAL,
    take_profit_price REAL,
    signal_score REAL,  -- Confluence score 0.0-1.0
    mvs_ob_present INTEGER,
    mvs_fvg_confirmed INTEGER,
    mvs_mss_confirmed INTEGER,
    market_regime TEXT,
    adx_at_entry REAL,
    hurst_at_entry REAL,
    funding_rate_at_entry REAL,
    atr_14_at_entry REAL,
    cvd_divergence INTEGER,
    obi_at_entry REAL,
    pnl_r REAL,
    pnl_pct REAL,
    fees_maker_taker REAL,
    fees_funding REAL,
    slippage_entry_pct REAL,
    slippage_exit_pct REAL,
    fill_rate_entry REAL,
    fill_rate_exit REAL,
    consecutive_losses_at_entry INTEGER,
    portfolio_heat_at_entry REAL,
    exit_reason TEXT,  -- STOP | TAKE_PROFIT | TIME_EXIT | MANUAL | CIRCUIT_BREAKER
    timestamp_entry TEXT NOT NULL,  -- ISO 8601 UTC
    timestamp_exit TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

TRADES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp_entry);
"""

SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,  -- LONG | SHORT | NONE
    signal_valid INTEGER NOT NULL,
    ob_present INTEGER,
    fvg_confirmed INTEGER,
    mss_confirmed INTEGER,
    entry_price REAL,
    stop_price REAL,
    target_price REAL,
    risk_reward REAL,
    confidence REAL,
    confluence_score REAL,
    invalidation_reason TEXT,
    timestamp TEXT NOT NULL,  -- ISO 8601 UTC
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

SIGNALS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts ON signals(symbol, timestamp DESC);
"""

METRICS_SNAPSHOTS_TABLE = """
CREATE TABLE IF NOT EXISTS metrics_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rolling_win_rate REAL,
    rolling_profit_factor REAL,
    rolling_expectancy_r REAL,
    sharpe_60d REAL,
    max_drawdown REAL,
    consecutive_losses INTEGER,
    avg_slippage_pct REAL,
    fill_rate REAL,
    daily_pnl REAL,
    total_trades INTEGER,
    open_positions INTEGER,
    portfolio_heat REAL,
    timestamp TEXT NOT NULL,  -- ISO 8601 UTC
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

METRICS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics_snapshots(timestamp DESC);
"""


# =============================================================================
# DATABASE CONNECTION MANAGER
# =============================================================================

class DatabaseManager:
    """Async SQLite database manager with connection pooling simulation."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._connection: Optional[aiosqlite.Connection] = None
    
    async def connect(self) -> aiosqlite.Connection:
        """Establish database connection and ensure tables exist."""
        if self._connection is None or self._connection.total_changes < 0:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            self._connection = await aiosqlite.connect(str(self.db_path))
            self._connection.row_factory = aiosqlite.Row
            
            # Enable WAL mode for better concurrent access
            await self._connection.execute("PRAGMA journal_mode=WAL;")
            await self._connection.execute("PRAGMA synchronous=NORMAL;")
            await self._connection.execute("PRAGMA cache_size=10000;")
            
            logger.info(f"Connected to database: {self.db_path}")
        
        return self._connection
    
    async def disconnect(self):
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Database connection closed")
    
    async def init_schema(self):
        """Create all tables and indexes."""
        conn = await self.connect()
        
        tables = [
            CANDLES_TABLE, CANDLES_INDEX,
            LIVE_STATE_TABLE,
            TRADES_TABLE, TRADES_INDEX,
            SIGNALS_TABLE, SIGNALS_INDEX,
            METRICS_SNAPSHOTS_TABLE, METRICS_INDEX,
        ]
        
        for table_sql in tables:
            await conn.execute(table_sql)
        
        await conn.commit()
        logger.info("Database schema initialized")
    
    @property
    def connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._connection


# =============================================================================
# CANDLES REPOSITORY
# =============================================================================

class CandlesRepository:
    """CRUD operations for OHLCV candle data."""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    async def upsert_candle(
        self,
        symbol: str,
        timeframe: str,
        timestamp: int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        turnover: float = 0,
        trades_count: int = 0,
    ) -> bool:
        """Insert or update a single candle."""
        conn = await self.db.connect()
        
        query = """
        INSERT INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume, turnover, trades_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, timeframe, timestamp) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume,
            turnover = excluded.turnover,
            trades_count = excluded.trades_count,
            updated_at = CURRENT_TIMESTAMP
        """
        
        await conn.execute(query, (
            symbol, timeframe, timestamp,
            open, high, low, close, volume, turnover, trades_count
        ))
        await conn.commit()
        return True
    
    async def upsert_candles_batch(
        self,
        symbol: str,
        timeframe: str,
        candles: list[dict[str, Any]],
    ) -> int:
        """Batch insert/update multiple candles. Returns count inserted."""
        conn = await self.db.connect()
        
        query = """
        INSERT OR REPLACE INTO candles 
        (symbol, timeframe, timestamp, open, high, low, close, volume, turnover, trades_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        rows = []
        for c in candles:
            rows.append((
                symbol,
                timeframe,
                c.get('timestamp', c.get('time', 0)),
                c.get('open', 0),
                c.get('high', 0),
                c.get('low', 0),
                c.get('close', 0),
                c.get('volume', 0),
                c.get('turnover', 0),
                c.get('trades_count', 0),
            ))
        
        await conn.executemany(query, rows)
        await conn.commit()
        return len(rows)
    
    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 1000,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Fetch candles with optional time range filtering."""
        conn = await self.db.connect()
        
        query = """
        SELECT timestamp, open, high, low, close, volume, turnover, trades_count
        FROM candles
        WHERE symbol = ? AND timeframe = ?
        """
        params: list[Any] = [symbol, timeframe]
        
        if start_ts is not None:
            query += " AND timestamp >= ?"
            params.append(start_ts)
        
        if end_ts is not None:
            query += " AND timestamp <= ?"
            params.append(end_ts)
        
        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)
        
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()
        
        return [
            {
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'turnover': row['turnover'],
                'trades_count': row['trades_count'],
            }
            for row in rows
        ]
    
    async def get_latest_candle(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[dict[str, Any]]:
        """Get the most recent candle for a symbol/timeframe."""
        conn = await self.db.connect()
        
        query = """
        SELECT timestamp, open, high, low, close, volume, turnover, trades_count
        FROM candles
        WHERE symbol = ? AND timeframe = ?
        ORDER BY timestamp DESC
        LIMIT 1
        """
        
        cursor = await conn.execute(query, (symbol, timeframe))
        row = await cursor.fetchone()
        
        if row is None:
            return None
        
        return {
            'timestamp': row['timestamp'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume'],
            'turnover': row['turnover'],
            'trades_count': row['trades_count'],
        }
    
    async def get_candle_count(
        self,
        symbol: str,
        timeframe: str,
    ) -> int:
        """Count stored candles for a symbol/timeframe."""
        conn = await self.db.connect()
        
        query = """
        SELECT COUNT(*) as cnt FROM candles
        WHERE symbol = ? AND timeframe = ?
        """
        
        cursor = await conn.execute(query, (symbol, timeframe))
        row = await cursor.fetchone()
        
        return row['cnt'] if row else 0
    
    async def delete_old_candles(
        self,
        symbol: str,
        timeframe: str,
        keep_days: int = 90,
    ) -> int:
        """Delete candles older than keep_days. Returns count deleted."""
        conn = await self.db.connect()
        
        cutoff_ts = int((datetime.now(timezone.utc).timestamp() - keep_days * 86400) * 1000)
        
        query = """
        DELETE FROM candles
        WHERE symbol = ? AND timeframe = ? AND timestamp < ?
        """
        
        cursor = await conn.execute(query, (symbol, timeframe, cutoff_ts))
        await conn.commit()
        
        return cursor.rowcount


# =============================================================================
# LIVE STATE REPOSITORY
# =============================================================================

class LiveStateRepository:
    """CRUD operations for live trading state."""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    async def upsert_live_state(
        self,
        symbol: str,
        last_candle_ts: Optional[int] = None,
        last_1m_ts: Optional[int] = None,
        last_15m_ts: Optional[int] = None,
        last_1h_ts: Optional[int] = None,
        last_4h_ts: Optional[int] = None,
        current_regime: Optional[str] = None,
        adx_value: Optional[float] = None,
        hurst_value: Optional[float] = None,
        atr_percentile: Optional[float] = None,
        funding_rate: Optional[float] = None,
        open_interest: Optional[float] = None,
        cvd_value: Optional[float] = None,
        obi_value: Optional[float] = None,
        active_fvgs: Optional[list[dict]] = None,
        active_obs: Optional[list[dict]] = None,
        circuit_breaker_status: str = 'ACTIVE',
        pause_reason: Optional[str] = None,
        pause_until: Optional[int] = None,
    ) -> bool:
        """Update live state for a symbol."""
        conn = await self.db.connect()
        
        # Serialize JSON fields
        fvgs_json = json.dumps(active_fvgs) if active_fvgs else '[]'
        obs_json = json.dumps(active_obs) if active_obs else '[]'
        
        query = """
        INSERT INTO live_state (
            symbol, last_candle_ts, last_1m_ts, last_15m_ts, last_1h_ts, last_4h_ts,
            current_regime, adx_value, hurst_value, atr_percentile,
            funding_rate, open_interest, cvd_value, obi_value,
            active_fvgs, active_obs,
            circuit_breaker_status, pause_reason, pause_until
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            last_candle_ts = COALESCE(excluded.last_candle_ts, live_state.last_candle_ts),
            last_1m_ts = COALESCE(excluded.last_1m_ts, live_state.last_1m_ts),
            last_15m_ts = COALESCE(excluded.last_15m_ts, live_state.last_15m_ts),
            last_1h_ts = COALESCE(excluded.last_1h_ts, live_state.last_1h_ts),
            last_4h_ts = COALESCE(excluded.last_4h_ts, live_state.last_4h_ts),
            current_regime = COALESCE(excluded.current_regime, live_state.current_regime),
            adx_value = COALESCE(excluded.adx_value, live_state.adx_value),
            hurst_value = COALESCE(excluded.hurst_value, live_state.hurst_value),
            atr_percentile = COALESCE(excluded.atr_percentile, live_state.atr_percentile),
            funding_rate = COALESCE(excluded.funding_rate, live_state.funding_rate),
            open_interest = COALESCE(excluded.open_interest, live_state.open_interest),
            cvd_value = COALESCE(excluded.cvd_value, live_state.cvd_value),
            obi_value = COALESCE(excluded.obi_value, live_state.obi_value),
            active_fvgs = excluded.active_fvgs,
            active_obs = excluded.active_obs,
            circuit_breaker_status = excluded.circuit_breaker_status,
            pause_reason = excluded.pause_reason,
            pause_until = excluded.pause_until,
            updated_at = CURRENT_TIMESTAMP
        """
        
        await conn.execute(query, (
            symbol,
            last_candle_ts, last_1m_ts, last_15m_ts, last_1h_ts, last_4h_ts,
            current_regime, adx_value, hurst_value, atr_percentile,
            funding_rate, open_interest, cvd_value, obi_value,
            fvgs_json, obs_json,
            circuit_breaker_status, pause_reason, pause_until,
        ))
        await conn.commit()
        return True
    
    async def get_live_state(self, symbol: str) -> Optional[dict[str, Any]]:
        """Get current live state for a symbol."""
        conn = await self.db.connect()
        
        query = """
        SELECT * FROM live_state WHERE symbol = ?
        """
        
        cursor = await conn.execute(query, (symbol,))
        row = await cursor.fetchone()
        
        if row is None:
            return None
        
        # Parse JSON fields
        active_fvgs = json.loads(row['active_fvgs']) if row['active_fvgs'] else []
        active_obs = json.loads(row['active_obs']) if row['active_obs'] else []
        
        return {
            'symbol': row['symbol'],
            'last_candle_ts': row['last_candle_ts'],
            'last_1m_ts': row['last_1m_ts'],
            'last_15m_ts': row['last_15m_ts'],
            'last_1h_ts': row['last_1h_ts'],
            'last_4h_ts': row['last_4h_ts'],
            'current_regime': row['current_regime'],
            'adx_value': row['adx_value'],
            'hurst_value': row['hurst_value'],
            'atr_percentile': row['atr_percentile'],
            'funding_rate': row['funding_rate'],
            'open_interest': row['open_interest'],
            'cvd_value': row['cvd_value'],
            'obi_value': row['obi_value'],
            'active_fvgs': active_fvgs,
            'active_obs': active_obs,
            'circuit_breaker_status': row['circuit_breaker_status'],
            'pause_reason': row['pause_reason'],
            'pause_until': row['pause_until'],
            'updated_at': row['updated_at'],
        }
    
    async def get_all_live_states(self) -> list[dict[str, Any]]:
        """Get live state for all symbols."""
        conn = await self.db.connect()
        
        query = "SELECT * FROM live_state"
        cursor = await conn.execute(query)
        rows = await cursor.fetchall()
        
        states = []
        for row in rows:
            active_fvgs = json.loads(row['active_fvgs']) if row['active_fvgs'] else []
            active_obs = json.loads(row['active_obs']) if row['active_obs'] else []
            
            states.append({
                'symbol': row['symbol'],
                'last_candle_ts': row['last_candle_ts'],
                'last_1m_ts': row['last_1m_ts'],
                'last_15m_ts': row['last_15m_ts'],
                'last_1h_ts': row['last_1h_ts'],
                'last_4h_ts': row['last_4h_ts'],
                'current_regime': row['current_regime'],
                'adx_value': row['adx_value'],
                'hurst_value': row['hurst_value'],
                'atr_percentile': row['atr_percentile'],
                'funding_rate': row['funding_rate'],
                'open_interest': row['open_interest'],
                'cvd_value': row['cvd_value'],
                'obi_value': row['obi_value'],
                'active_fvgs': active_fvgs,
                'active_obs': active_obs,
                'circuit_breaker_status': row['circuit_breaker_status'],
                'pause_reason': row['pause_reason'],
                'pause_until': row['pause_until'],
                'updated_at': row['updated_at'],
            })
        
        return states
    
    async def update_circuit_breaker(
        self,
        symbol: str,
        status: str,
        reason: Optional[str] = None,
        pause_until: Optional[int] = None,
    ) -> bool:
        """Update circuit breaker status for a symbol."""
        return await self.upsert_live_state(
            symbol=symbol,
            circuit_breaker_status=status,
            pause_reason=reason,
            pause_until=pause_until,
        )
    
    async def update_regime(
        self,
        symbol: str,
        regime: str,
        adx: float,
        hurst: float,
        atr_pct: float,
    ) -> bool:
        """Update regime classification and metrics."""
        return await self.upsert_live_state(
            symbol=symbol,
            current_regime=regime,
            adx_value=adx,
            hurst_value=hurst,
            atr_percentile=atr_pct,
        )
    
    async def update_indicators(
        self,
        symbol: str,
        active_fvgs: list[dict],
        active_obs: list[dict],
        cvd_value: float,
        obi_value: float,
    ) -> bool:
        """Update active FVGs, OBs, and volume indicators."""
        return await self.upsert_live_state(
            symbol=symbol,
            active_fvgs=active_fvgs,
            active_obs=active_obs,
            cvd_value=cvd_value,
            obi_value=obi_value,
        )


# =============================================================================
# TRADES REPOSITORY
# =============================================================================

class TradesRepository:
    """CRUD operations for trade logs."""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    async def create_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        entry_price_expected: float,
        position_size_notional: float,
        position_size_margin: float,
        leverage: int,
        stop_loss_price: float,
        take_profit_price: float,
        signal_score: float,
        mvs_ob_present: bool,
        mvs_fvg_confirmed: bool,
        mvs_mss_confirmed: bool,
        market_regime: str,
        adx_at_entry: float,
        hurst_at_entry: float,
        funding_rate_at_entry: float,
        atr_14_at_entry: float,
        cvd_divergence: bool,
        obi_at_entry: float,
        consecutive_losses_at_entry: int,
        portfolio_heat_at_entry: float,
        timestamp_entry: str,
    ) -> bool:
        """Create a new trade record on entry."""
        conn = await self.db.connect()
        
        query = """
        INSERT INTO trades (
            trade_id, symbol, side, status,
            entry_price_expected, position_size_notional, position_size_margin, leverage,
            stop_loss_price, take_profit_price,
            signal_score, mvs_ob_present, mvs_fvg_confirmed, mvs_mss_confirmed,
            market_regime, adx_at_entry, hurst_at_entry, funding_rate_at_entry,
            atr_14_at_entry, cvd_divergence, obi_at_entry,
            consecutive_losses_at_entry, portfolio_heat_at_entry,
            timestamp_entry
        )
        VALUES (?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        await conn.execute(query, (
            trade_id, symbol, side,
            entry_price_expected, position_size_notional, position_size_margin, leverage,
            stop_loss_price, take_profit_price,
            signal_score, 1 if mvs_ob_present else 0, 1 if mvs_fvg_confirmed else 0, 1 if mvs_mss_confirmed else 0,
            market_regime, adx_at_entry, hurst_at_entry, funding_rate_at_entry,
            atr_14_at_entry, 1 if cvd_divergence else 0, obi_at_entry,
            consecutive_losses_at_entry, portfolio_heat_at_entry,
            timestamp_entry,
        ))
        await conn.commit()
        return True
    
    async def update_trade_exit(
        self,
        trade_id: str,
        exit_price_actual: float,
        pnl_r: float,
        pnl_pct: float,
        fees_maker_taker: float,
        fees_funding: float,
        slippage_entry_pct: float,
        slippage_exit_pct: float,
        fill_rate_entry: float,
        fill_rate_exit: float,
        exit_reason: str,
        timestamp_exit: str,
    ) -> bool:
        """Update trade record on exit."""
        conn = await self.db.connect()
        
        query = """
        UPDATE trades SET
            status = 'CLOSED',
            exit_price_actual = ?,
            pnl_r = ?,
            pnl_pct = ?,
            fees_maker_taker = ?,
            fees_funding = ?,
            slippage_entry_pct = ?,
            slippage_exit_pct = ?,
            fill_rate_entry = ?,
            fill_rate_exit = ?,
            exit_reason = ?,
            timestamp_exit = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE trade_id = ?
        """
        
        await conn.execute(query, (
            exit_price_actual, pnl_r, pnl_pct,
            fees_maker_taker, fees_funding,
            slippage_entry_pct, slippage_exit_pct,
            fill_rate_entry, fill_rate_exit,
            exit_reason, timestamp_exit,
            trade_id,
        ))
        await conn.commit()
        return True
    
    async def update_trade_entry_actual(
        self,
        trade_id: str,
        entry_price_actual: float,
    ) -> bool:
        """Update actual entry price after fill."""
        conn = await self.db.connect()
        
        query = """
        UPDATE trades SET
            entry_price_actual = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE trade_id = ?
        """
        
        await conn.execute(query, (entry_price_actual, trade_id))
        await conn.commit()
        return True
    
    async def get_trade(self, trade_id: str) -> Optional[dict[str, Any]]:
        """Get a single trade by ID."""
        conn = await self.db.connect()
        
        query = "SELECT * FROM trades WHERE trade_id = ?"
        cursor = await conn.execute(query, (trade_id,))
        row = await cursor.fetchone()
        
        if row is None:
            return None
        
        return self._row_to_dict(row)
    
    async def get_trades(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get trades with optional filters."""
        conn = await self.db.connect()
        
        query = "SELECT * FROM trades WHERE 1=1"
        params: list[Any] = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        if start_date:
            query += " AND timestamp_entry >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp_entry <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp_entry DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()
        
        return [self._row_to_dict(row) for row in rows]
    
    async def get_open_trades(self, symbol: Optional[str] = None) -> list[dict[str, Any]]:
        """Get all open trades, optionally filtered by symbol."""
        conn = await self.db.connect()
        
        query = "SELECT * FROM trades WHERE status = 'OPEN'"
        params: list[Any] = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY timestamp_entry ASC"
        
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()
        
        return [self._row_to_dict(row) for row in rows]
    
    async def get_trade_count(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
    ) -> int:
        """Count trades with optional filters."""
        conn = await self.db.connect()
        
        query = "SELECT COUNT(*) as cnt FROM trades WHERE 1=1"
        params: list[Any] = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        cursor = await conn.execute(query, params)
        row = await cursor.fetchone()
        
        return row['cnt'] if row else 0
    
    def _row_to_dict(self, row: aiosqlite.Row) -> dict[str, Any]:
        """Convert database row to dictionary."""
        return {
            'id': row['id'],
            'trade_id': row['trade_id'],
            'symbol': row['symbol'],
            'side': row['side'],
            'status': row['status'],
            'entry_price_expected': row['entry_price_expected'],
            'entry_price_actual': row['entry_price_actual'],
            'exit_price_actual': row['exit_price_actual'],
            'position_size_notional': row['position_size_notional'],
            'position_size_margin': row['position_size_margin'],
            'leverage': row['leverage'],
            'stop_loss_price': row['stop_loss_price'],
            'take_profit_price': row['take_profit_price'],
            'signal_score': row['signal_score'],
            'mvs_ob_present': bool(row['mvs_ob_present']),
            'mvs_fvg_confirmed': bool(row['mvs_fvg_confirmed']),
            'mvs_mss_confirmed': bool(row['mvs_mss_confirmed']),
            'market_regime': row['market_regime'],
            'adx_at_entry': row['adx_at_entry'],
            'hurst_at_entry': row['hurst_at_entry'],
            'funding_rate_at_entry': row['funding_rate_at_entry'],
            'atr_14_at_entry': row['atr_14_at_entry'],
            'cvd_divergence': bool(row['cvd_divergence']),
            'obi_at_entry': row['obi_at_entry'],
            'pnl_r': row['pnl_r'],
            'pnl_pct': row['pnl_pct'],
            'fees_maker_taker': row['fees_maker_taker'],
            'fees_funding': row['fees_funding'],
            'slippage_entry_pct': row['slippage_entry_pct'],
            'slippage_exit_pct': row['slippage_exit_pct'],
            'fill_rate_entry': row['fill_rate_entry'],
            'fill_rate_exit': row['fill_rate_exit'],
            'consecutive_losses_at_entry': row['consecutive_losses_at_entry'],
            'portfolio_heat_at_entry': row['portfolio_heat_at_entry'],
            'exit_reason': row['exit_reason'],
            'timestamp_entry': row['timestamp_entry'],
            'timestamp_exit': row['timestamp_exit'],
            'created_at': row['created_at'],
            'updated_at': row['updated_at'],
        }


# =============================================================================
# SIGNALS REPOSITORY
# =============================================================================

class SignalsRepository:
    """CRUD operations for signal evaluations."""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    async def log_signal(
        self,
        symbol: str,
        direction: str,
        signal_valid: bool,
        ob_present: Optional[bool] = None,
        fvg_confirmed: Optional[bool] = None,
        mss_confirmed: Optional[bool] = None,
        entry_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        target_price: Optional[float] = None,
        risk_reward: Optional[float] = None,
        confidence: Optional[float] = None,
        confluence_score: Optional[float] = None,
        invalidation_reason: Optional[str] = None,
    ) -> bool:
        """Log a signal evaluation."""
        conn = await self.db.connect()
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        query = """
        INSERT INTO signals (
            symbol, direction, signal_valid,
            ob_present, fvg_confirmed, mss_confirmed,
            entry_price, stop_price, target_price,
            risk_reward, confidence, confluence_score,
            invalidation_reason, timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        await conn.execute(query, (
            symbol, direction, 1 if signal_valid else 0,
            1 if ob_present else 0 if ob_present is not None else None,
            1 if fvg_confirmed else 0 if fvg_confirmed is not None else None,
            1 if mss_confirmed else 0 if mss_confirmed is not None else None,
            entry_price, stop_price, target_price,
            risk_reward, confidence, confluence_score,
            invalidation_reason, timestamp,
        ))
        await conn.commit()
        return True
    
    async def get_latest_signals(
        self,
        symbol: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get most recent signal evaluations."""
        conn = await self.db.connect()
        
        query = "SELECT * FROM signals WHERE 1=1"
        params: list[Any] = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()
        
        return [
            {
                'id': row['id'],
                'symbol': row['symbol'],
                'direction': row['direction'],
                'signal_valid': bool(row['signal_valid']),
                'ob_present': bool(row['ob_present']) if row['ob_present'] is not None else None,
                'fvg_confirmed': bool(row['fvg_confirmed']) if row['fvg_confirmed'] is not None else None,
                'mss_confirmed': bool(row['mss_confirmed']) if row['mss_confirmed'] is not None else None,
                'entry_price': row['entry_price'],
                'stop_price': row['stop_price'],
                'target_price': row['target_price'],
                'risk_reward': row['risk_reward'],
                'confidence': row['confidence'],
                'confluence_score': row['confluence_score'],
                'invalidation_reason': row['invalidation_reason'],
                'timestamp': row['timestamp'],
            }
            for row in rows
        ]


# =============================================================================
# METRICS SNAPSHOTS REPOSITORY
# =============================================================================

class MetricsSnapshotsRepository:
    """CRUD operations for metrics snapshots."""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    async def log_metrics_snapshot(
        self,
        rolling_win_rate: float,
        rolling_profit_factor: float,
        rolling_expectancy_r: float,
        sharpe_60d: float,
        max_drawdown: float,
        consecutive_losses: int,
        avg_slippage_pct: float,
        fill_rate: float,
        daily_pnl: float,
        total_trades: int,
        open_positions: int,
        portfolio_heat: float,
    ) -> bool:
        """Log a metrics snapshot."""
        conn = await self.db.connect()
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        query = """
        INSERT INTO metrics_snapshots (
            rolling_win_rate, rolling_profit_factor, rolling_expectancy_r,
            sharpe_60d, max_drawdown, consecutive_losses,
            avg_slippage_pct, fill_rate, daily_pnl,
            total_trades, open_positions, portfolio_heat, timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        await conn.execute(query, (
            rolling_win_rate, rolling_profit_factor, rolling_expectancy_r,
            sharpe_60d, max_drawdown, consecutive_losses,
            avg_slippage_pct, fill_rate, daily_pnl,
            total_trades, open_positions, portfolio_heat,
            timestamp,
        ))
        await conn.commit()
        return True
    
    async def get_latest_metrics(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get most recent metrics snapshots."""
        conn = await self.db.connect()
        
        query = """
        SELECT * FROM metrics_snapshots
        ORDER BY timestamp DESC
        LIMIT ?
        """
        
        cursor = await conn.execute(query, (limit,))
        rows = await cursor.fetchall()
        
        return [
            {
                'id': row['id'],
                'rolling_win_rate': row['rolling_win_rate'],
                'rolling_profit_factor': row['rolling_profit_factor'],
                'rolling_expectancy_r': row['rolling_expectancy_r'],
                'sharpe_60d': row['sharpe_60d'],
                'max_drawdown': row['max_drawdown'],
                'consecutive_losses': row['consecutive_losses'],
                'avg_slippage_pct': row['avg_slippage_pct'],
                'fill_rate': row['fill_rate'],
                'daily_pnl': row['daily_pnl'],
                'total_trades': row['total_trades'],
                'open_positions': row['open_positions'],
                'portfolio_heat': row['portfolio_heat'],
                'timestamp': row['timestamp'],
            }
            for row in rows
        ]


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_database_manager(db_name: str = 'live_state.db') -> DatabaseManager:
    """Create a DatabaseManager instance using paths from paths.py."""
    db_path = DB_PATHS[db_name] if db_name in DB_PATHS else str(Path(DB_PATHS['data_dir']) / db_name)
    return DatabaseManager(db_path)


async def initialize_database():
    """Initialize all database connections and schemas."""
    # Initialize main databases
    live_state_db = create_database_manager('live_state.db')
    trades_db = create_database_manager('trades.db')
    analysis_db = create_database_manager('analysis_runs.db')
    
    await live_state_db.init_schema()
    await trades_db.init_schema()
    await analysis_db.init_schema()
    
    logger.info("All databases initialized successfully")
    
    return {
        'live_state': live_state_db,
        'trades': trades_db,
        'analysis': analysis_db,
    }
