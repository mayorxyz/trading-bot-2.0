"""
Live Trading Runner - Main async loop orchestrating the trading bot.

REFACTORED: Primary output is now trade signals via signals/output.py.
Execution is optional and controlled by AUTO_EXECUTE config flag.

Flow per symbol (staggered start, 100ms between):
1. Subscribe WebSocket: OHLCV (1m, 15m, 1H, 4H), orderbook (20 levels), aggTrades, funding
2. On each 15m candle close:
   a. Update indicators (structure, FVG, OB, CVD, OBI)
   b. Update regime (4H close only)
   c. Run funding_pre_trade_check
   d. Run MVS evaluation
   e. If MVS valid → run confluence scorer
   f. Check circuit_breaker.check_all()
   g. Check portfolio.can_open_position()
   h. If AUTO_EXECUTE=true and all pass → place maker limit order
      Else → publish signal via Telegram and /signals API endpoint
3. On each bar: update_position() for all open trades (if auto-executing)
4. On each trade close: log_trade(), update metrics, run CUSUM

MACRO EVENT GATE: No new entries 30 min before / 10 min after CPI, NFP, FOMC.
SYMBOL ISOLATION: Each symbol runs in its own asyncio task.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

from tbb.core.config import settings
from tbb.core.logger import get_logger
from tbb.data.websocket import BybitWebSocketManager
from tbb.data.store import DataStore
from tbb.indicators.structure import detect_bos, detect_choch, detect_swing_highs, detect_swing_lows
from tbb.indicators.fvg import detect_fvgs, filter_fresh_fvgs, update_fvg_mitigation
from tbb.indicators.order_block import detect_order_blocks, update_ob_touches
from tbb.indicators.regime import classify_regime, compute_adx, compute_hurst, compute_atr_percentile
from tbb.indicators.volume import compute_delta, compute_cvd, compute_obi
from tbb.indicators.funding import get_current_funding_rate, funding_pre_trade_check
from tbb.signals.mvs import evaluate_mvs
from tbb.signals.confluence import score_confluence
from tbb.signals.output import publish_trade_signal, signal_output_manager, TradeSignal
from tbb.risk.circuit_breaker import CircuitBreaker
from tbb.risk.portfolio import PortfolioManager
from tbb.risk.position_sizing import calculate_position_size
from tbb.execution.order import OrderExecutor
from tbb.monitoring.metrics import LiveMetrics
from tbb.monitoring.drift import DriftDetector

logger = get_logger(__name__)


@dataclass
class SymbolContext:
    """Holds all state and data for a single symbol."""
    symbol: str
    htf_df: Optional[any] = None  # 4H DataFrame
    ltf_df: Optional[any] = None  # 15m DataFrame
    regime: Optional[str] = None
    adx_value: Optional[float] = None
    hurst_value: Optional[float] = None
    atr_value: Optional[float] = None
    cvd_value: Optional[float] = None
    obi_value: Optional[float] = None
    funding_rate: float = 0.0
    last_4h_close: Optional[datetime] = None
    last_15m_close: Optional[datetime] = None
    active_signals: List[str] = field(default_factory=list)  # Signal IDs


@dataclass
class MacroEvent:
    """Macro economic event for gating entries."""
    name: str
    timestamp: datetime
    impact: str  # "HIGH", "MEDIUM", "LOW"


# Hardcoded macro events (in production, fetch from economic calendar API)
MACRO_EVENTS = [
    # Example: MacroEvent("CPI", datetime(2024, 1, 11, 13, 30, tzinfo=timezone.utc), "HIGH"),
    # Add actual events based on economic calendar
]


class TradingBotRunner:
    """
    Main orchestrator for live trading bot.
    
    Manages WebSocket connections, indicator updates, signal generation,
    and optional auto-execution based on AUTO_EXECUTE config flag.
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.symbol_contexts: Dict[str, SymbolContext] = {}
        self.ws_manager: Optional[BybitWebSocketManager] = None
        self.data_store = DataStore()
        self.circuit_breaker = CircuitBreaker()
        self.portfolio_manager = PortfolioManager()
        self.order_executor = OrderExecutor() if settings.AUTO_EXECUTE else None
        self.drift_detector = DriftDetector()
        
        # Macro event blacklist
        self.macro_events = MACRO_EVENTS
        
        # Running state
        self._running = False
        self._tasks: Dict[str, asyncio.Task] = {}
        
        logger.info(f"TradingBotRunner initialized for {len(symbols)} symbols")
        logger.info(f"AUTO_EXECUTE mode: {settings.AUTO_EXECUTE}")
    
    async def start(self):
        """Start the trading bot runner."""
        self._running = True
        
        # Initialize WebSocket manager
        self.ws_manager = BybitWebSocketManager()
        await self.ws_manager.connect()
        
        # Initialize symbol contexts
        for symbol in self.symbols:
            self.symbol_contexts[symbol] = SymbolContext(symbol=symbol)
            
            # Subscribe to WebSocket streams
            await self.ws_manager.subscribe_symbol(symbol)
            logger.info(f"Subscribed to {symbol}")
        
        # Start symbol tasks with staggered start (100ms between)
        for i, symbol in enumerate(self.symbols):
            await asyncio.sleep(0.1)  # 100ms stagger
            task = asyncio.create_task(self._run_symbol_loop(symbol))
            self._tasks[symbol] = task
            logger.info(f"Started task for {symbol}")
        
        logger.info("Trading bot runner started")
    
    async def stop(self):
        """Stop the trading bot runner gracefully."""
        self._running = False
        
        # Cancel all symbol tasks
        for symbol, task in self._tasks.items():
            task.cancel()
            logger.info(f"Cancelled task for {symbol}")
        
        # Close WebSocket connection
        if self.ws_manager:
            await self.ws_manager.disconnect()
        
        logger.info("Trading bot runner stopped")
    
    async def _run_symbol_loop(self, symbol: str):
        """
        Main loop for a single symbol.
        Runs independently for each symbol (symbol isolation).
        """
        ctx = self.symbol_contexts[symbol]
        logger.info(f"Starting symbol loop for {symbol}")
        
        while self._running:
            try:
                # Wait for new 15m candle close
                await self._wait_for_15m_close(symbol)
                
                if not self._running:
                    break
                
                # Get latest DataFrames
                ctx.ltf_df = await self._get_dataframe(symbol, "15m")
                ctx.htf_df = await self._get_dataframe(symbol, "4h")
                
                if ctx.ltf_df is None or ctx.htf_df is None:
                    logger.warning(f"Missing data for {symbol}, skipping cycle")
                    continue
                
                # Update regime on 4H close
                if self._is_4h_close(ctx):
                    await self._update_regime(symbol, ctx)
                
                # Update volume indicators (CVD, OBI)
                await self._update_volume_indicators(symbol, ctx)
                
                # Update funding rate
                ctx.funding_rate = await get_current_funding_rate(symbol)
                
                # Check macro event gate
                if self._is_macro_event_blackout():
                    logger.info(f"Macro event blackout, skipping signal generation for {symbol}")
                    continue
                
                # Run MVS evaluation
                mvs_result = evaluate_mvs(ctx.htf_df, ctx.ltf_df, symbol)
                
                if not mvs_result.signal_valid:
                    logger.debug(f"No valid MVS signal for {symbol}: {mvs_result.invalidation_reason}")
                    continue
                
                # Run confluence scorer
                confluence_score = score_confluence(
                    symbol=symbol,
                    htf_df=ctx.htf_df,
                    ltf_df=ctx.ltf_df,
                    ob=mvs_result.ob_present,
                    fvg=mvs_result.fvg_confirmed,
                    mss=mvs_result.mss_confirmed,
                    cvd=ctx.cvd_value,
                    obi=ctx.obi_value,
                    funding=ctx.funding_rate,
                    regime=ctx.regime,
                )
                
                logger.info(
                    f"MVS signal for {symbol}: score={confluence_score:.2f}, "
                    f"entry={mvs_result.entry_price:.2f}, R:R={mvs_result.risk_reward:.2f}"
                )
                
                # Check minimum thresholds
                if confluence_score < settings.MIN_CONFLUENCE_SCORE:
                    logger.info(f"Confluence score {confluence_score:.2f} below threshold for {symbol}")
                    continue
                
                if mvs_result.risk_reward < settings.MIN_RISK_REWARD:
                    logger.info(f"R:R {mvs_result.risk_reward:.2f} below threshold for {symbol}")
                    continue
                
                # Check circuit breaker
                cb_result = self.circuit_breaker.check_all(LiveMetrics().compute([]))
                if cb_result.triggered:
                    logger.warning(f"Circuit breaker triggered: {cb_result.reason}")
                    continue
                
                # Calculate position size
                position_size_result = calculate_position_size(
                    account_balance=10000.0,  # TODO: Get from config/exchange
                    entry_price=mvs_result.entry_price,
                    stop_price=mvs_result.stop_price,
                    risk_pct=settings.ACCOUNT_RISK_PCT,
                    leverage=settings.DEFAULT_LEVERAGE,
                    funding_rate=ctx.funding_rate,
                    confluence_score=confluence_score,
                )
                
                # Check portfolio constraints
                can_open, rejection_reason = self.portfolio_manager.can_open_position(
                    symbol, position_size_result.risk_pct
                )
                if not can_open:
                    logger.info(f"Portfolio constraint: {rejection_reason} for {symbol}")
                    # Continue to publish signal even if portfolio blocks it
                    # User can see the signal and manually override
                
                # PUBLISH SIGNAL (primary output)
                max_age_bars = (
                    settings.FRESH_FVG_MAX_AGE_BARS_15M
                    if "15m" in settings.LTF_TIMEFRAME
                    else settings.FRESH_FVG_MAX_AGE_BARS_1H
                )
                
                signal = await publish_trade_signal(
                    symbol=symbol,
                    direction=mvs_result.direction,
                    entry_price=mvs_result.entry_price,
                    stop_loss=mvs_result.stop_price,
                    take_profit_1=mvs_result.entry_price + (mvs_result.entry_price - mvs_result.stop_price),  # 1R
                    take_profit_2=mvs_result.target_price,
                    risk_reward=mvs_result.risk_reward,
                    confluence_score=confluence_score,
                    mvs_ob_present=mvs_result.ob_present,
                    mvs_fvg_confirmed=mvs_result.fvg_confirmed,
                    mvs_mss_confirmed=mvs_result.mss_confirmed,
                    market_regime=ctx.regime or "UNKNOWN",
                    adx_value=ctx.adx_value,
                    hurst_value=ctx.hurst_value,
                    funding_rate=ctx.funding_rate,
                    atr_14=ctx.atr_value,
                    cvd_divergence=False,  # TODO: Implement CVD divergence detection
                    obi_value=ctx.obi_value,
                    max_age_bars=max_age_bars,
                    auto_execute=settings.AUTO_EXECUTE,
                    position_size=position_size_result.quantity if can_open else None,
                    notes=[rejection_reason] if not can_open else [],
                )
                
                ctx.active_signals.append(signal.signal_id)
                
                # AUTO-EXECUTE (optional, only if AUTO_EXECUTE=true)
                if settings.AUTO_EXECUTE and can_open:
                    # Funding pre-trade check
                    funding_check = funding_pre_trade_check(
                        ctx.funding_rate,
                        expected_hold_hours=24,  # TODO: Make configurable
                    )
                    if not funding_check.passed:
                        logger.warning(f"Funding check failed for {symbol}: {funding_check.reason}")
                        continue
                    
                    # Place limit order at FVG midpoint
                    try:
                        order_result = await self.order_executor.place_limit_order(
                            symbol=symbol,
                            side=mvs_result.direction,
                            price=mvs_result.entry_price,
                            size=position_size_result.quantity,
                            leverage=settings.DEFAULT_LEVERAGE,
                            reduce_only=False,
                        )
                        
                        if order_result.success:
                            logger.info(
                                f"Order placed for {symbol}: {order_result.side} "
                                f"{order_result.size} @ {order_result.fill_price}"
                            )
                        else:
                            logger.warning(f"Order placement failed: {order_result.message}")
                            
                    except Exception as e:
                        logger.error(f"Order execution error for {symbol}: {e}")
                
                # Update bars remaining for existing signals
                await signal_output_manager.decrement_bars_remaining(symbol, "15m")
                
            except asyncio.CancelledError:
                logger.info(f"Symbol loop cancelled for {symbol}")
                break
            except Exception as e:
                logger.error(f"Error in symbol loop for {symbol}: {e}", exc_info=True)
                await asyncio.sleep(5)  # Back off on error
    
    async def _wait_for_15m_close(self, symbol: str):
        """Wait until the next 15m candle closes."""
        ctx = self.symbol_contexts[symbol]
        now = datetime.now(timezone.utc)
        
        # Calculate next 15m boundary
        minutes_to_next = 15 - (now.minute % 15)
        next_close = now.replace(minute=now.minute + minutes_to_next, second=0, microsecond=0)
        if minutes_to_next >= 15:
            next_close = next_close + timedelta(minutes=15)
        
        sleep_seconds = (next_close - now).total_seconds()
        logger.debug(f"Waiting {sleep_seconds:.0f}s for next 15m close on {symbol}")
        
        await asyncio.sleep(sleep_seconds)
        ctx.last_15m_close = next_close
    
    def _is_4h_close(self, ctx: SymbolContext) -> bool:
        """Check if 4H candle just closed."""
        if ctx.last_15m_close is None:
            return False
        
        # 4H closes every 4 hours: 00:00, 04:00, 08:00, etc.
        hour = ctx.last_15m_close.hour
        return hour % 4 == 0 and ctx.last_15m_close.minute == 0
    
    async def _update_regime(self, symbol: str, ctx: SymbolContext):
        """Update regime classification on 4H close."""
        if ctx.htf_df is None or len(ctx.htf_df) < 100:
            logger.warning(f"Insufficient data for regime calculation on {symbol}")
            return
        
        ctx.adx_value = compute_adx(ctx.htf_df)
        ctx.hurst_value = compute_hurst(ctx.htf_df)
        ctx.atr_value = compute_atr_percentile(ctx.htf_df)
        
        if ctx.adx_value is not None and ctx.hurst_value is not None:
            ctx.regime = classify_regime(ctx.adx_value, ctx.hurst_value, ctx.atr_value or 50.0)
            logger.info(f"Regime updated for {symbol}: {ctx.regime} (ADX={ctx.adx_value:.1f}, Hurst={ctx.hurst_value:.2f})")
        
        ctx.last_4h_close = ctx.last_15m_close
    
    async def _update_volume_indicators(self, symbol: str, ctx: SymbolContext):
        """Update CVD and OBI indicators."""
        if ctx.ltf_df is None:
            return
        
        # TODO: Implement proper CVD calculation from aggTrades
        # For now, placeholder
        ctx.cvd_value = 0.0
        
        # TODO: Implement OBI from orderbook data
        ctx.obi_value = 0.0
    
    async def _get_dataframe(self, symbol: str, timeframe: str):
        """Get DataFrame for symbol/timeframe from data store."""
        # TODO: Implement proper retrieval from DataStore
        # For now, return None (will be populated when WebSocket integration is complete)
        return None
    
    def _is_macro_event_blackout(self) -> bool:
        """Check if we're in a macro event blackout window."""
        if not settings.MACRO_EVENT_GATE_ENABLED:
            return False
        
        now = datetime.now(timezone.utc)
        
        for event in self.macro_events:
            if event.impact != "HIGH":
                continue
            
            # Blackout window: 30 min before to 10 min after
            blackout_start = event.timestamp - timedelta(minutes=settings.MACRO_EVENT_BLACKOUT_MIN_BEFORE)
            blackout_end = event.timestamp + timedelta(minutes=settings.MACRO_EVENT_BLACKOUT_MIN_AFTER)
            
            if blackout_start <= now <= blackout_end:
                return True
        
        return False
