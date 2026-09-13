"""
FastAPI routes for TBB API endpoints.

Endpoints:
- GET /status - Bot running state, active positions, circuit breaker status
- GET /trades - Paginated trade log with filters
- GET /metrics - Live metrics snapshot
- GET /signals - Current MVS evaluation per symbol (active signals)
- GET /signals/history - Historical signals with pagination
- GET /portfolio - Open positions, portfolio heat, correlation matrix
- POST /pause - Manually trigger circuit breaker pause
- POST /resume - Resume after manual pause
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from tbb.core.config import settings
from tbb.core.logger import get_logger
from tbb.signals.output import signal_output_manager, TradeSignal, SignalStatus
from tbb.monitoring.metrics import LiveMetrics
from tbb.risk.circuit_breaker import CircuitBreaker, CircuitBreakerResult
from tbb.data.store import DataStore

logger = get_logger(__name__)

router = APIRouter()


# === Response Models ===

class StatusResponse(BaseModel):
    bot_running: bool
    auto_execute_enabled: bool
    circuit_breaker_status: str  # "ACTIVE", "PAUSED_DAY", "PAUSED_4H", "HALTED"
    active_positions_count: int
    active_signals_count: int
    last_update: str


class TradeLogResponse(BaseModel):
    trades: List[Dict[str, Any]]
    total_count: int
    limit: int
    offset: int


class MetricsResponse(BaseModel):
    rolling_win_rate: Optional[float]
    rolling_profit_factor: Optional[float]
    rolling_expectancy_r: Optional[float]
    sharpe_60d: Optional[float]
    max_drawdown: Optional[float]
    consecutive_losses: int
    avg_slippage_pct: Optional[float]
    fill_rate: Optional[float]
    daily_pnl: Optional[float]
    alert_status: Optional[str]


class SignalResponse(BaseModel):
    signal_id: str
    timestamp: str
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward: float
    confluence_score: float
    mvs_conditions: Dict[str, bool]
    market_context: Dict[str, Any]
    validity: Dict[str, Any]
    execution: Dict[str, Any]
    status: str


class PortfolioResponse(BaseModel):
    open_positions: List[Dict[str, Any]]
    portfolio_heat: float
    max_portfolio_heat: float
    correlation_matrix: Dict[str, Dict[str, float]]
    can_open_new_position: bool
    rejection_reason: Optional[str]


class PauseRequest(BaseModel):
    reason: str = "Manual pause via API"
    duration_hours: Optional[float] = None


class PauseResponse(BaseModel):
    success: bool
    message: str
    circuit_breaker_status: str


# === Helper Functions ===

def _trade_signal_to_response(signal: TradeSignal) -> SignalResponse:
    """Convert TradeSignal to SignalResponse for API."""
    data = signal.to_dict()
    return SignalResponse(**data)


# === Routes ===

@router.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Get bot running status, circuit breaker state, and active positions count.
    """
    # Get circuit breaker status
    cb = CircuitBreaker()
    cb_result = cb.check_all(LiveMetrics().compute([]))  # Empty trades for now
    
    # Map circuit breaker action to status string
    status_map = {
        "NONE": "ACTIVE",
        "PAUSE_DAY": "PAUSED_DAY",
        "PAUSE_4H": "PAUSED_4H",
        "HALT_ALL": "HALTED",
    }
    cb_status = status_map.get(cb_result.action, "ACTIVE")
    
    # Count active signals
    active_signals = signal_output_manager.get_active_signals()
    
    # TODO: Get actual position count from exchange
    active_positions = 0
    
    return StatusResponse(
        bot_running=True,
        auto_execute_enabled=settings.AUTO_EXECUTE,
        circuit_breaker_status=cb_status,
        active_positions_count=active_positions,
        active_signals_count=len(active_signals),
        last_update=datetime.utcnow().isoformat(),
    )


@router.get("/trades", response_model=TradeLogResponse)
async def get_trades(
    symbol: Optional[str] = Query(None, description="Filter by symbol (e.g., BTCUSDT)"),
    side: Optional[str] = Query(None, description="Filter by side (LONG|SHORT)"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    limit: int = Query(50, ge=1, le=200, description="Number of trades to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """
    Get paginated trade log with optional filters.
    
    Returns trades from the trades.db database with all mandatory fields.
    """
    # TODO: Implement actual database query
    # For now, return empty response with structure
    return TradeLogResponse(
        trades=[],
        total_count=0,
        limit=limit,
        offset=offset,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get live metrics snapshot including win rate, Sharpe, expectancy, drawdown.
    """
    # TODO: Get actual trades from database
    metrics = LiveMetrics().compute([])
    
    return MetricsResponse(
        rolling_win_rate=metrics.rolling_win_rate,
        rolling_profit_factor=metrics.rolling_profit_factor,
        rolling_expectancy_r=metrics.rolling_expectancy_r,
        sharpe_60d=metrics.sharpe_60d,
        max_drawdown=metrics.max_drawdown,
        consecutive_losses=metrics.consecutive_losses,
        avg_slippage_pct=metrics.avg_slippage_pct,
        fill_rate=metrics.fill_rate,
        daily_pnl=metrics.daily_pnl,
        alert_status=metrics.alert_status if hasattr(metrics, 'alert_status') else None,
    )


@router.get("/signals", response_model=List[SignalResponse])
async def get_signals(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    status: Optional[str] = Query(None, description="Filter by status (ACTIVE, PENDING, etc.)"),
):
    """
    Get current active trade signals.
    
    Returns signals that have been generated by MVS + confluence scorer
    but not yet expired, filled, or cancelled.
    """
    signals = signal_output_manager.get_active_signals(symbol=symbol)
    
    # Filter by status if provided
    if status:
        try:
            target_status = SignalStatus(status)
            signals = [s for s in signals if s.status == target_status]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    
    return [_trade_signal_to_response(s) for s in signals]


@router.get("/signals/history", response_model=List[SignalResponse])
async def get_signal_history(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    limit: int = Query(50, ge=1, le=200, description="Number of signals to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """
    Get historical trade signals (including expired, filled, cancelled).
    """
    signals = signal_output_manager.get_signal_history(
        symbol=symbol,
        limit=limit,
        offset=offset,
    )
    return [_trade_signal_to_response(s) for s in signals]


@router.get("/signals/{signal_id}", response_model=SignalResponse)
async def get_signal_by_id(signal_id: str):
    """
    Get a specific signal by ID.
    """
    signal = signal_output_manager.get_signal_by_id(signal_id)
    if not signal:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")
    return _trade_signal_to_response(signal)


@router.get("/portfolio", response_model=PortfolioResponse)
async def get_portfolio():
    """
    Get portfolio overview: open positions, heat, correlation matrix.
    """
    # TODO: Implement actual portfolio tracking
    # For now, return placeholder response
    return PortfolioResponse(
        open_positions=[],
        portfolio_heat=0.0,
        max_portfolio_heat=settings.MAX_PORTFOLIO_HEAT,
        correlation_matrix={},
        can_open_new_position=True,
        rejection_reason=None,
    )


@router.post("/pause", response_model=PauseResponse)
async def pause_bot(request: PauseRequest):
    """
    Manually trigger circuit breaker pause.
    
    This overrides all signal logic and blocks new entries until resumed.
    """
    # TODO: Implement actual circuit breaker state management
    # For now, return success response
    logger.warning(f"Manual pause triggered via API: {request.reason}")
    
    return PauseResponse(
        success=True,
        message=f"Bot paused: {request.reason}",
        circuit_breaker_status="PAUSED_MANUAL",
    )


@router.post("/resume", response_model=PauseResponse)
async def resume_bot():
    """
    Resume bot after manual pause.
    
    Clears manual pause state and allows normal operation to continue.
    """
    # TODO: Implement actual circuit breaker state management
    # For now, return success response
    logger.info("Bot resumed via API")
    
    return PauseResponse(
        success=True,
        message="Bot resumed successfully",
        circuit_breaker_status="ACTIVE",
    )
