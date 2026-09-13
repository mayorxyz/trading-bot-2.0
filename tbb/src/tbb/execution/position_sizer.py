"""
Position Sizer - Calculates position size based on risk parameters and exchange constraints.
Respects min_order_size, step_size (lot size), and account equity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from tbb.core.config import settings
from tbb.core.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    quantity: float  # Actual contract quantity (rounded to step_size)
    notional_usd: float  # quantity * entry_price
    margin_required: float  # notional / leverage
    risk_amount_usd: float  # Account risk in USD
    risk_pct: float  # Risk as % of account
    leverage: int
    entry_price: float
    stop_loss: float
    liquidation_price: float
    respects_min_order: bool
    respects_max_notional: bool
    adjusted_for_step_size: bool
    original_quantity: float  # Before step_size rounding
    step_size: float
    min_order_size: float


def calculate_liquidation_price(
    entry_price: float,
    leverage: int,
    side: str,
    is_isolated: bool = True,
) -> float:
    """
    Calculate approximate liquidation price for isolated margin.
    
    For LONG: liq = entry * (1 - 1/leverage + maintenance_margin_rate)
    For SHORT: liq = entry * (1 + 1/leverage - maintenance_margin_rate)
    
    Bybit UTA maintenance margin rate ≈ 0.5% (0.005) for major pairs.
    """
    mmr = 0.005  # Maintenance margin rate
    
    if side == "LONG":
        liq_price = entry_price * (1 - 1/leverage + mmr)
    else:  # SHORT
        liq_price = entry_price * (1 + 1/leverage - mmr)
    
    return max(liq_price, 0.0)


def get_lot_size_constraints(symbol: str) -> tuple[float, float]:
    """
    Get min order size and step size for a symbol.
    In production, fetch from exchange instrument info.
    For MVP, use hardcoded values for common pairs.
    """
    # Bybit linear contract specifications
    lot_sizes = {
        "BTCUSDT": {"min": 0.001, "step": 0.001},
        "ETHUSDT": {"min": 0.01, "step": 0.01},
        "SOLUSDT": {"min": 0.1, "step": 0.1},
        "XRPUSDT": {"min": 1.0, "step": 1.0},
        "ADAUSDT": {"min": 1.0, "step": 1.0},
        "DOGEUSDT": {"min": 1.0, "step": 1.0},
    }
    
    defaults = {"min": 1.0, "step": 0.01}
    
    config = lot_sizes.get(symbol, defaults)
    return config["min"], config["step"]


def calculate_position_size(
    account_equity: float,
    entry_price: float,
    stop_loss: float,
    side: str,  # "LONG" | "SHORT"
    symbol: str,
    leverage: int = None,
    risk_per_trade_pct: float = None,
) -> PositionSizeResult:
    """
    Calculate position size based on R-risk and exchange constraints.
    
    Formula: size = (account_equity * risk_pct) / |entry - stop|
    
    Then adjust for:
    1. Min order size (notional >= min)
    2. Step size (lot size precision)
    3. Max notional per order (exchange limit)
    
    Returns PositionSizeResult with all details.
    """
    leverage = leverage or settings.DEFAULT_LEVERAGE
    risk_per_trade_pct = risk_per_trade_pct or settings.RISK_PER_TRADE_PCT
    
    # Get exchange constraints
    min_order_size, step_size = get_lot_size_constraints(symbol)
    
    # Calculate raw risk amount
    risk_amount_usd = account_equity * risk_per_trade_pct
    
    # Calculate distance to stop (risk per contract)
    if side == "LONG":
        risk_per_contract = entry_price - stop_loss
    else:  # SHORT
        risk_per_contract = stop_loss - entry_price
    
    if risk_per_contract <= 0:
        logger.error(f"Invalid stop loss: entry={entry_price}, stop={stop_loss}, side={side}")
        raise ValueError("Stop loss must be below entry for LONG or above entry for SHORT")
    
    # Calculate raw quantity
    raw_quantity = risk_amount_usd / risk_per_contract
    
    # Round to step size
    adjusted_quantity = round(raw_quantity / step_size) * step_size
    adjusted_for_step_size = (adjusted_quantity != raw_quantity)
    
    # Check min order size (by notional value)
    notional_usd = adjusted_quantity * entry_price
    min_notional = min_order_size * entry_price
    
    respects_min_order = notional_usd >= min_notional
    if not respects_min_order:
        # Scale up to meet minimum
        adjusted_quantity = min_order_size
        notional_usd = adjusted_quantity * entry_price
        logger.warning(
            f"Position size below minimum. Adjusted to {adjusted_quantity} ({notional_usd:.2f} USDT)"
        )
    
    # Calculate margin required
    margin_required = notional_usd / leverage
    
    # Calculate liquidation price
    liq_price = calculate_liquidation_price(entry_price, leverage, side)
    
    # Verify stop is inside liquidation price (safety buffer)
    if side == "LONG":
        stop_inside_liq = stop_loss > liq_price * 1.005  # 0.5% buffer
        if not stop_inside_liq:
            logger.warning(
                f"Stop loss ({stop_loss}) too close to liquidation ({liq_price:.2f}). "
                f"Consider reducing leverage."
            )
    else:  # SHORT
        stop_inside_liq = stop_loss < liq_price * 0.995
        if not stop_inside_liq:
            logger.warning(
                f"Stop loss ({stop_loss}) too close to liquidation ({liq_price:.2f}). "
                f"Consider reducing leverage."
            )
    
    # Max notional check (Bybit max order size varies by symbol, ~2M USDT default)
    max_notional = 2_000_000
    respects_max_notional = notional_usd <= max_notional
    if not respects_max_notional:
        logger.warning(f"Notional ({notional_usd:.2f}) exceeds max order size ({max_notional})")
        # Scale down
        adjusted_quantity = max_notional / entry_price
        adjusted_quantity = round(adjusted_quantity / step_size) * step_size
        notional_usd = adjusted_quantity * entry_price
    
    result = PositionSizeResult(
        quantity=adjusted_quantity,
        notional_usd=notional_usd,
        margin_required=margin_required,
        risk_amount_usd=risk_amount_usd,
        risk_pct=risk_per_trade_pct,
        leverage=leverage,
        entry_price=entry_price,
        stop_loss=stop_loss,
        liquidation_price=liq_price,
        respects_min_order=respects_min_order,
        respects_max_notional=respects_max_notional,
        adjusted_for_step_size=adjusted_for_step_size,
        original_quantity=raw_quantity,
        step_size=step_size,
        min_order_size=min_order_size,
    )
    
    logger.info(
        f"Position size calculated: {symbol} {side} qty={result.quantity:.4f} "
        f"(@ {entry_price:.2f}) | Notional: ${notional_usd:.2f} | "
        f"Risk: ${risk_amount_usd:.2f} ({risk_per_trade_pct*100:.1f}%) | "
        f"Liq: ${liq_price:.2f}"
    )
    
    return result


def validate_position_against_limits(
    symbol: str,
    proposed_notional: float,
    current_open_positions: list,
    max_open_positions: int = None,
    max_portfolio_heat: float = None,
) -> tuple[bool, str]:
    """
    Validate if a new position can be opened given current portfolio state.
    
    Checks:
    1. Open positions count < MAX_OPEN_POSITIONS
    2. Total portfolio heat (open risk) < MAX_PORTFOLIO_HEAT
    
    Returns (can_open, reason_if_rejected)
    """
    max_open_positions = max_open_positions or settings.MAX_OPEN_POSITIONS
    max_portfolio_heat = max_portfolio_heat or settings.MAX_PORTFOLIO_HEAT
    
    # Check concurrent position limit
    if len(current_open_positions) >= max_open_positions:
        return False, f"Max concurrent positions ({max_open_positions}) reached"
    
    # Calculate current portfolio heat
    total_heat = sum(pos.get("risk_pct", 0.01) for pos in current_open_positions)
    proposed_heat = proposed_notional / 10000  # Rough estimate: assume 1% risk
    
    if total_heat + proposed_heat > max_portfolio_heat:
        return False, f"Portfolio heat ({total_heat + proposed_heat:.2%}) would exceed limit ({max_portfolio_heat:.2%})"
    
    return True, ""
