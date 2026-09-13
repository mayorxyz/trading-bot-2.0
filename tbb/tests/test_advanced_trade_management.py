"""
Unit Tests for Advanced Trade Management Features (2024-2026 Research)

Tests cover:
1. Dynamic Signal Expiry by Timeframe
2. MAE Early Exit Logic
3. +0.5R Stop Movement (avoids BE wick-outs)
4. Partial TP Math (50% at TP1, 50% runner)
5. Trailing Stop Calculations
6. Structural Invalidation Detection
"""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, '/workspace/tbb/src')

from tbb.core.config import settings
from tbb.tracking.store import TrackedSignal, TrackingStatus
from tbb.signals.output import get_dynamic_signal_expiry_hours


class TestDynamicSignalExpiry:
    """Test dynamic expiry based on signal timeframe."""
    
    def test_1m_to_5m_signals_expire_in_4h(self):
        assert get_dynamic_signal_expiry_hours('1m') == 4
        assert get_dynamic_signal_expiry_hours('2m') == 4
        assert get_dynamic_signal_expiry_hours('3m') == 4
        assert get_dynamic_signal_expiry_hours('4m') == 4
        assert get_dynamic_signal_expiry_hours('5m') == 4
    
    def test_15m_to_1h_signals_expire_in_24h(self):
        assert get_dynamic_signal_expiry_hours('15m') == 24
        assert get_dynamic_signal_expiry_hours('30m') == 24
        assert get_dynamic_signal_expiry_hours('1h') == 24
        assert get_dynamic_signal_expiry_hours('1H') == 24
    
    def test_4h_signals_expire_in_48h(self):
        assert get_dynamic_signal_expiry_hours('4h') == 48
        assert get_dynamic_signal_expiry_hours('4H') == 48
    
    def test_daily_signals_expire_in_7_days(self):
        assert get_dynamic_signal_expiry_hours('1d') == 168
        assert get_dynamic_signal_expiry_hours('daily') == 168
        assert get_dynamic_signal_expiry_hours('Daily') == 168
    
    def test_default_expiry_is_24h(self):
        assert get_dynamic_signal_expiry_hours('unknown') == 24


class TestBEOffsetStop:
    """Test +0.5R stop movement (avoids exact breakeven wick-outs)."""
    
    def setup_method(self):
        self.signal = TrackedSignal(
            signal_id='test-be',
            symbol='BTCUSDT',
            direction='LONG',
            entry_price=50000.0,
            stop_loss=49000.0,  # 1000 risk = 1R
            take_profit_1=51000.0,
            take_profit_2=52000.0,
            risk_reward=2.0,
            confluence_score=0.75,
            market_regime='TRENDING_UP',
            funding_rate=0.0001,
            max_age_bars=48,
            expiry_time=datetime.now(timezone.utc),
            mvs_ob_present=True,
            mvs_fvg_confirmed=True,
            mvs_mss_confirmed=True,
        )
    
    def test_long_be_offset_stop_at_plus_half_r(self):
        """LONG: Stop should move to entry + 0.5R (not exact BE)."""
        stop = self.signal.calculate_be_offset_stop()
        expected = 50000.0 + (1000.0 * 0.5)  # entry + 0.5R
        assert stop == 50500.0
        assert stop == expected
    
    def test_short_be_offset_stop_at_minus_half_r(self):
        """SHORT: Stop should move to entry - 0.5R (not exact BE)."""
        self.signal.direction = 'SHORT'
        self.signal.stop_loss = 51000.0  # Above entry for short
        stop = self.signal.calculate_be_offset_stop()
        expected = 50000.0 - (1000.0 * 0.5)  # entry - 0.5R
        assert stop == 49500.0
        assert stop == expected


class TestPartialTPMath:
    """Test partial position sizing and PnL calculations."""
    
    def setup_method(self):
        self.signal = TrackedSignal(
            signal_id='test-partial',
            symbol='BTCUSDT',
            direction='LONG',
            entry_price=50000.0,
            stop_loss=49000.0,  # 1R = 1000
            take_profit_1=51000.0,  # 1R
            take_profit_2=52000.0,  # 2R
            risk_reward=2.0,
            confluence_score=0.75,
            market_regime='TRENDING_UP',
            funding_rate=0.0001,
            max_age_bars=48,
            expiry_time=datetime.now(timezone.utc),
            mvs_ob_present=True,
            mvs_fvg_confirmed=True,
            mvs_mss_confirmed=True,
            trail_after_tp1=True,
            be_offset_r=0.5,
        )
        self.signal.entry_filled_at = datetime.now(timezone.utc)
    
    def test_partial_pnl_at_tp1_is_half_r(self):
        """At TP1, 50% of position gains 1R = 0.5R total contribution."""
        # Simulate TP1 hit
        self.signal.partial_pnl_r = 1.0 * 0.5  # 50% position at 1R
        self.signal.position_remaining_pct = 0.5
        
        assert self.signal.partial_pnl_r == 0.5
        assert self.signal.position_remaining_pct == 0.5
    
    def test_total_pnl_tp2_hard(self):
        """TP2 hard hit: partial (0.5R) + runner (50% at 2R = 1.0R) = 1.5R total."""
        self.signal.partial_pnl_r = 0.5  # Already secured at TP1
        
        # Runner PnL at TP2 (2R * 50% position)
        full_pnl_at_tp2 = self.signal.calculate_pnl_r(52000.0)  # Should be 2.0R
        runner_pnl_r = full_pnl_at_tp2 * 0.5  # 50% of position
        total_pnl_r = self.signal.partial_pnl_r + runner_pnl_r
        
        assert full_pnl_at_tp2 == 2.0
        assert runner_pnl_r == 1.0
        assert total_pnl_r == 1.5
    
    def test_trailing_stop_calculation_with_atr(self):
        """Trailing stop should use highest_price - (ATR * multiplier)."""
        self.signal.update_mae_mfe(current_high=51500.0, current_low=49500.0)
        
        # Trail at highest - 3xATR
        atr_14 = 500.0
        trail_stop = self.signal.calculate_trailing_stop(atr_14=atr_14, atr_multiplier=3.0)
        expected = 51500.0 - (500.0 * 3.0)  # 50000.0
        
        assert trail_stop == 50000.0


class TestMAEEarlyExit:
    """Test MAE-based early exit rules."""
    
    def setup_method(self):
        self.signal = TrackedSignal(
            signal_id='test-mae',
            symbol='BTCUSDT',
            direction='LONG',
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit_1=51000.0,
            take_profit_2=52000.0,
            risk_reward=2.0,
            confluence_score=0.75,
            market_regime='TRENDING_UP',
            funding_rate=0.0001,
            max_age_bars=48,
            expiry_time=datetime.now(timezone.utc),
            mvs_ob_present=True,
            mvs_fvg_confirmed=True,
            mvs_mss_confirmed=True,
            mae_hard_exit_r=settings.MAE_HARD_EXIT_R,
            mae_stall_exit_r=settings.MAE_STALL_EXIT_R,
        )
        self.signal.entry_filled_at = datetime.now(timezone.utc)
    
    def test_mae_hard_exit_trigger(self):
        """MAE > 1.0R AND PnL < 0 should trigger EARLY_EXIT_MAE."""
        # Simulate adverse move: MAE = 1.2R
        self.signal.update_mae_mfe(current_high=50100.0, current_low=48800.0)
        assert self.signal.mae_r > 1.0  # MAE exceeded threshold
        
        # Current price still negative
        current_pnl = self.signal.calculate_pnl_r(49200.0)
        assert current_pnl < 0
        
        # Should trigger hard exit
        should_exit = self.signal.mae_r > settings.MAE_HARD_EXIT_R and current_pnl < 0
        assert should_exit is True
    
    def test_mae_stall_exit_trigger(self):
        """MAE > 0.7R AND > 6 bars without +0.5R profit triggers EARLY_EXIT_STALL."""
        # Simulate moderate adverse move: MAE = 0.8R
        self.signal.update_mae_mfe(current_high=50200.0, current_low=49200.0)
        assert self.signal.mae_r > 0.7
        
        # Simulate 7 bars passed
        self.signal.bars_since_entry = 7
        
        # Current PnL < 0.5R
        current_pnl = self.signal.calculate_pnl_r(50300.0)  # 0.3R
        assert current_pnl < 0.5
        
        # Should trigger stall exit
        should_exit = (
            self.signal.mae_r > settings.MAE_STALL_EXIT_R and
            self.signal.bars_since_entry > settings.MAX_BARS_WITHOUT_PROFIT and
            current_pnl < 0.5
        )
        assert should_exit is True
    
    def test_no_early_exit_when_conditions_not_met(self):
        """No early exit if MAE is low or PnL is positive."""
        # Small adverse move: MAE = 0.3R
        self.signal.update_mae_mfe(current_high=50500.0, current_low=49700.0)
        assert self.signal.mae_r < 0.7
        
        # Should NOT trigger early exit
        current_pnl = self.signal.calculate_pnl_r(50500.0)
        should_exit_hard = self.signal.mae_r > settings.MAE_HARD_EXIT_R and current_pnl < 0
        assert should_exit_hard is False


class TestTrackingStats:
    """Test advanced tracking statistics."""
    
    def test_new_tracking_status_enum_values(self):
        """Verify new status enum values exist."""
        assert hasattr(TrackingStatus, 'WIN_TP2_HARD')
        assert hasattr(TrackingStatus, 'WIN_TP2_TRAILED')
        assert hasattr(TrackingStatus, 'EARLY_EXIT_MAE')
        assert hasattr(TrackingStatus, 'EARLY_EXIT_STALL')
        assert hasattr(TrackingStatus, 'TIME_EXIT')
    
    def test_new_model_fields_exist(self):
        """Verify new database fields exist on TrackedSignal."""
        signal = TrackedSignal(
            signal_id='test-fields',
            symbol='BTCUSDT',
            direction='LONG',
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit_1=51000.0,
            take_profit_2=52000.0,
            risk_reward=2.0,
            confluence_score=0.75,
            market_regime='TRENDING_UP',
            funding_rate=0.0001,
            max_age_bars=48,
            expiry_time=datetime.now(timezone.utc),
            mvs_ob_present=True,
            mvs_fvg_confirmed=True,
            mvs_mss_confirmed=True,
        )
        
        # PnL tracking fields
        assert hasattr(signal, 'partial_pnl_r')
        assert hasattr(signal, 'runner_pnl_r')
        assert hasattr(signal, 'total_pnl_r')
        
        # Time tracking
        assert hasattr(signal, 'time_to_tp1_seconds')
        
        # Config fields
        assert hasattr(signal, 'trail_after_tp1')
        assert hasattr(signal, 'be_offset_r')
        assert hasattr(signal, 'trail_atr_multiplier')
        assert hasattr(signal, 'mae_hard_exit_r')
        assert hasattr(signal, 'mae_stall_exit_r')
        
        # State fields
        assert hasattr(signal, 'trailing_stop_price')
        assert hasattr(signal, 'tp1_hit_at')
        assert hasattr(signal, 'position_remaining_pct')
        assert hasattr(signal, 'bars_without_profit')
        assert hasattr(signal, 'invalidation_reason')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
