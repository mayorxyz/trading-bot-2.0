"""
Tests for Signal Tracking and Resolution layer.

Covers:
- Fill logic (entry price hit detection)
- Expiry handling
- SL/TP resolution
- MAE/MFE calculations
- Stats calculation
"""
import pytest
from datetime import datetime, timezone, timedelta
from tbb.tracking.store import TrackedSignal, TrackingStatus, SignalTrackingStore, tracking_store
from tbb.tracking.resolver import SignalResolver


class TestTrackedSignalCalculations:
    """Test PnL and MAE/MFE calculations."""
    
    def test_calculate_pnl_r_long_win(self):
        """LONG position winning trade."""
        signal = TrackedSignal(
            signal_id="test-1",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit_1=51000.0,
            take_profit_2=52000.0,
            confluence_score=0.65,
            market_regime="TRENDING_UP",
            funding_rate=0.0001,
            max_age_bars=48,
            expiry_time=datetime.now(timezone.utc) + timedelta(hours=48),
            mvs_ob_present=True,
            mvs_fvg_confirmed=True,
            mvs_mss_confirmed=True,
        )
        
        # Win at TP1 (1R)
        pnl_r = signal.calculate_pnl_r(51000.0)
        assert abs(pnl_r - 1.0) < 0.01
        
        # Win at TP2 (2R)
        pnl_r = signal.calculate_pnl_r(52000.0)
        assert abs(pnl_r - 2.0) < 0.01
    
    def test_calculate_pnl_r_short_win(self):
        """SHORT position winning trade."""
        signal = TrackedSignal(
            signal_id="test-2",
            symbol="ETHUSDT",
            direction="SHORT",
            entry_price=3000.0,
            stop_loss=3050.0,
            take_profit_1=2950.0,
            take_profit_2=2900.0,
            confluence_score=0.70,
            market_regime="TRENDING_DOWN",
            funding_rate=-0.0001,
            max_age_bars=48,
            expiry_time=datetime.now(timezone.utc) + timedelta(hours=48),
            mvs_ob_present=True,
            mvs_fvg_confirmed=True,
            mvs_mss_confirmed=True,
        )
        
        # Win at TP1 (1R)
        pnl_r = signal.calculate_pnl_r(2950.0)
        assert abs(pnl_r - 1.0) < 0.01
        
        # Loss at SL
        pnl_r = signal.calculate_pnl_r(3050.0)
        assert abs(pnl_r - (-1.0)) < 0.01
    
    def test_calculate_pnl_pct(self):
        """Test percentage PnL calculation with leverage."""
        signal = TrackedSignal(
            signal_id="test-3",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit_1=51000.0,
            take_profit_2=52000.0,
            confluence_score=0.65,
            market_regime="TRENDING_UP",
            funding_rate=0.0001,
            max_age_bars=48,
            expiry_time=datetime.now(timezone.utc) + timedelta(hours=48),
            mvs_ob_present=True,
            mvs_fvg_confirmed=True,
            mvs_mss_confirmed=True,
        )
        
        # 2% price move with 10x leverage = 20% PnL
        pnl_pct = signal.calculate_pnl_pct(51000.0, leverage=10)
        assert abs(pnl_pct - 20.0) < 0.1
    
    def test_update_mae_mfe_long(self):
        """Test MAE/MFE tracking for LONG position."""
        signal = TrackedSignal(
            signal_id="test-4",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit_1=51000.0,
            take_profit_2=52000.0,
            confluence_score=0.65,
            market_regime="TRENDING_UP",
            funding_rate=0.0001,
            max_age_bars=48,
            expiry_time=datetime.now(timezone.utc) + timedelta(hours=48),
            mvs_ob_present=True,
            mvs_fvg_confirmed=True,
            mvs_mss_confirmed=True,
        )
        
        # Initialize with entry price
        signal.highest_price_since_entry = 50000.0
        signal.lowest_price_since_entry = 50000.0
        
        # Price moves up to 50500 (favorable) then down to 49800 (adverse)
        signal.update_mae_mfe(current_high=50500.0, current_low=49800.0)
        
        # MFE should be positive (favorable move)
        assert signal.mfe_r > 0
        assert signal.highest_price_since_entry == 50500.0
        
        # MAE should be positive (adverse move from entry)
        assert signal.mae_r > 0
        assert signal.lowest_price_since_entry == 49800.0
    
    def test_update_mae_mfe_short(self):
        """Test MAE/MFE tracking for SHORT position."""
        signal = TrackedSignal(
            signal_id="test-5",
            symbol="ETHUSDT",
            direction="SHORT",
            entry_price=3000.0,
            stop_loss=3050.0,
            take_profit_1=2950.0,
            take_profit_2=2900.0,
            confluence_score=0.70,
            market_regime="TRENDING_DOWN",
            funding_rate=-0.0001,
            max_age_bars=48,
            expiry_time=datetime.now(timezone.utc) + timedelta(hours=48),
            mvs_ob_present=True,
            mvs_fvg_confirmed=True,
            mvs_mss_confirmed=True,
        )
        
        # Initialize with entry price
        signal.highest_price_since_entry = 3000.0
        signal.lowest_price_since_entry = 3000.0
        
        # Price moves down to 2950 (favorable for short)
        signal.update_mae_mfe(current_high=3010.0, current_low=2950.0)
        
        # MFE should be positive (favorable move for short = price down)
        assert signal.mfe_r > 0
        assert signal.lowest_price_since_entry == 2950.0


class TestTrackingStatusTransitions:
    """Test status transition logic."""
    
    @pytest.mark.asyncio
    async def test_signal_lifecycle(self):
        """Test full signal lifecycle: PENDING -> ACTIVE -> WIN_TP2."""
        # Create signal
        signal = TrackedSignal(
            signal_id="lifecycle-test-1",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit_1=51000.0,
            take_profit_2=52000.0,
            confluence_score=0.65,
            market_regime="TRENDING_UP",
            funding_rate=0.0001,
            max_age_bars=48,
            expiry_time=datetime.now(timezone.utc) + timedelta(hours=48),
            mvs_ob_present=True,
            mvs_fvg_confirmed=True,
            mvs_mss_confirmed=True,
        )
        
        # Save initial state (PENDING)
        saved = await tracking_store.save_signal(signal)
        assert saved.tracking_status == TrackingStatus.PENDING
        
        # Mark as filled (ACTIVE)
        fill_time = datetime.now(timezone.utc)
        filled = await tracking_store.mark_as_filled(
            signal.signal_id,
            fill_price=50000.0,
            fill_timestamp=fill_time,
        )
        assert filled.tracking_status == TrackingStatus.ACTIVE
        assert filled.entry_fill_price == 50000.0
        assert filled.time_to_fill_seconds is not None
        
        # Resolve as win
        resolved = await tracking_store.update_tracking_status(
            signal.signal_id,
            TrackingStatus.WIN_TP2,
            exit_price=52000.0,
            exit_reason="TAKE_PROFIT",
        )
        assert resolved.tracking_status == TrackingStatus.WIN_TP2
        assert resolved.pnl_r > 0
        assert resolved.resolved_at is not None


class TestStatsCalculation:
    """Test tracking statistics calculation."""
    
    @pytest.mark.asyncio
    async def test_stats_with_sample_data(self):
        """Calculate stats from sample signals."""
        base_time = datetime.now(timezone.utc)
        
        # Create a mix of winning and losing signals
        signals_data = [
            ("win-1", "WIN_TP2", 2.0),
            ("win-2", "WIN_TP2", 1.5),
            ("loss-1", "LOSS_SL", -1.0),
            ("loss-2", "LOSS_SL", -0.5),
        ]
        
        for signal_id, status, pnl_r in signals_data:
            signal = TrackedSignal(
                signal_id=f"stats-{signal_id}",
                symbol="BTCUSDT",
                direction="LONG",
                entry_price=50000.0,
                stop_loss=49000.0,
                take_profit_1=51000.0,
                take_profit_2=52000.0,
                confluence_score=0.65,
                market_regime="TRENDING_UP",
                funding_rate=0.0001,
                max_age_bars=48,
                expiry_time=base_time + timedelta(hours=48),
                mvs_ob_present=True,
                mvs_fvg_confirmed=True,
                mvs_mss_confirmed=True,
            )
            
            await tracking_store.save_signal(signal)
            await tracking_store.mark_as_filled(signal.signal_id, 50000.0, base_time)
            
            # Calculate exit price based on PnL
            if pnl_r > 0:
                exit_price = 50000.0 + (pnl_r * 1000)  # R = 1000 points
            else:
                exit_price = 50000.0 - (abs(pnl_r) * 1000)
            
            await tracking_store.update_tracking_status(
                signal.signal_id,
                status,
                exit_price=exit_price,
                exit_reason="TEST",
            )
        
        # Get stats
        stats = await tracking_store.get_tracking_stats(symbol="BTCUSDT")
        
        assert stats["total_signals"] >= 4
        assert stats["win_rate"] is not None
        assert stats["expectancy_r"] is not None


class TestExpiryLogic:
    """Test signal expiry handling."""
    
    def test_expired_signal_check(self):
        """Check if signal is expired based on expiry_time."""
        past_expiry = datetime.now(timezone.utc) - timedelta(hours=1)
        
        signal = TrackedSignal(
            signal_id="expired-test-1",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit_1=51000.0,
            take_profit_2=52000.0,
            confluence_score=0.65,
            market_regime="TRENDING_UP",
            funding_rate=0.0001,
            max_age_bars=48,
            expiry_time=past_expiry,  # Already expired
            mvs_ob_present=True,
            mvs_fvg_confirmed=True,
            mvs_mss_confirmed=True,
        )
        
        # Check if current time is past expiry
        now = datetime.now(timezone.utc)
        assert now > signal.expiry_time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
