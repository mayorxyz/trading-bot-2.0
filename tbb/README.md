# TBB - Trading Bot Backbone

Production-grade cryptocurrency algorithmic trading bot backend for Bybit Perpetual Futures.

## Overview

TBB implements a systematic trading approach based on Smart Money Concepts (SMC) with:
- **Structure detection**: BOS, CHoCH, swing highs/lows
- **Order Blocks & FVG**: Institutional entry zones
- **Regime classification**: ADX + Hurst + ATR percentile
- **Volume analysis**: CVD, Delta, Order Book Imbalance
- **5-Pillar Confluence**: Weighted scoring system (0.0-1.0)
- **Risk management**: Position sizing, portfolio heat, circuit breakers
- **Live monitoring**: Rolling metrics, drift detection, Telegram alerts

## Tech Stack

- **Language**: Python 3.11+
- **Framework**: FastAPI (async)
- **Exchange**: Bybit Perpetual Futures (via `pybit` SDK)
- **Database**: SQLite (MVP) → TimescaleDB (production)
- **Indicators**: NumPy, Pandas, TA-Lib
- **Scheduler**: Asyncio task loops

## Installation

```bash
# Clone repository
git clone <repo-url>
cd tbb

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment file and configure
cp .env.example .env
# Edit .env with your Bybit API credentials

# Install TA-Lib (system dependency required)
# Ubuntu: sudo apt-get install libta-lib
# See: https://ta-lib.github.io/ta-lib-python/install.html
```

## Configuration

Edit `.env` file:

```env
# Bybit API (testnet recommended for testing)
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret
BYBIT_TESTNET=true

# Risk settings
ACCOUNT_RISK_PCT=0.01        # 1% per trade
MAX_CONCURRENT_POSITIONS=4
MAX_PORTFOLIO_HEAT=0.06      # 6% total open risk

# Symbols
TRADING_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT
```

## Project Structure

```
tbb/
├── src/tbb/
│   ├── core/           # Paths, config, logging
│   ├── data/           # OHLCV fetch, WebSocket, SQLite store
│   ├── indicators/     # Structure, FVG, OB, regime, volume, funding
│   ├── signals/        # MVS strategy, confluence scorer, pipeline
│   ├── risk/           # Position sizing, portfolio, circuit breaker
│   ├── execution/      # Order placement, slippage tracking
│   ├── monitoring/     # Metrics, drift, trade logger, alerts
│   ├── live/           # Main async runner loop
│   └── api/            # FastAPI REST endpoints
├── data/               # SQLite databases
├── logs/               # Application logs
├── tests/              # Unit tests
├── .env.example        # Environment template
├── requirements.txt    # Dependencies
└── README.md
```

## Usage

### Start the Bot

```bash
# Run the live trading loop
python -m src.tbb.live.runner
```

### API Endpoints

Start the FastAPI server:

```bash
uvicorn src.tbb.api.app:app --reload --host 0.0.0.0 --port 8000
```

Available endpoints:
- `GET /status` - Bot running state, active positions
- `GET /trades` - Paginated trade log
- `GET /metrics` - Live metrics (win rate, Sharpe, expectancy)
- `GET /signals` - Current MVS evaluation per symbol
- `GET /portfolio` - Open positions, correlation matrix
- `POST /pause` - Manually pause trading
- `POST /resume` - Resume after pause

## Strategy: Minimum Viable Strategy (MVS)

The core strategy uses 3 signals:

1. **HTF Order Block (4H)** - Structural bias from institutional levels
2. **FVG in displacement** - Entry zone within/near order block
3. **MSS/CHoCH (15m)** - Reversal confirmation on lower timeframe

**Entry**: Limit order at 50% FVG midpoint
**Stop**: max(Entry - 2×ATR, OB_low - 0.5×ATR)
**Target**: Prior liquidity pool (swing high/low)

**Performance Targets**:
- Win Rate: 55-65%
- R:R: 2.5:1 to 3:1
- Expectancy: +0.50R to +0.65R
- Max Drawdown: < 18%

## Risk Management

### Circuit Breakers
- Daily DD limit: -3% → pause new entries
- Consecutive losses: 8 → 4-hour pause
- Portfolio DD: -15% → halt all, manual review
- Slippage threshold: 30% worse than expected → pause

### Position Sizing
- 1% account risk per trade
- Adjusted down for high funding rates
- Reduced for correlated positions (>0.7 correlation)

### Trade Management
- Partial TP: 50% at 1R → move SL to breakeven
- Trailing: 3× ATR chandelier on remainder
- Time exit: Close if PnL < +0.5R after 48 hours

## Monitoring

### Live Metrics (rolling 100 trades)
- Win rate
- Profit factor
- Expectancy (R-multiple)
- Sharpe ratio (60-day annualized)
- Max drawdown

### Drift Detection
- CUSUM on trade R-multiples
- Z-score on rolling expectancy
- Alert when live performance deviates from backtest

### Alerts (Telegram)
- **CRITICAL**: Circuit breaker triggers, liquidation risk, CUSUM alert
- **WARNING**: Win rate gap, Sharpe degradation, 6-7 consecutive losses
- **DAILY SUMMARY**: PnL, ROI, trade count, top/bottom trades

## Important Warnings

⚠️ **This is experimental software. Use at your own risk.**

- Always start on Bybit testnet
- Never use API keys with withdrawal permissions
- IP-whitelist your API keys
- Start with minimal capital
- Monitor closely during initial runs
- Backtest thoroughly before live deployment

## Development

```bash
# Run tests
pytest tests/

# Code formatting
black src/
isort src/

# Linting
flake8 src/
```

## License

MIT License - See LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and consult with a licensed financial advisor before making investment decisions.
