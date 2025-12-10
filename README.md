# Crypto AI Bot - Starter Skeleton
This is a starter skeleton for a Binance + OpenAI + RL full-automatic trading system.
**Purpose**: development & paper-trade pipeline (NOT production-ready).

## Structure
- `data/` - data downloader
- `envs/` - gym environment
- `agents/` - training scripts
- `backtest/` - backtesting utilities
- `executor/` - order manager (ccxt)
- `config/` - configuration
- `scripts/` - scripts to run paper-trade
- `.env.example` - example environment variables

## Quickstart (paper-trade)
1. Create virtualenv and install requirements: `pip install -r requirements.txt`
2. Copy `.env.example` -> `.env` and fill your keys (use testnet keys for safety)
3. Run data downloader: `python data/downloader.py`
4. Run a simple paper-trade run: `python scripts/run_papertrade.py`

## Important
- Always paper-trade and backtest thoroughly before using real funds.
- Use least-privilege API keys and enable IP restrictions.
- This skeleton intentionally keeps things simple to get you started.
