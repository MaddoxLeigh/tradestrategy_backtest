# Trading Strategy Backtester

A simple but powerful tool to test trading strategies using real market data. This project helps you backtest the Moving Average Crossover strategy with RSI and momentum filters.

## What's Inside
- 📈 Moving Average Crossover strategy with RSI and momentum
- 📊 Real-time data from Yahoo Finance
- 📉 Performance tracking and visualization
- 🎯 Trade analysis and statistics

## Quick Start
1. Install the requirements:
```bash
pip install -r requirements.txt
```

2. Run the strategy:
```bash
python run_strategy.py
```

3. Enter a stock symbol when prompted (e.g., AAPL, TSLA, MSFT)

## How It Works
The strategy looks at three main things to make trading decisions:
1. Moving averages (trend following)
2. RSI (momentum indicator)
3. Price momentum

When all conditions line up, it gives you a buy or sell signal. The backtester then:
- Tracks your trades
- Calculates your returns
- Shows you detailed performance metrics
- Creates nice visualizations of your results

## Features
- 🚀 Easy to use interface
- 📱 Works with any stock on Yahoo Finance
- 📊 Shows detailed trade information
- 📈 Creates beautiful charts of your results
- 💰 Tracks performance metrics like:
  - Total return
  - Win rate
  - Sharpe ratio
  - Maximum drawdown
  - Profit factor

## Project Structure
```
project/
├── src/                    # Main code
│   ├── strategies/        # Trading strategies
│   ├── backtester.py     # Backtesting engine
│   └── utils/            # Helper functions
├── docs/                  # Documentation
└── requirements.txt       # Python packages needed
```

## Tips for Best Results
1. Start with well-known stocks (AAPL, MSFT, TSLA)
2. Use the last 6 months of data for testing
3. Pay attention to the win rate and profit factor
4. Look at the charts to understand when the strategy works best

## Need Help?
Check out the detailed strategy explanation in `docs/strategy_explanation.md` to understand exactly how the strategy works.

## Remember
- This is a tool for testing and learning
- Past performance doesn't guarantee future results
- Always do your own research
- Start with paper trading before using real money 