# Trading Strategy Backtester

A professional-grade backtesting framework for implementing and testing trading strategies. This project provides a robust implementation of the Moving Average Crossover strategy with RSI and momentum filters.

## Technical Overview

### Core Components
- Moving Average Crossover strategy with RSI and momentum filters
- Yahoo Finance data integration
- Performance metrics calculation
- Trade analysis and visualization
- Parameter optimization framework

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MaddoxLeigh/tradestrategy_backtest.git
cd tradestrategy_backtest
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Implementation Details

### Strategy Components
- **Moving Averages**: 10 and 20-period EMAs for trend identification
- **RSI**: 14-period window for momentum analysis
- **Price Momentum**: 5-period lookback for trend confirmation

### Signal Generation
- Long entry: EMA crossover + RSI < 30 or momentum > 2%
- Short entry: EMA crossunder + RSI > 70 or momentum < -2%

### Performance Metrics
- Total return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Profit factor
- Average trade return

## Usage

1. Run the strategy:
```bash
python run_strategy.py
```

2. Enter a stock symbol when prompted (e.g., AAPL, TSLA, MSFT)

3. View results:
- Equity curve
- Trade history
- Performance metrics
- Technical indicators

## Project Structure
```
project/
├── src/                    # Source code
│   ├── strategies/        # Trading strategies
│   ├── backtester.py     # Backtesting engine
│   └── utils/            # Utility functions
├── docs/                  # Documentation
├── examples/             # Example scripts
└── requirements.txt      # Dependencies
```

## Technical Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- yfinance
- scipy

## Development
1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Run tests
5. Submit a pull request

## Testing
Run the test suite:
```bash
python -m pytest tests/
```

## Performance Optimization
- Vectorized operations for signal generation
- Efficient data handling with pandas
- Optimized parameter search
- Memory-efficient backtesting

## Contributing
1. Follow PEP 8 style guide
2. Add tests for new features
3. Update documentation
4. Maintain backward compatibility 