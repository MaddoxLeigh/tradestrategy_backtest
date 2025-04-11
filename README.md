# Trading Strategy Backtester

A backtesting framework for implementing and testing trading strategies. This project provides an implementation of the Moving Average Crossover strategy with RSI and momentum filters.

## Technical Overview

### Core Components
- Moving Average Crossover strategy with RSI and momentum filters
- Yahoo Finance data integration
- Performance metrics calculation
- Trade analysis and visualisation
- Parameter optimisation framework

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

## Advanced Usage

### Creating Custom Strategies
To create a custom strategy, inherit from the `BaseStrategy` class and implement the `generate_signals` method:

```python
from src.strategies.base_strategy import BaseStrategy
from src.indicators.technical_indicators import TechnicalIndicators

class MyCustomStrategy(BaseStrategy):
    def __init__(self, param1: int, param2: float):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        self.indicators = TechnicalIndicators()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Implement your strategy logic here
        signals = pd.DataFrame(index=data.index)
        signals['position'] = 0  # Initialize positions
        
        # Your strategy logic here
        
        return signals
```

### Data Requirements
The framework expects historical price data in the following format:

| Column    | Type      | Description                |
|-----------|-----------|----------------------------|
| timestamp | datetime  | Date and time of the price |
| open      | float     | Opening price             |
| high      | float     | Highest price             |
| low       | float     | Lowest price              |
| close     | float     | Closing price             |
| volume    | int       | Trading volume            |

### Visualization
The backtester provides visualization tools:
```python
backtester.plot_results()  # Shows equity curve and trades
```

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




