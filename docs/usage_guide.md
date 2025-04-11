# eToro Trading Strategy Backtester - Usage Guide

This guide will help you get started with using the eToro Trading Strategy Backtester framework.

## Table of Contents
1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Creating Custom Strategies](#creating-custom-strategies)
4. [Data Requirements](#data-requirements)
5. [Performance Metrics](#performance-metrics)
6. [Optimization](#optimization)
7. [Visualization](#visualization)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd project
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

## Basic Usage

The framework provides a simple interface for backtesting trading strategies. Here's a basic example:

```python
from src.data.data_processor import DataProcessor
from src.strategies.moving_average_crossover import MovingAverageCrossoverStrategy
from src.backtester import Backtester

# Load and process data
data_processor = DataProcessor()
data = data_processor.load_data('path/to/your/data.csv')

# Create strategy
strategy = MovingAverageCrossoverStrategy(fast_period=20, slow_period=50)

# Run backtest
backtester = Backtester(strategy)
results = backtester.run(data)

# Print results
print(backtester.get_summary())

# Plot results
backtester.plot_results()
```

## Creating Custom Strategies

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

## Data Requirements

The framework expects historical price data in the following format:

| Column    | Description                |
|-----------|----------------------------|
| timestamp | Date and time of the price |
| open      | Opening price             |
| high      | Highest price             |
| low       | Lowest price              |
| close     | Closing price             |
| volume    | Trading volume            |

## Performance Metrics

The backtester calculates several performance metrics:

- **Total Return**: Overall return on investment
- **Annual Return**: Annualized return
- **Sharpe Ratio**: Risk-adjusted return
- **Max Drawdown**: Maximum peak-to-trough decline
- **Number of Trades**: Total number of trades executed

## Optimization

The framework includes tools for optimizing strategy parameters:

```python
# Optimize strategy parameters
best_params = strategy.optimize_parameters(data)

# Create strategy with optimized parameters
optimized_strategy = MovingAverageCrossoverStrategy(
    fast_period=best_params['fast_period'],
    slow_period=best_params['slow_period']
)
```

## Visualization

The framework provides built-in visualization tools:

```python
# Plot portfolio value and returns
backtester.plot_results()

# Plot strategy signals
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['close'], label='Price')
plt.plot(data.index, results['portfolio']['indicator1'], label='Indicator 1')
plt.plot(data.index, results['portfolio']['indicator2'], label='Indicator 2')

# Plot buy/sell signals
buy_signals = results['portfolio'][results['portfolio']['position'] == 1]
sell_signals = results['portfolio'][results['portfolio']['position'] == -1]

plt.scatter(buy_signals.index, buy_signals['price'], 
            marker='^', color='g', label='Buy Signal')
plt.scatter(sell_signals.index, sell_signals['price'], 
            marker='v', color='r', label='Sell Signal')

plt.legend()
plt.show()
```

## Best Practices

1. **Data Quality**: Ensure your data is clean and complete
2. **Parameter Optimization**: Use the optimization tools to find optimal parameters
3. **Risk Management**: Implement proper position sizing and risk management
4. **Backtest Period**: Use sufficient historical data for meaningful results
5. **Transaction Costs**: Consider transaction costs in your strategy
6. **Out-of-Sample Testing**: Test your strategy on unseen data

## Troubleshooting

Common issues and solutions:

1. **Data Format Errors**: Ensure your data matches the required format
2. **Missing Dependencies**: Install all required packages from requirements.txt
3. **Memory Issues**: For large datasets, consider using data sampling
4. **Performance Issues**: Optimize your strategy code for better performance

## Support

For additional support or questions, please:
1. Check the documentation
2. Review the example code
3. Submit an issue on the repository 