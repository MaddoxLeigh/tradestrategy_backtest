import pandas as pd
import matplotlib.pyplot as plt
from src.data.data_processor import DataProcessor
from src.strategies.moving_average_crossover import MovingAverageCrossoverStrategy
from src.backtester import Backtester

def main():
    # Load and process data
    data_processor = DataProcessor()
    data = data_processor.load_data('path/to/your/data.csv')
    
    # Create and optimize strategy
    strategy = MovingAverageCrossoverStrategy()
    best_params = strategy.optimize_parameters(data)
    
    print("Best Parameters:")
    print(f"Fast Period: {best_params['fast_period']}")
    print(f"Slow Period: {best_params['slow_period']}")
    print(f"Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
    print(f"Total Return: {best_params['total_return']:.2%}")
    print(f"Max Drawdown: {best_params['max_drawdown']:.2%}")
    
    # Create strategy with optimized parameters
    optimized_strategy = MovingAverageCrossoverStrategy(
        fast_period=best_params['fast_period'],
        slow_period=best_params['slow_period']
    )
    
    # Run backtest
    backtester = Backtester(optimized_strategy)
    results = backtester.run(data)
    
    # Print results
    print("\nBacktest Results:")
    print(backtester.get_summary())
    
    # Plot results
    backtester.plot_results()
    
    # Plot strategy signals
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['close'], label='Price')
    plt.plot(data.index, results['portfolio']['fast_ma'], label='Fast MA')
    plt.plot(data.index, results['portfolio']['slow_ma'], label='Slow MA')
    
    # Plot buy signals
    buy_signals = results['portfolio'][results['portfolio']['position'] == 1]
    plt.scatter(buy_signals.index, buy_signals['price'], 
                marker='^', color='g', label='Buy Signal')
    
    # Plot sell signals
    sell_signals = results['portfolio'][results['portfolio']['position'] == -1]
    plt.scatter(sell_signals.index, sell_signals['price'], 
                marker='v', color='r', label='Sell Signal')
    
    plt.title('Moving Average Crossover Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main() 