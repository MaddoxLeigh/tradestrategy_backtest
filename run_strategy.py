import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.strategies.moving_average_crossover import MovingAverageCrossoverStrategy
from src.backtester import Backtester
import yfinance as yf
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance.
    
    Args:
        symbol (str): Stock symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        pd.DataFrame: Stock data
    """
    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
    
    try:
        # Download data with auto_adjust=True
        data = yf.download(symbol, start=start_date, end=end_date, interval='1d', auto_adjust=True)
        
        if data.empty:
            raise ValueError(f"No data found for {symbol} in the specified date range")
        
        # Ensure required columns are present
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Got: {data.columns}")
        
        # Rename columns to lowercase
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        data = data.rename(columns=column_mapping)
        
        logger.info(f"Successfully downloaded {len(data)} rows of data")
        return data
        
    except Exception as e:
        logger.error(f"Error in fetch_data: {str(e)}")
        raise

def main():
    # Set date range for last 6 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months ago
    
    # Format dates as strings
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    logger.info(f"Date range: {start_date_str} to {end_date_str}")
    
    # Get stock symbol from user input
    while True:
        symbol = input("Enter stock symbol (e.g., AAPL, TSLA, MSFT): ").upper()
        if symbol:
            break
        print("Please enter a valid ticker symbol")
    
    # Fetch data
    try:
        data = fetch_data(symbol, start_date_str, end_date_str)
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        sys.exit(1)
    
    # Initialize strategy with only the required parameters
    strategy = MovingAverageCrossoverStrategy(
        fast_period=10,  # Shorter periods for more signals
        slow_period=20
    )
    
    # Initialize backtester
    backtester = Backtester(strategy, initial_capital=100000.0)
    
    # Run backtest
    logger.info("Running backtest...")
    results = backtester.run(data)
    
    # Print results
    print("\nBacktest Results:")
    print("-" * 50)
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.1%}")
    print(f"Average Trade Return: {results['avg_trade_return']:.2%}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    
    # Print detailed trade list
    if 'trades' in results and not results['trades'].empty:
        print("\nDetailed Trade List:")
        print("-" * 100)
        trades_df = results['trades'].copy()
        
        # Format the trades DataFrame for better readability
        trades_df['entry_date'] = trades_df['entry_date'].dt.strftime('%Y-%m-%d')
        trades_df['exit_date'] = trades_df['exit_date'].dt.strftime('%Y-%m-%d')
        trades_df['entry_price'] = trades_df['entry_price'].round(2)
        trades_df['exit_price'] = trades_df['exit_price'].round(2)
        trades_df['return'] = (trades_df['return'] * 100).round(2)
        trades_df['position'] = trades_df['position'].map({1: 'LONG', -1: 'SHORT'})
        
        # Calculate position size (1% of initial capital)
        position_size = 100000.0 * 0.01
        trades_df['position_size'] = position_size
        
        # Print the formatted trades
        print(trades_df[['entry_date', 'exit_date', 'position', 'entry_price', 'exit_price', 
                        'position_size', 'return']].to_string(index=False))
    else:
        print("\nNo trades were executed during the backtest period.")
    
    # Plot results
    backtester.plot_results(results)

if __name__ == "__main__":
    main() 