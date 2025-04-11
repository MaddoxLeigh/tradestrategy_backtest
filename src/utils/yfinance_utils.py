import yfinance as yf
import pandas as pd
from typing import Optional

def get_yfinance_data(
    symbol: str,
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = '1d'
) -> pd.DataFrame:
    """
    Fetch data from yfinance and convert it to the required format.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to today.
        interval (str): Data interval ('1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo')
        
    Returns:
        pd.DataFrame: Data in the required format
    """
    # Fetch data from yfinance
    ticker = yf.Ticker(symbol)
    data = ticker.history(
        start=start_date,
        end=end_date,
        interval=interval
    )
    
    # Reset index to make date a column
    data = data.reset_index()
    
    # Rename columns to match our required format
    data = data.rename(columns={
        'Date': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # Ensure timestamp is in datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Sort by timestamp
    data = data.sort_values('timestamp')
    
    # Reset index
    data = data.reset_index(drop=True)
    
    return data

def save_yfinance_data(
    symbol: str,
    start_date: str,
    end_date: Optional[str] = None,
    interval: str = '1d',
    output_path: str = 'data.csv'
) -> None:
    """
    Fetch data from yfinance and save it to a CSV file.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to today.
        interval (str): Data interval ('1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo')
        output_path (str): Path to save the CSV file
    """
    data = get_yfinance_data(symbol, start_date, end_date, interval)
    data.to_csv(output_path, index=False) 