import pandas as pd
import numpy as np
from typing import Optional, Union
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Class for processing and preparing historical trading data.
    """
    
    def __init__(self):
        pass
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load historical data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info(f"Loading data from {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        data = pd.read_csv(file_path)
        logger.info(f"Loaded {len(data)} rows of data")
        return self._validate_and_clean_data(data)
    
    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the loaded data.
        
        Args:
            data (pd.DataFrame): Raw data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        logger.info("Validating and cleaning data...")
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in data.columns]
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError("Data must contain all required columns: timestamp, open, high, low, close, volume")
        
        # Convert timestamp to datetime
        logger.info("Converting timestamp to datetime")
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Sort by timestamp
        logger.info("Sorting by timestamp")
        data = data.sort_values('timestamp')
        
        # Remove duplicates
        logger.info("Removing duplicates")
        data = data.drop_duplicates(subset=['timestamp'])
        
        # Handle missing values using new methods
        logger.info("Handling missing values")
        data = data.ffill()  # Forward fill
        data = data.bfill()  # Backward fill for any remaining NaNs
        
        # Ensure numeric columns are numeric
        logger.info("Converting numeric columns")
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove any rows with NaN values after conversion
        logger.info("Removing rows with NaN values")
        data = data.dropna()
        
        # Ensure all prices are positive
        logger.info("Validating price values")
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            data = data[data[col] > 0]
        
        # Ensure volume is non-negative
        logger.info("Validating volume values")
        data = data[data['volume'] >= 0]
        
        # Reset index
        logger.info("Resetting index")
        data = data.reset_index(drop=True)
        
        logger.info(f"Data cleaning complete. Final shape: {data.shape}")
        return data
    
    def resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample data to a different timeframe.
        
        Args:
            data (pd.DataFrame): Input data
            timeframe (str): Target timeframe (e.g., '1H', '4H', '1D')
            
        Returns:
            pd.DataFrame: Resampled data
        """
        resampled = data.set_index('timestamp').resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        
        return resampled
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add common technical indicators to the data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with added indicators
        """
        logger.info("Adding technical indicators...")
        
        # Calculate moving averages
        logger.info("Calculating moving averages")
        data['sma_20'] = data['close'].rolling(window=20, min_periods=1).mean()
        data['sma_50'] = data['close'].rolling(window=50, min_periods=1).mean()
        
        # Calculate returns
        logger.info("Calculating returns")
        data['returns'] = data['close'].pct_change().fillna(0)
        
        # Calculate log returns
        logger.info("Calculating log returns")
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1)).fillna(0)
        
        # Calculate volatility (20-day rolling)
        logger.info("Calculating volatility")
        data['volatility'] = data['returns'].rolling(window=20, min_periods=1).std() * np.sqrt(252)
        
        logger.info("Technical indicators added successfully")
        return data
    
    def split_data(self, data: pd.DataFrame, train_ratio: float = 0.7) -> tuple:
        """
        Split data into training and testing sets.
        
        Args:
            data (pd.DataFrame): Input data
            train_ratio (float): Ratio of data to use for training
            
        Returns:
            tuple: (train_data, test_data)
        """
        split_idx = int(len(data) * train_ratio)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        return train_data, test_data 