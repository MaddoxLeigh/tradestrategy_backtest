from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    All strategies must implement the generate_signals method.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the strategy with optional parameters.
        
        Args:
            params (Dict[str, Any], optional): Strategy parameters. Defaults to None.
        """
        self.params = params or {}
        self.signals = None
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the input data.
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            pd.DataFrame: DataFrame containing trading signals
        """
        pass
    
    def calculate_position_size(self, capital: float, risk_per_trade: float) -> float:
        """
        Calculate position size based on capital and risk per trade.
        
        Args:
            capital (float): Available capital
            risk_per_trade (float): Risk per trade as a percentage
            
        Returns:
            float: Position size
        """
        return capital * (risk_per_trade / 100)
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the input data has all required columns.
        
        Args:
            data (pd.DataFrame): Input data to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in data.columns for col in required_columns) 