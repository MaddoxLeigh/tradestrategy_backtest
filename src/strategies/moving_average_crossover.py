import pandas as pd
import numpy as np
from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)

class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover strategy with RSI and price momentum filters.
    """
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        """
        Initialize the strategy.
        
        Args:
            fast_period (int): Period for fast moving average
            slow_period (int): Period for slow moving average
        """
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.momentum_period = 5
        
    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate the input data format.
        
        Args:
            data (pd.DataFrame): Input data to validate
            
        Raises:
            ValueError: If data format is invalid
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check if all required columns are present
        if not all(col in data.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in data.columns]
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check if there are enough data points for the slow moving average
        if len(data) < self.slow_period:
            logger.error(f"Not enough data points. Need at least {self.slow_period}, got {len(data)}")
            raise ValueError(f"Not enough data points. Need at least {self.slow_period}, got {len(data)}")
            
        # Check for NaN values
        if data[required_columns].isnull().any().any():
            logger.error("Data contains NaN values")
            raise ValueError("Data contains NaN values")
            
        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.error("Data index must be datetime")
            raise ValueError("Data index must be datetime")
    
    def calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate RSI indicator.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.Series: RSI values
        """
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_momentum(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate price momentum.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.Series: Momentum values
        """
        return data['close'].pct_change(periods=self.momentum_period)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on moving average crossovers and RSI.
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            pd.DataFrame: DataFrame with signals
        """
        try:
            # Validate data first
            self.validate_data(data)
            
            # Create a copy of the data to avoid modifying the original
            data = data.copy()
            
            # Calculate indicators
            data['fast_ma'] = data['close'].rolling(window=self.fast_period, min_periods=1).mean()
            data['slow_ma'] = data['close'].rolling(window=self.slow_period, min_periods=1).mean()
            data['rsi'] = self.calculate_rsi(data)
            data['momentum'] = self.calculate_momentum(data)
            
            # Initialize signals DataFrame with proper index
            signals = pd.DataFrame(index=data.index)
            signals['position'] = 0
            
            # Generate signals using vectorized operations
            # Buy signals: Fast MA crosses above Slow MA OR RSI is oversold
            fast_ma_crosses_up = (data['fast_ma'] > data['slow_ma']) & (data['fast_ma'].shift(1) <= data['slow_ma'].shift(1))
            rsi_oversold = data['rsi'] < self.rsi_oversold
            price_momentum = data['momentum'] > 0.01  # 1% momentum threshold
            
            # Sell signals: Fast MA crosses below Slow MA OR RSI is overbought
            fast_ma_crosses_down = (data['fast_ma'] < data['slow_ma']) & (data['fast_ma'].shift(1) >= data['slow_ma'].shift(1))
            rsi_overbought = data['rsi'] > self.rsi_overbought
            price_momentum_negative = data['momentum'] < -0.01  # -1% momentum threshold
            
            # Set positions using boolean indexing
            # Buy if any of these conditions are met
            signals.loc[fast_ma_crosses_up | rsi_oversold | price_momentum, 'position'] = 1
            # Sell if any of these conditions are met
            signals.loc[fast_ma_crosses_down | rsi_overbought | price_momentum_negative, 'position'] = -1
            
            # Fill NaN values with 0 (no position)
            signals['position'] = signals['position'].fillna(0)
            
            # Add trade information
            signals['entry_price'] = data['close']
            signals['exit_price'] = data['close'].shift(-1)
            
            # Store indicators in signals DataFrame
            signals['fast_ma'] = data['fast_ma']
            signals['slow_ma'] = data['slow_ma']
            signals['rsi'] = data['rsi']
            signals['momentum'] = data['momentum']
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise
    
    def optimize_parameters(self, data: pd.DataFrame, 
                          fast_range: range = range(5, 51, 5),
                          slow_range: range = range(10, 101, 10)) -> dict:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            data (pd.DataFrame): Historical price data
            fast_range (range): Range of fast MA periods to test
            slow_range (range): Range of slow MA periods to test
            
        Returns:
            dict: Best parameters and their performance
        """
        best_sharpe = -np.inf
        best_params = None
        
        for fast in fast_range:
            for slow in slow_range:
                if fast >= slow:
                    continue
                    
                try:
                    # Create strategy with current parameters
                    strategy = MovingAverageCrossoverStrategy(fast, slow)
                    
                    # Run backtest
                    from ..backtester import Backtester
                    backtester = Backtester(strategy)
                    results = backtester.run(data)
                    
                    # Skip if we got invalid results
                    if not results or np.isnan(results['sharpe_ratio']):
                        continue
                    
                    # Update best parameters if current is better
                    if results['sharpe_ratio'] > best_sharpe:
                        best_sharpe = results['sharpe_ratio']
                        best_params = {
                            'fast_period': fast,
                            'slow_period': slow,
                            'sharpe_ratio': best_sharpe,
                            'total_return': results['total_return'],
                            'max_drawdown': results['max_drawdown']
                        }
                        logger.info(f"New best parameters found: fast={fast}, slow={slow}, sharpe={best_sharpe:.2f}")
                except Exception as e:
                    logger.error(f"Error with parameters fast={fast}, slow={slow}: {str(e)}")
                    continue
        
        if best_params is None:
            # If no valid parameters found, use defaults
            best_params = {
                'fast_period': 20,
                'slow_period': 50,
                'sharpe_ratio': 0,
                'total_return': 0,
                'max_drawdown': 0
            }
            logger.warning("No valid parameters found, using defaults")
        
        return best_params 