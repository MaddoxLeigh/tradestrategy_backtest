import numpy as np
import pandas as pd
from typing import Optional, Tuple

class TechnicalIndicators:
    """
    Class containing various technical indicators used in trading strategies.
    """
    
    def __init__(self):
        pass
    
    def sma(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            data (pd.Series): Price data
            period (int): Moving average period
            
        Returns:
            pd.Series: SMA values
        """
        return data.rolling(window=period, min_periods=1).mean()
    
    def ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            data (pd.Series): Price data
            period (int): Moving average period
            
        Returns:
            pd.Series: EMA values
        """
        return data.ewm(span=period, min_periods=1).mean()
    
    def rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            data (pd.Series): Price data
            period (int): RSI period
            
        Returns:
            pd.Series: RSI values
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def macd(self, data: pd.Series, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data (pd.Series): Price data
            fastperiod (int): Fast EMA period
            slowperiod (int): Slow EMA period
            signalperiod (int): Signal line period
            
        Returns:
            tuple: (macd, signal, hist)
        """
        exp1 = data.ewm(span=fastperiod, min_periods=1).mean()
        exp2 = data.ewm(span=slowperiod, min_periods=1).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signalperiod, min_periods=1).mean()
        hist = macd - signal
        return macd, signal, hist
    
    def bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data (pd.Series): Price data
            period (int): Moving average period
            std_dev (float): Standard deviation multiplier
            
        Returns:
            tuple: (upper_band, middle_band, lower_band)
        """
        middle_band = data.rolling(window=period, min_periods=1).mean()
        std = data.rolling(window=period, min_periods=1).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        return upper_band, middle_band, lower_band
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            period (int): ATR period
            
        Returns:
            pd.Series: ATR values
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean()
    
    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                  k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            k_period (int): %K period
            d_period (int): %D period
            
        Returns:
            tuple: (slowk, slowd)
        """
        lowest_low = low.rolling(window=k_period, min_periods=1).min()
        highest_high = high.rolling(window=k_period, min_periods=1).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period, min_periods=1).mean()
        return k, d 