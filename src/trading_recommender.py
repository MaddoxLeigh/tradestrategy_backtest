import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingRecommender:
    """
    Class for generating real-time trading recommendations based on the same strategy as the backtester.
    """
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50, rsi_period: int = 14,
                 rsi_overbought: int = 70, rsi_oversold: int = 30):
        """
        Initialize the trading recommender with strategy parameters.
        
        Args:
            fast_period (int): Fast moving average period
            slow_period (int): Slow moving average period
            rsi_period (int): RSI calculation period
            rsi_overbought (int): RSI overbought threshold
            rsi_oversold (int): RSI oversold threshold
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        
    def calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate RSI indicator.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.Series: RSI values
        """
        # Calculate price changes
        delta = data['Close'].diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        
        # Set gains and losses
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_recommendation(self, ticker: str, lookback_days: int = 100) -> Dict[str, Any]:
        """
        Get trading recommendation for a given ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            lookback_days (int): Number of days of historical data to use
            
        Returns:
            Dict[str, Any]: Trading recommendation and indicators
        """
        try:
            # Fetch historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Calculate indicators
            data['fast_ma'] = data['Close'].rolling(window=self.fast_period, min_periods=1).mean()
            data['slow_ma'] = data['Close'].rolling(window=self.slow_period, min_periods=1).mean()
            data['rsi'] = self.calculate_rsi(data)
            
            # Get latest values
            latest = data.iloc[-1]
            prev = data.iloc[-2]
            
            # Generate recommendation based on strategy
            recommendation = "HOLD"
            reason = []
            confidence_score = 0.0
            
            # Calculate MA crossover strength
            latest_fast_ma = float(latest['fast_ma'].iloc[0])
            latest_slow_ma = float(latest['slow_ma'].iloc[0])
            prev_fast_ma = float(prev['fast_ma'].iloc[0])
            prev_slow_ma = float(prev['slow_ma'].iloc[0])
            
            ma_distance = (latest_fast_ma - latest_slow_ma) / latest_slow_ma
            ma_crossover_strength = abs(ma_distance) * 100  # Convert to percentage
            
            # Calculate RSI strength
            current_rsi = float(latest['rsi'].iloc[0])
            rsi_strength = 0.0
            if current_rsi > self.rsi_overbought:
                rsi_strength = (current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            elif current_rsi < self.rsi_oversold:
                rsi_strength = (self.rsi_oversold - current_rsi) / self.rsi_oversold
            
            # Calculate price momentum
            latest_close = float(latest['Close'].iloc[0])
            prev_close = float(prev['Close'].iloc[0])
            price_change = (latest_close - prev_close) / prev_close
            momentum_strength = abs(price_change) * 100  # Convert to percentage
            
            # Check moving average crossover
            if latest_fast_ma > latest_slow_ma and prev_fast_ma <= prev_slow_ma:
                recommendation = "BUY"
                reason.append("Fast MA crossed above Slow MA")
                confidence_score += ma_crossover_strength
            elif latest_fast_ma < latest_slow_ma and prev_fast_ma >= prev_slow_ma:
                recommendation = "SELL"
                reason.append("Fast MA crossed below Slow MA")
                confidence_score += ma_crossover_strength
            
            # Check RSI
            if current_rsi > self.rsi_overbought:
                if recommendation == "BUY":
                    recommendation = "HOLD"
                    reason.append("RSI indicates overbought conditions")
                    confidence_score -= rsi_strength * 50  # Reduce confidence due to overbought
                elif recommendation == "HOLD":
                    recommendation = "SELL"
                    reason.append("RSI indicates overbought conditions")
                    confidence_score += rsi_strength * 50
            elif current_rsi < self.rsi_oversold:
                if recommendation == "SELL":
                    recommendation = "HOLD"
                    reason.append("RSI indicates oversold conditions")
                    confidence_score -= rsi_strength * 50  # Reduce confidence due to oversold
                elif recommendation == "HOLD":
                    recommendation = "BUY"
                    reason.append("RSI indicates oversold conditions")
                    confidence_score += rsi_strength * 50
            
            # Add momentum to confidence score
            if recommendation == "BUY" and price_change > 0:
                confidence_score += momentum_strength
            elif recommendation == "SELL" and price_change < 0:
                confidence_score += momentum_strength
            
            # Normalize confidence score to 0-100 range
            confidence_score = max(0, min(100, confidence_score))
            
            # Calculate ATR for TP and SL
            data = yf.download(ticker, period="20d")
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            # Calculate TP and SL based on ATR
            if recommendation == "BUY":
                # For BUY: TP is 2 ATR above, SL is 1 ATR below
                tp = latest_close + (atr * 2)
                sl = latest_close - atr
            elif recommendation == "SELL":
                # For SELL: TP is 2 ATR below, SL is 1 ATR above
                tp = latest_close - (atr * 2)
                sl = latest_close + atr
            else:
                tp = sl = latest_close
            
            return {
                'ticker': ticker,
                'recommendation': recommendation,
                'confidence_score': confidence_score,
                'reasons': reason,
                'current_price': latest_close,
                'indicators': {
                    'fast_ma': latest_fast_ma,
                    'slow_ma': latest_slow_ma,
                    'rsi': current_rsi,
                    'ma_distance': ma_distance * 100,  # Convert to percentage
                    'price_change': price_change * 100  # Convert to percentage
                },
                'tp': tp,
                'sl': sl
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendation for {ticker}: {str(e)}")
            raise

def main():
    """
    Main function to demonstrate usage of the TradingRecommender.
    """
    # Initialize recommender with default parameters
    recommender = TradingRecommender()
    
    # Get user input
    ticker = input("Enter stock ticker (e.g., AAPL): ").upper()
    
    try:
        # Get recommendation
        result = recommender.get_recommendation(ticker)
        
        # Print single row output with better formatting
        print(f"\nTrading Recommendation for {result['ticker']}")
        print("=" * 50)
        print(f"Current Price: ${result['current_price']:,.2f}")
        print(f"Signal:       {result['recommendation']}")
        print(f"Confidence:   {result['confidence_score']:.1f}%")
        print(f"Take Profit:  ${result['tp']:,.2f}")
        print(f"Stop Loss:    ${result['sl']:,.2f}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 