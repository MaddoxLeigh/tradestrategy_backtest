import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List
from trading_recommender import TradingRecommender

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockScanner:
    """
    Class for scanning multiple stocks and generating trading recommendations.
    """
    
    def __init__(self, stocks: List[str] = None):
        """
        Initialize the stock scanner.
        
        Args:
            stocks (List[str], optional): List of stock tickers to scan. Defaults to S&P 500 stocks.
        """
        if stocks is None:
            # Read S&P 500 symbols from CSV
            try:
                sp500_df = pd.read_csv('sp500.csv')
                self.stocks = sp500_df['Symbol'].tolist()
                logger.info(f"Loaded {len(self.stocks)} stocks from S&P 500")
            except Exception as e:
                logger.error(f"Error loading S&P 500 symbols: {str(e)}")
                raise
        else:
            self.stocks = stocks
            
        self.recommender = TradingRecommender()
    
    def scan_stocks(self) -> List[Dict[str, Any]]:
        """
        Scan all stocks and generate recommendations.
        
        Returns:
            List[Dict[str, Any]]: List of recommendations sorted by confidence
        """
        recommendations = []
        
        for ticker in self.stocks:
            try:
                # Get recommendation for each stock
                result = self.recommender.get_recommendation(ticker)
                current_price = result['current_price']
                
                # Calculate ATR for TP and SL
                data = yf.download(ticker, period="20d")
                high_low = data['High'] - data['Low']
                high_close = np.abs(data['High'] - data['Close'].shift())
                low_close = np.abs(data['Low'] - data['Close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = np.max(ranges, axis=1)
                atr = true_range.rolling(14).mean().iloc[-1]
                
                # Calculate TP and SL based on ATR
                if result['recommendation'] == "BUY":
                    # For BUY: TP is 2 ATR above, SL is 1 ATR below
                    tp = current_price + (atr * 2)
                    sl = current_price - atr
                elif result['recommendation'] == "SELL":
                    # For SELL: TP is 2 ATR below, SL is 1 ATR above
                    tp = current_price - (atr * 2)
                    sl = current_price + atr
                else:
                    tp = sl = current_price
                
                # Add to recommendations
                recommendations.append({
                    'ticker': ticker,
                    'price': current_price,
                    'signal': result['recommendation'],
                    'confidence': result['confidence_score'],
                    'tp': tp,
                    'sl': sl
                })
                
            except Exception as e:
                logger.error(f"Error scanning {ticker}: {str(e)}")
                continue
        
        # Sort by confidence (highest first)
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        return recommendations

def main():
    """
    Main function to run the stock scanner.
    """
    # Initialize scanner
    scanner = StockScanner()
    
    # Scan stocks
    print("\nScanning S&P 500 stocks...")
    recommendations = scanner.scan_stocks()
    
    # Print results
    print("\nTrading Recommendations (Sorted by Confidence)")
    print("=" * 100)
    print(f"{'Ticker':<8} {'Price':>10} {'Signal':<8} {'Confidence':>10} {'Take Profit':>12} {'Stop Loss':>12}")
    print("-" * 100)
    
    for rec in recommendations:
        print(f"{rec['ticker']:<8} ${rec['price']:>9,.2f} {rec['signal']:<8} {rec['confidence']:>9.1f}% "
              f"${rec['tp']:>11,.2f} ${rec['sl']:>11,.2f}")
    
    print("=" * 100)

if __name__ == "__main__":
    main() 