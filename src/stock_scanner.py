import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List
from trading_recommender import TradingRecommender
import os

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
                ## Add csv with desired stock tickers 
                #sp500_df = pd.read_csv('sp500.csv')
                etoro_df = pd.read_csv('etoro-stocks.csv')
                self.stocks = etoro_df['Ticker'].tolist()
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stock_recommendations_{timestamp}.csv"
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filepath = os.path.join(output_dir, filename)
        
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
                
                # Create recommendation dictionary
                recommendation = {
                    'ticker': ticker,
                    'price': current_price,
                    'signal': result['recommendation'],
                    'confidence': result['confidence_score'],
                    'tp': tp,
                    'sl': sl,
                    'timestamp': datetime.now()
                }
                
                # Add to recommendations list
                recommendations.append(recommendation)
                
                # Sort recommendations by confidence
                recommendations.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Write current state to CSV
                df = pd.DataFrame(recommendations)
                df.to_csv(filepath, index=False)
                logger.info(f"Updated CSV with {ticker} recommendation")
                
            except Exception as e:
                logger.error(f"Error scanning {ticker}: {str(e)}")
                continue
        
        return recommendations

    def write_recommendations_to_csv(self, recommendations: List[Dict[str, Any]], filename: str = None) -> str:
        """
        Write recommendations to a CSV file.
        
        Args:
            recommendations (List[Dict[str, Any]]): List of recommendations to write
            filename (str, optional): Name of the CSV file. If None, generates a timestamp-based name.
        
        Returns:
            str: Path to the created CSV file
        """
        try:
            logger.info(f"Starting to write {len(recommendations)} recommendations to CSV")
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"stock_recommendations_{timestamp}.csv"
                logger.info(f"Generated filename: {filename}")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
            logger.info(f"Checking/creating output directory: {output_dir}")
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"Created output directory: {output_dir}")
            
            filepath = os.path.join(output_dir, filename)
            logger.info(f"Full filepath: {filepath}")
            
            # Convert recommendations to DataFrame
            logger.info("Converting recommendations to DataFrame")
            df = pd.DataFrame(recommendations)
            
            # Add timestamp column
            df['timestamp'] = datetime.now()
            
            # Write to CSV
            logger.info(f"Writing to CSV file: {filepath}")
            df.to_csv(filepath, index=False)
            logger.info(f"Successfully wrote recommendations to {filepath}")
            
            return filepath
        except Exception as e:
            logger.error(f"Error writing to CSV: {str(e)}")
            raise

def main():
    """
    Main function to run the stock scanner.
    """
    try:
        # Initialize scanner
        logger.info("Initializing stock scanner")
        scanner = StockScanner()
        
        # Scan stocks
        logger.info("Starting stock scan")
        print("\nScanning S&P 500 stocks...")
        recommendations = scanner.scan_stocks()
        logger.info(f"Found {len(recommendations)} recommendations")
        
        # Print results
        print("\nTrading Recommendations (Sorted by Confidence)")
        print("=" * 100)
        print(f"{'Ticker':<8} {'Price':>10} {'Signal':<8} {'Confidence':>10} {'Take Profit':>12} {'Stop Loss':>12}")
        print("-" * 100)
        
        for rec in recommendations:
            print(f"{rec['ticker']:<8} ${rec['price']:>9,.2f} {rec['signal']:<8} {rec['confidence']:>9.1f}% "
                  f"${rec['tp']:>11,.2f} ${rec['sl']:>11,.2f}")
        
        print("=" * 100)
        print(f"\nRecommendations have been saved to: output/stock_recommendations_*.csv")
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main() 