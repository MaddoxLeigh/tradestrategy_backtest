import pandas as pd
import numpy as np
from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Backtester:
    """
    Class for backtesting trading strategies.
    """
    
    def __init__(self, strategy: BaseStrategy, initial_capital: float = 100000.0):
        """
        Initialize the backtester.
        
        Args:
            strategy (BaseStrategy): Trading strategy to backtest
            initial_capital (float): Initial capital for backtesting
        """
        logger.info(f"Initializing backtester with initial capital: ${initial_capital:,.2f}")
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.positions = pd.DataFrame()
        self.trades = []
    
    def run(self, data: pd.DataFrame) -> dict:
        """
        Run the backtest.
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            dict: Backtest results
        """
        try:
            # Generate signals
            signals = self.strategy.generate_signals(data)
            
            # Initialize portfolio with all necessary columns
            portfolio = pd.DataFrame(index=data.index)
            portfolio['position'] = signals['position']
            portfolio['close'] = data['close']
            
            # Calculate indicators
            portfolio['fast_ma'] = portfolio['close'].rolling(window=self.strategy.fast_period, min_periods=1).mean()
            portfolio['slow_ma'] = portfolio['close'].rolling(window=self.strategy.slow_period, min_periods=1).mean()
            portfolio['rsi'] = self.strategy.calculate_rsi(portfolio)
            
            # Track trades
            trades = []
            current_trade = None
            
            # Calculate returns and track trades
            portfolio['returns'] = 0.0
            for i in range(1, len(portfolio)):
                current_date = portfolio.index[i]
                prev_date = portfolio.index[i-1]
                
                # Check for position changes
                if portfolio.loc[current_date, 'position'] != portfolio.loc[prev_date, 'position']:
                    # Close previous trade if exists
                    if current_trade is not None:
                        current_trade['exit_date'] = current_date
                        current_trade['exit_price'] = portfolio.loc[current_date, 'close']
                        current_trade['return'] = (current_trade['exit_price'] / current_trade['entry_price'] - 1) * current_trade['position']
                        trades.append(current_trade)
                    
                    # Start new trade
                    if portfolio.loc[current_date, 'position'] != 0:
                        current_trade = {
                            'entry_date': current_date,
                            'entry_price': portfolio.loc[current_date, 'close'],
                            'position': portfolio.loc[current_date, 'position']
                        }
                    else:
                        current_trade = None
                
                # Calculate daily returns
                if current_trade is not None:
                    portfolio.loc[current_date, 'returns'] = (
                        portfolio.loc[current_date, 'close'] / portfolio.loc[prev_date, 'close'] - 1
                    ) * current_trade['position']
            
            # Close any open trade at the end
            if current_trade is not None:
                current_trade['exit_date'] = portfolio.index[-1]
                current_trade['exit_price'] = portfolio.loc[portfolio.index[-1], 'close']
                current_trade['return'] = (current_trade['exit_price'] / current_trade['entry_price'] - 1) * current_trade['position']
                trades.append(current_trade)
            
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(trades)
            
            # Calculate portfolio value
            portfolio['portfolio_value'] = self.initial_capital * (1 + portfolio['returns']).cumprod()
            
            # Calculate performance metrics
            daily_returns = portfolio['returns']
            total_return = (portfolio['portfolio_value'].iloc[-1] / self.initial_capital) - 1
            
            # Calculate Sharpe ratio (handle division by zero)
            if daily_returns.std() != 0:
                sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            else:
                sharpe_ratio = 0.0
            
            # Calculate max drawdown
            portfolio['peak'] = portfolio['portfolio_value'].cummax()
            portfolio['drawdown'] = (portfolio['portfolio_value'] - portfolio['peak']) / portfolio['peak']
            max_drawdown = portfolio['drawdown'].min()
            
            # Calculate trade statistics
            if not trades_df.empty:
                winning_trades = trades_df[trades_df['return'] > 0]
                losing_trades = trades_df[trades_df['return'] < 0]
                win_rate = len(winning_trades) / len(trades_df)
                avg_trade_return = trades_df['return'].mean()
                
                # Calculate profit factor
                winning_trades_sum = winning_trades['return'].sum()
                losing_trades_sum = abs(losing_trades['return'].sum())
                profit_factor = winning_trades_sum / losing_trades_sum if losing_trades_sum != 0 else float('inf')
            else:
                win_rate = 0.0
                avg_trade_return = 0.0
                profit_factor = 0.0
            
            # Store results
            self.results = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_trade_return': avg_trade_return,
                'profit_factor': profit_factor,
                'portfolio': portfolio,
                'trades': trades_df
            }
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise
    
    def print_trades(self, results: dict = None) -> None:
        """
        Print detailed trade information to terminal.
        
        Args:
            results (dict, optional): Backtest results. If None, uses stored results.
        """
        if results is None:
            if not hasattr(self, 'results'):
                raise ValueError("No results to print. Run the backtest first.")
            results = self.results
            
        trades_df = results['trades'].copy()
        if not trades_df.empty:
            print("\nTrade Details:")
            print("-" * 80)
            print(f"{'Date':<12} {'Type':<8} {'Entry':<10} {'Exit':<10} {'Return':<10} {'Result':<8}")
            print("-" * 80)
            
            for _, trade in trades_df.iterrows():
                trade_type = "Buy" if trade['position'] > 0 else "Sell"
                result = "Win" if trade['return'] > 0 else "Loss"
                return_str = f"{trade['return']:+.2%}"
                print(f"{trade['entry_date'].strftime('%Y-%m-%d'):<12} {trade_type:<8} "
                      f"${trade['entry_price']:.2f} ${trade['exit_price']:.2f} "
                      f"{return_str:<10} {result:<8}")
            
            # Print summary statistics
            print("-" * 80)
            print(f"Total Trades: {len(trades_df)}")
            print(f"Winning Trades: {len(trades_df[trades_df['return'] > 0])} "
                  f"({results['win_rate']:.1%})")
            print(f"Average Return: {results['avg_trade_return']:.2%}")
            print(f"Profit Factor: {results['profit_factor']:.2f}")
            print("-" * 80)
    
    def plot_results(self, results: dict = None) -> None:
        """
        Plot backtest results.
        
        Args:
            results (dict, optional): Backtest results. If None, uses stored results.
        """
        try:
            # Use stored results if none provided
            if results is None:
                if not hasattr(self, 'results'):
                    raise ValueError("No results to plot. Run the backtest first.")
                results = self.results
            
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 12))
            gs = plt.GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
            
            # Plot 1: Price and Moving Averages
            ax1 = plt.subplot(gs[0])
            ax1.plot(results['portfolio'].index, results['portfolio']['close'], 'k-', label='Price', alpha=0.7)
            ax1.plot(results['portfolio'].index, results['portfolio']['fast_ma'], 'b-', label=f'Fast MA ({self.strategy.fast_period})', alpha=0.7)
            ax1.plot(results['portfolio'].index, results['portfolio']['slow_ma'], 'r-', label=f'Slow MA ({self.strategy.slow_period})', alpha=0.7)
            
            # Plot buy/sell signals with simple annotations
            trades_df = results['trades'].copy()
            if not trades_df.empty:
                # Plot trades
                for _, trade in trades_df.iterrows():
                    # Determine if it's a winning or losing trade
                    is_win = trade['return'] > 0
                    color = 'green' if is_win else 'red'
                    marker = '^' if trade['position'] > 0 else 'v'
                    
                    # Plot the trade point
                    ax1.scatter(trade['entry_date'], trade['entry_price'], color=color, marker=marker, s=100)
                    
                    # Add W/L annotation
                    result = 'W' if is_win else 'L'
                    ax1.annotate(result, 
                               (trade['entry_date'], trade['entry_price']),
                               xytext=(0, 10), textcoords='offset points',
                               ha='center', va='bottom', color=color,
                               fontsize=10, weight='bold')
            
            ax1.set_title('Price and Moving Averages with Trade Signals', fontsize=12, pad=20)
            ax1.set_ylabel('Price ($)', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left', fontsize=8)
            
            # Plot 2: RSI
            ax2 = plt.subplot(gs[1])
            ax2.plot(results['portfolio'].index, results['portfolio']['rsi'], color='purple', linestyle='-', label='RSI', alpha=0.7)
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.3)
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.3)
            ax2.set_title('RSI Indicator', fontsize=12, pad=20)
            ax2.set_ylabel('RSI', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)
            
            # Plot 3: Equity Curve
            ax3 = plt.subplot(gs[2])
            ax3.plot(results['portfolio'].index, results['portfolio']['portfolio_value'], 'b-', label='Equity')
            ax3.set_title('Equity Curve', fontsize=12, pad=20)
            ax3.set_ylabel('Portfolio Value ($)', fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Returns
            ax4 = plt.subplot(gs[3])
            ax4.plot(results['portfolio'].index, results['portfolio']['returns'].cumsum(), 'g-', label='Cumulative Returns')
            ax4.set_title('Cumulative Returns', fontsize=12, pad=20)
            ax4.set_ylabel('Returns (%)', fontsize=10)
            ax4.grid(True, alpha=0.3)
            
            # Format x-axis dates for all subplots
            for ax in [ax1, ax2, ax3, ax4]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.show()
            
            # Print trade details to terminal
            self.print_trades(results)
            
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            raise
        
    def get_summary(self) -> str:
        """
        Get a summary of the backtest results.
        
        Returns:
            str: Summary of results
        """
        if self.positions is None:
            raise ValueError("No results to summarize. Run the backtest first.")
            
        summary = f"""
        Backtest Summary
        ---------------
        Initial Capital: ${self.initial_capital:,.2f}
        Final Portfolio Value: ${self.positions['portfolio_value'].iloc[-1]:,.2f}
        Total Return: {self.positions['returns'].iloc[-1]:.2%}
        Sharpe Ratio: {self.positions['sharpe_ratio']:.2f}
        Max Drawdown: {self.positions['max_drawdown']:.2%}
        Win Rate: {self.positions['win_rate']:.2%}
        Total Trades: {self.positions['total_trades']}
        Profit Factor: {self.positions['profit_factor']:.2f}
        """
        
        return summary 