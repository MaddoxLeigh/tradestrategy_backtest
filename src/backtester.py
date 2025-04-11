import pandas as pd
import numpy as np
from typing import Dict, Any
from src.strategies.base_strategy import BaseStrategy
import logging
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
            current_position_size = self.initial_capital * 0.01  # 1% position size
            
            # Calculate returns and track trades
            portfolio['returns'] = 0.0
            portfolio['portfolio_value'] = self.initial_capital
            
            for i in range(1, len(portfolio)):
                current_date = portfolio.index[i]
                prev_date = portfolio.index[i-1]
                
                # Check for position changes
                if portfolio.loc[current_date, 'position'] != portfolio.loc[prev_date, 'position']:
                    # Close previous trade if exists
                    if current_trade is not None:
                        current_trade['exit_date'] = current_date
                        current_trade['exit_price'] = portfolio.loc[current_date, 'close']
                        trade_return = (current_trade['exit_price'] / current_trade['entry_price'] - 1) * current_trade['position']
                        current_trade['return'] = trade_return
                        current_trade['position_size'] = current_position_size
                        trades.append(current_trade)
                        
                        # Update portfolio value with trade return
                        portfolio.loc[current_date, 'portfolio_value'] = (
                            portfolio.loc[prev_date, 'portfolio_value'] * (1 + trade_return * 0.01)
                        )
                    
                    # Start new trade
                    if portfolio.loc[current_date, 'position'] != 0:
                        current_trade = {
                            'entry_date': current_date,
                            'entry_price': portfolio.loc[current_date, 'close'],
                            'position': portfolio.loc[current_date, 'position']
                        }
                    else:
                        current_trade = None
                        portfolio.loc[current_date, 'portfolio_value'] = portfolio.loc[prev_date, 'portfolio_value']
                
                # Calculate daily returns
                if current_trade is not None:
                    daily_return = (
                        portfolio.loc[current_date, 'close'] / portfolio.loc[prev_date, 'close'] - 1
                    ) * current_trade['position']
                    portfolio.loc[current_date, 'returns'] = daily_return
                    portfolio.loc[current_date, 'portfolio_value'] = (
                        portfolio.loc[prev_date, 'portfolio_value'] * (1 + daily_return * 0.01)
                    )
                else:
                    portfolio.loc[current_date, 'portfolio_value'] = portfolio.loc[prev_date, 'portfolio_value']
            
            # Close any open trade at the end
            if current_trade is not None:
                current_trade['exit_date'] = portfolio.index[-1]
                current_trade['exit_price'] = portfolio.loc[portfolio.index[-1], 'close']
                trade_return = (current_trade['exit_price'] / current_trade['entry_price'] - 1) * current_trade['position']
                current_trade['return'] = trade_return
                current_trade['position_size'] = current_position_size
                trades.append(current_trade)
            
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(trades)
            
            # Calculate performance metrics
            total_return = (portfolio['portfolio_value'].iloc[-1] / self.initial_capital) - 1
            
            # Calculate Sharpe ratio (handle division by zero)
            daily_returns = portfolio['returns']
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
        Plot backtest results using Plotly.
        
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
            fig = make_subplots(rows=4, cols=1, 
                              shared_xaxes=True,
                              vertical_spacing=0.05,
                              subplot_titles=('Price and Moving Averages', 
                                            'RSI Indicator',
                                            'Equity Curve',
                                            'Drawdown'),
                              row_heights=[0.4, 0.2, 0.2, 0.2])
            
            # Plot 1: Price and Moving Averages
            fig.add_trace(go.Scatter(x=results['portfolio'].index, 
                                   y=results['portfolio']['close'],
                                   name='Price',
                                   line=dict(color='black')),
                        row=1, col=1)
            
            fig.add_trace(go.Scatter(x=results['portfolio'].index,
                                   y=results['portfolio']['fast_ma'],
                                   name=f'Fast MA ({self.strategy.fast_period})',
                                   line=dict(color='blue')),
                        row=1, col=1)
            
            fig.add_trace(go.Scatter(x=results['portfolio'].index,
                                   y=results['portfolio']['slow_ma'],
                                   name=f'Slow MA ({self.strategy.slow_period})',
                                   line=dict(color='red')),
                        row=1, col=1)
            
            # Plot trades
            trades_df = results['trades'].copy()
            if not trades_df.empty:
                # Calculate position size
                position_size = self.initial_capital * 0.01
                
                # Add buy signals
                buy_trades = trades_df[trades_df['position'] > 0]
                fig.add_trace(go.Scatter(
                    x=buy_trades['entry_date'],
                    y=buy_trades['entry_price'],
                    mode='markers',
                    name='Buy',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color='green'
                    ),
                    hovertemplate=(
                        "Date: %{x}<br>" +
                        "Price: $%{y:.2f}<br>" +
                        "Position: LONG<br>" +
                        f"Position Size: ${position_size:,.2f}<br>" +
                        "Return: %{customdata:.2f}%<extra></extra>"
                    ),
                    customdata=buy_trades['return'] * 100
                ), row=1, col=1)
                
                # Add sell signals
                sell_trades = trades_df[trades_df['position'] < 0]
                fig.add_trace(go.Scatter(
                    x=sell_trades['entry_date'],
                    y=sell_trades['entry_price'],
                    mode='markers',
                    name='Sell',
                    marker=dict(
                        symbol='triangle-down',
                        size=10,
                        color='red'
                    ),
                    hovertemplate=(
                        "Date: %{x}<br>" +
                        "Price: $%{y:.2f}<br>" +
                        "Position: SHORT<br>" +
                        f"Position Size: ${position_size:,.2f}<br>" +
                        "Return: %{customdata:.2f}%<extra></extra>"
                    ),
                    customdata=sell_trades['return'] * 100
                ), row=1, col=1)
            
            # Plot 2: RSI
            fig.add_trace(go.Scatter(x=results['portfolio'].index,
                                   y=results['portfolio']['rsi'],
                                   name='RSI',
                                   line=dict(color='purple')),
                        row=2, col=1)
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # Plot 3: Equity Curve
            fig.add_trace(go.Scatter(x=results['portfolio'].index,
                                   y=results['portfolio']['portfolio_value'],
                                   name='Equity',
                                   line=dict(color='blue')),
                        row=3, col=1)
            
            # Plot 4: Drawdown
            fig.add_trace(go.Scatter(x=results['portfolio'].index,
                                   y=results['portfolio']['drawdown'] * 100,
                                   name='Drawdown',
                                   line=dict(color='red')),
                        row=4, col=1)
            
            # Update layout
            fig.update_layout(
                height=1000,
                title_text="Backtest Results",
                showlegend=True,
                hovermode='x unified'
            )
            
            # Update y-axes labels
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            fig.update_yaxes(title_text="Portfolio Value ($)", row=3, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=4, col=1)
            
            # Show the plot
            fig.show()
            
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

    def scan_tickers(self, tickers: list, data_provider, lookback_days: int = 5) -> pd.DataFrame:
        """
        Scan multiple tickers for potential trading opportunities.
        
        Args:
            tickers (list): List of stock tickers to scan
            data_provider: Data provider object that can fetch historical data
            lookback_days (int): Number of days to look back for analysis
            
        Returns:
            pd.DataFrame: DataFrame containing potential opportunities with metrics
        """
        opportunities = []
        
        for ticker in tickers:
            try:
                # Fetch historical data
                data = data_provider.get_historical_data(ticker, lookback_days)
                
                # Generate signals for this ticker
                signals = self.strategy.generate_signals(data)
                
                # Get the latest signal
                latest_signal = signals['position'].iloc[-1]
                
                if latest_signal != 0:  # If there's a trading signal
                    # Calculate additional metrics
                    current_price = data['close'].iloc[-1]
                    rsi = data['rsi'].iloc[-1] if 'rsi' in data.columns else None
                    momentum = data['momentum'].iloc[-1] if 'momentum' in data.columns else None
                    
                    opportunity = {
                        'ticker': ticker,
                        'signal': 'LONG' if latest_signal > 0 else 'SHORT',
                        'current_price': current_price,
                        'rsi': rsi,
                        'momentum': momentum,
                        'timestamp': data.index[-1]
                    }
                    opportunities.append(opportunity)
                    
            except Exception as e:
                logger.warning(f"Error scanning {ticker}: {str(e)}")
                continue
        
        if opportunities:
            return pd.DataFrame(opportunities)
        return pd.DataFrame(columns=['ticker', 'signal', 'current_price', 'rsi', 'momentum', 'timestamp']) 