import pytest
import pandas as pd
import numpy as np
from src.strategies.moving_average_crossover import MovingAverageCrossoverStrategy

@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)
    prices = 100 + np.random.randn(len(dates)).cumsum()
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.random.rand(len(dates)),
        'low': prices - np.random.rand(len(dates)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    return data

def test_strategy_initialization():
    """Test strategy initialization with default parameters."""
    strategy = MovingAverageCrossoverStrategy()
    assert strategy.fast_period == 20
    assert strategy.slow_period == 50

def test_strategy_initialization_custom_params():
    """Test strategy initialization with custom parameters."""
    strategy = MovingAverageCrossoverStrategy(fast_period=10, slow_period=30)
    assert strategy.fast_period == 10
    assert strategy.slow_period == 30

def test_generate_signals(sample_data):
    """Test signal generation."""
    strategy = MovingAverageCrossoverStrategy()
    signals = strategy.generate_signals(sample_data)
    
    # Check that signals DataFrame has required columns
    assert all(col in signals.columns for col in ['price', 'fast_ma', 'slow_ma', 'position'])
    
    # Check that position values are valid
    assert all(pos in [-1, 0, 1] for pos in signals['position'].unique())
    
    # Check that signals are generated for all data points
    assert len(signals) == len(sample_data)

def test_optimize_parameters(sample_data):
    """Test parameter optimization."""
    strategy = MovingAverageCrossoverStrategy()
    best_params = strategy.optimize_parameters(sample_data)
    
    # Check that best parameters are returned
    assert 'fast_period' in best_params
    assert 'slow_period' in best_params
    assert 'sharpe_ratio' in best_params
    assert 'total_return' in best_params
    assert 'max_drawdown' in best_params
    
    # Check that fast period is less than slow period
    assert best_params['fast_period'] < best_params['slow_period']

def test_invalid_data():
    """Test handling of invalid data."""
    strategy = MovingAverageCrossoverStrategy()
    invalid_data = pd.DataFrame({'invalid_column': [1, 2, 3]})
    
    with pytest.raises(ValueError):
        strategy.generate_signals(invalid_data)

def test_edge_cases(sample_data):
    """Test edge cases in signal generation."""
    strategy = MovingAverageCrossoverStrategy()
    
    # Test with empty DataFrame
    empty_data = pd.DataFrame()
    with pytest.raises(ValueError):
        strategy.generate_signals(empty_data)
    
    # Test with single row
    single_row = sample_data.iloc[:1]
    signals = strategy.generate_signals(single_row)
    assert len(signals) == 1
    assert signals['position'].iloc[0] == 0  # Should be no position for single data point 