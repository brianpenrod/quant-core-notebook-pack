import numpy as np

def sharpe_ratio(returns, risk_free=0.0):
    """
    Calculates the annualized Sharpe Ratio.
    Assumes daily returns.
    """
    excess_returns = returns - risk_free
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)
    
    if std_excess == 0:
        return 0.0
        
    # Annualize (Sqrt(252) for trading days)
    return (mean_excess / std_excess) * np.sqrt(252)

def max_drawdown(prices):
    """
    Calculates Maximum Drawdown (Peak to Trough decline).
    """
    # Create a cumulative series of max prices
    rolling_max = np.maximum.accumulate(prices)
    drawdowns = (prices - rolling_max) / rolling_max
    return np.min(drawdowns)

def information_ratio(returns, benchmark_returns):
    """
    Calculates Information Ratio (Active Return / Tracking Error).
    """
    active_return = returns - benchmark_returns
    tracking_error = np.std(active_return)
    
    if tracking_error == 0:
        return 0.0
        
    return np.mean(active_return) / tracking_error
