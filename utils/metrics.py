import numpy as np


def compute_win_rate(returns, axis=0):
    """
    Compute the win rate (fraction of +1 rewards) across runs or episodes.

    Args:
        returns (np.ndarray): Array of returns. Can be 1D or 2D.
        axis (int): Axis along which to compute the fraction of wins.

    Returns:
        win_rate (np.ndarray or float): Fraction of returns equal to +1.
    """
    wins = (returns == 1).astype(float)
    return np.mean(wins, axis=axis)