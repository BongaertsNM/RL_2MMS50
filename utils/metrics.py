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


def compute_confidence_interval(data, axis=0, confidence=0.95):
    """
    Compute mean and confidence interval for provided data.

    Uses normal approximation: mean +/- z * sem.

    Args:
        data (np.ndarray): Data values.
        axis (int): Axis along which to compute statistics.
        confidence (float): Confidence level; default 0.95.

    Returns:
        mean (np.ndarray): Mean of data.
        lower (np.ndarray): Lower bound of confidence interval.
        upper (np.ndarray): Upper bound of confidence interval.
    """
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis, ddof=1)
    n = data.shape[axis]
    sem = std / np.sqrt(n)
    # Z-score for normal distribution
    z = abs(np.percentile(np.random.randn(1000000), [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100]))
    # But simpler: for 95% confidence, z ~1.96; otherwise approximate with scipy? Use normal quantile approximation
    z_val = 1.96 if confidence == 0.95 else abs(np.sqrt(2) * np.erfinv(confidence))
    lower = mean - z_val * sem
    upper = mean + z_val * sem
    return mean, lower, upper
