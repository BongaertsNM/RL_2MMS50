import numpy as np
from utils.metrics import compute_win_rate, compute_confidence_interval


def summary_statistics(returns):
    """
    Compute mean return and 95% confidence interval over runs, for each episode.

    Args:
        returns (np.ndarray): shape (n_runs, n_episodes)

    Returns:
        mean (np.ndarray): mean return per episode.
        lower (np.ndarray): lower bound of 95% CI per episode.
        upper (np.ndarray): upper bound of 95% CI per episode.
    """
    mean, lower, upper = compute_confidence_interval(returns, axis=0)
    return mean, lower, upper


def summary_final_performance(returns):
    """
    Compute mean return and 95% CI at the final episode across runs.

    Args:
        returns (np.ndarray): shape (n_runs, n_episodes)

    Returns:
        mean (float): mean return at last episode.
        lower (float): lower bound of 95% CI at last episode.
        upper (float): upper bound of 95% CI at last episode.
    """
    last_returns = returns[:, -1]
    mean = np.mean(last_returns)
    std = np.std(last_returns, ddof=1)
    sem = std / np.sqrt(len(last_returns))
    z = 1.96  # for 95% CI
    lower = mean - z * sem
    upper = mean + z * sem
    return mean, lower, upper


def win_rate_over_time(returns):
    """
    Compute win rate (fraction of +1 returns) per episode across runs.

    Args:
        returns (np.ndarray): shape (n_runs, n_episodes)

    Returns:
        win_rates (np.ndarray): win rate per episode.
    """
    return compute_win_rate(returns, axis=0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Summarize experiment returns')
    parser.add_argument('input', help='Path to .npy file containing returns array')
    args = parser.parse_args()

    returns = np.load(args.input)
    mean, lower, upper = summary_statistics(returns)
    final_mean, final_lower, final_upper = summary_final_performance(returns)

    print(f"Final episode mean return: {final_mean:.3f} (95% CI: [{final_lower:.3f}, {final_upper:.3f}])")
    # Optionally print win rate at final episode
    from evaluation_metrics import win_rate_over_time
    win_rates = win_rate_over_time(returns)
    final_win_rate = win_rates[-1]
    print(f"Final episode win rate: {final_win_rate:.3f}")
