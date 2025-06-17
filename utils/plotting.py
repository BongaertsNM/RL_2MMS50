import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(returns, label=None):
    """
    Plot mean return with 95% confidence interval shading.

    Args:
        returns (np.ndarray): Array of shape (n_runs, n_episodes) with episode returns.
        label (str, optional): Label for the plot.
    """
    means = np.mean(returns, axis=0)
    sems = np.std(returns, axis=0, ddof=1) / np.sqrt(returns.shape[0])
    episodes = np.arange(1, returns.shape[1] + 1)

    plt.plot(episodes, means, label=label)
    plt.fill_between(episodes,
                     means - 1.96 * sems,
                     means + 1.96 * sems,
                     alpha=0.2)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    if label:
        plt.legend()
    plt.tight_layout()
