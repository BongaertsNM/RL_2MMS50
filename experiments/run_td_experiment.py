# experiments/run_td_experiment.py

import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np

from configs.grid_configs import TD0_GRID, NUM_EPISODES, SEEDS
from environments.blackjack_env import make_env
from agents.td0_agent import TD0Agent
from train_logic.trainer import run_episode


def main():
    """
    Run TD(0) prediction under a fixed policy with varying thresholds,
    then plot the per-episode average return (raw + smoothed).
    """
    parser = argparse.ArgumentParser(
        description="Run TD(0) Blackjack prediction experiments (average return)."
    )
    parser.add_argument('--episodes',   type=int, default=NUM_EPISODES,
                        help='Number of episodes per seed')
    parser.add_argument('--num-seeds',  type=int, default=len(SEEDS),
                        help='Number of independent runs (seeds 0..num-seeds-1)')
    parser.add_argument('--alpha',      type=float, default=None,
                        help='Learning rate override')
    parser.add_argument('--gamma',      type=float, default=None,
                        help='Discount factor override')
    parser.add_argument('--threshold',  type=int,   default=None,
                        help='Stick threshold override')
    parser.add_argument('--show',       action='store_true',
                        help='Show the plot interactively')
    args = parser.parse_args()

    # Determine hyperparameter lists
    if args.alpha is not None and args.gamma is not None and args.threshold is not None:
        alphas = [args.alpha]
        gammas = [args.gamma]
        thresholds = [args.threshold]
    else:
        alphas = TD0_GRID['alpha']
        gammas = TD0_GRID['gamma']
        thresholds = TD0_GRID['threshold']

    seeds    = list(range(args.num_seeds))
    episodes = args.episodes

    for alpha, gamma, threshold in itertools.product(alphas, gammas, thresholds):
        print(f"\nRunning TD(0): episodes={episodes}, seeds={args.num_seeds}, "
              f"alpha={alpha}, gamma={gamma}, threshold={threshold}")

        # Collect returns across seeds
        all_returns = np.zeros((len(seeds), episodes))
        for i, seed in enumerate(seeds):
            env   = make_env(seed=seed)
            agent = TD0Agent(nA=2, alpha=alpha, gamma=gamma, threshold=threshold)
            ep_returns = []

            for ep in range(episodes):
                G = run_episode(env, agent)
                ep_returns.append(G)

            all_returns[i, :] = ep_returns

        # Compute per-episode mean return
        mean_returns = all_returns.mean(axis=0)

        # Smooth with moving average
        window = 100
        smooth = np.convolve(mean_returns, np.ones(window)/window, mode='valid')
        x = np.arange(window-1, episodes)

        # Plot raw & smoothed average return
        plt.figure(figsize=(8, 4))
        plt.plot(mean_returns, alpha=0.2, label='Raw Return')
        plt.plot(x, smooth,      linewidth=2, label=f'Smoothed (w={window})')
        plt.xlabel('Episode')
        plt.ylabel('Average Return')
        plt.title(f'TD(0) Average Return over {episodes} Episodes\n'
                  f'α={alpha}, γ={gamma}, θ={threshold}')
        plt.legend()
        plt.tight_layout()

        # Save figure
        fname = f"td0_return_{episodes}eps_{args.num_seeds}seeds_alpha{alpha}_gamma{gamma}_th{threshold}.png"
        plt.savefig(fname)
        print(f"Saved plot to {fname}")

        if args.show:
            plt.show()
        plt.close()


if __name__ == '__main__':
    main()
