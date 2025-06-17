# experiments/run_td_experiment.py

import argparse
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend: no GUI windows
import matplotlib.pyplot as plt

from configs.grid_configs import TD0_GRID, NUM_EPISODES, SEEDS
from environments.blackjack_env import make_env
from agents.td0_agent import TD0Agent
from train_logic.trainer import run_episode
from utils.metrics import compute_win_rate

def main():
    """
    Run TD(0) prediction under a fixed stick-threshold policy,
    then plot the per-episode win rate (raw + smoothed) without opening windows.
    """
    parser = argparse.ArgumentParser(
        description="Run TD(0) Blackjack prediction experiments (win rate)."
    )
    parser.add_argument('--episodes',   type=int, default=NUM_EPISODES,
                        help='Number of episodes per seed')
    parser.add_argument('--num-seeds',  type=int, default=len(SEEDS),
                        help='Number of independent runs (seeds 0..num-seeds-1)')
    parser.add_argument('--alpha',      type=float, default=None,
                        help='Learning rate override')
    parser.add_argument('--gamma',      type=float, default=None,
                        help='Discount factor override')
    parser.add_argument('--threshold',  type=int, default=None,
                        help='Stick threshold override')
    args = parser.parse_args()

    # Decide which hyperparameters to run
    if args.alpha is not None and args.gamma is not None and args.threshold is not None:
        alphas     = [args.alpha]
        gammas     = [args.gamma]
        thresholds = [args.threshold]
    else:
        alphas     = TD0_GRID['alpha']
        gammas     = TD0_GRID['gamma']
        thresholds = TD0_GRID['threshold']

    seeds    = list(range(args.num_seeds))
    episodes = args.episodes

    # Loop over all combinations
    for alpha, gamma, threshold in itertools.product(alphas, gammas, thresholds):
        print(f"\nRunning TD(0): episodes={episodes}, seeds={args.num_seeds}, "
              f"alpha={alpha}, gamma={gamma}, threshold={threshold}")

        # Collect per-episode returns for each seed
        all_returns = np.zeros((len(seeds), episodes))
        for i, seed in enumerate(seeds):
            env   = make_env(seed=seed)
            agent = TD0Agent(nA=2, alpha=alpha, gamma=gamma, threshold=threshold)
            ep_returns = []
            for ep in range(episodes):
                G = run_episode(env, agent)
                ep_returns.append(G)
            all_returns[i, :] = ep_returns

        # Compute win rate (fraction of +1 returns) per episode
        win_rates = compute_win_rate(all_returns, axis=0)  # shape: (episodes,)

        # Smooth with moving average
        window = 100
        smooth = np.convolve(win_rates, np.ones(window)/window, mode='valid')
        x = np.arange(window-1, episodes)

        # Plot raw & smoothed win rate
        plt.figure(figsize=(8, 4))
        plt.plot(win_rates, alpha=0.2, label='Raw Win Rate')
        plt.plot(x, smooth,      linewidth=2, label=f'Smoothed (w={window})')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.ylim(0, 1)
        plt.title(f'TD(0) Win Rate over {episodes} Episodes\n'
                  f'α={alpha}, γ={gamma}, θ={threshold}')
        plt.legend()
        plt.tight_layout()

        # Save figure
        fname = f"td0_winrate_{episodes}eps_{args.num_seeds}seeds_alpha{alpha}_gamma{gamma}_th{threshold}.png"
        plt.savefig(fname)
        print(f"Saved plot to {fname}")
        plt.close()

if __name__ == '__main__':
    main()
