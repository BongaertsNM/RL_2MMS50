# experiments/run_sarsa.py

import argparse
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend: no GUI windows
import matplotlib.pyplot as plt

from configs.grid_configs import SARSA_GRID, NUM_EPISODES, SEEDS
from environments.blackjack_env import make_env
from agents.sarsa_agent import SARSAgent
from train_logic.trainer import run_episode
from utils.metrics import compute_win_rate

def main():
    """
    Run SARSA control with per-episode epsilon decay,
    then plot the per-episode win rate (raw + smoothed) without opening windows.
    """
    parser = argparse.ArgumentParser(
        description="Run SARSA Blackjack experiments (win rate)."
    )
    parser.add_argument('--episodes',   type=int, default=NUM_EPISODES,
                        help='Number of episodes per seed')
    parser.add_argument('--num-seeds',  type=int, default=len(SEEDS),
                        help='Number of independent runs (seeds 0..num-seeds-1)')
    parser.add_argument('--alpha',      type=float, default=None,
                        help='Learning rate override')
    parser.add_argument('--gamma',      type=float, default=None,
                        help='Discount factor override')
    args = parser.parse_args()

    # Determine hyperparameters
    if args.alpha is not None and args.gamma is not None:
        alphas = [args.alpha]
        gammas = [args.gamma]
    else:
        alphas = SARSA_GRID['alpha']
        gammas = SARSA_GRID['gamma']

    seeds    = list(range(args.num_seeds))
    episodes = args.episodes

    for alpha, gamma in itertools.product(alphas, gammas):
        print(f"\nRunning SARSA: episodes={episodes}, seeds={args.num_seeds}, α={alpha}, γ={gamma}")

        # Collect returns across seeds
        all_returns = np.zeros((len(seeds), episodes))
        for i, seed in enumerate(seeds):
            env   = make_env(seed=seed)
            agent = SARSAgent(nA=2, alpha=alpha, gamma=gamma, epsilon=1.0)
            ep_returns = []

            for ep in range(episodes):
                # Linear decay of ε from 1.0 down to 0.01
                agent.epsilon = max(0.01, 1.0 - ep/episodes)
                G = run_episode(env, agent)
                ep_returns.append(G)

            all_returns[i, :] = ep_returns

        # Compute per-episode win rate
        win_rates = compute_win_rate(all_returns, axis=0)

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
        plt.title(f'SARSA Win Rate over {episodes} Episodes\nα={alpha}, γ={gamma}')
        plt.legend()
        plt.tight_layout()

        # Save figure
        fname = f"sarsa_winrate_{episodes}eps_{args.num_seeds}seeds_alpha{alpha}_gamma{gamma}.png"
        plt.savefig(fname)
        print(f"Saved plot to {fname}")
        plt.close()

if __name__ == '__main__':
    main()
