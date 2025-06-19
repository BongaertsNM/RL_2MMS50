# experiments/train_sarsa.py

import argparse
import itertools
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from configs.grid_configs import SARSA_GRID, NUM_EPISODES, SEEDS
from environments.blackjack_env import make_env
from agents.sarsa_agent import SARSAgent
from train_logic.trainer import run_episode
from utils.metrics import compute_win_rate


def main():
    parser = argparse.ArgumentParser(
        description="Train tabular SARSA on Blackjack"
    )
    parser.add_argument('--episodes',   type=int, default=NUM_EPISODES,
                        help='Episodes per seed')
    parser.add_argument('--num-seeds',  type=int, default=len(SEEDS),
                        help='Number of independent runs')
    parser.add_argument('--alpha',      type=float, default=None,
                        help='Learning rate override')
    parser.add_argument('--gamma',      type=float, default=None,
                        help='Discount factor override')
    args = parser.parse_args()

    # Determine which α and γ to run
    if args.alpha is not None and args.gamma is not None:
        alphas = [args.alpha]
        gammas = [args.gamma]
    else:
        alphas = SARSA_GRID['alpha']
        gammas = SARSA_GRID['gamma']

    seeds    = list(range(args.num_seeds))
    episodes = args.episodes

    # Prepare directories
    model_dir   = os.path.join('models', 'models_sarsa')
    results_dir = os.path.join('results', 'results_sarsa')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    for alpha, gamma in itertools.product(alphas, gammas):
        print(f"\nTraining SARSA: α={alpha}, γ={gamma}, seeds={len(seeds)}, eps={episodes}")
        all_returns = np.zeros((len(seeds), episodes))

        for i, seed in enumerate(seeds):
            env   = make_env(seed=seed)
            agent = SARSAgent(nA=2, alpha=alpha, gamma=gamma, epsilon=1.0)
            ep_returns = []

            for ep in range(episodes):
                agent.epsilon = max(0.01, 1.0 - ep/episodes)
                G = run_episode(env, agent)
                ep_returns.append(G)

            all_returns[i, :] = ep_returns

            # Save only seed 0's Q-table for this (alpha, gamma) config
            if i == 0:
                model_path = os.path.join(
                    model_dir,
                    f"sarsa_q_alpha{alpha}_gamma{gamma}_seed{seed}.pkl"
                )
                with open(model_path, 'wb') as f:
                    pickle.dump(dict(agent.Q), f)
                print(f"  Saved Q-table for seed {seed} to {model_path}")

        # Plot learning curve (win rate)
        win_rates = compute_win_rate(all_returns, axis=0)
        window    = 100
        smooth    = np.convolve(win_rates, np.ones(window)/window, mode='valid')
        x         = np.arange(window-1, episodes)

        plt.figure(figsize=(8,4))
        plt.plot(win_rates, alpha=0.2, label='Raw Win Rate')
        if len(smooth) == len(x):
            plt.plot(x, smooth, linewidth=2, label=f'Smoothed (w={window})')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.ylim(0,1)
        plt.title(f"SARSA Win Rate (α={alpha}, γ={gamma})")
        plt.legend()
        plt.tight_layout()

        plot_fname = f"sarsa_winrate_{episodes}eps_{len(seeds)}seeds_a{alpha}_g{gamma}.png"
        out_path   = os.path.join(results_dir, plot_fname)
        plt.savefig(out_path)
        print(f"  Saved curve to {out_path}")
        plt.close()


if __name__ == '__main__':
    main()
