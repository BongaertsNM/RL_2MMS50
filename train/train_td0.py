# experiments/train_td0.py

import argparse
import itertools
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from configs.grid_configs import TD0_GRID, NUM_EPISODES, SEEDS
from environments.blackjack_env import make_env
from agents.td0_agent import TD0Agent
from train_logic.trainer import run_episode
from utils.metrics import compute_win_rate

def main():
    parser = argparse.ArgumentParser(
        description="Train tabular TD(0) on Blackjack"
    )
    parser.add_argument('--episodes',  type=int,   default=NUM_EPISODES,
                        help='Episodes per seed')
    parser.add_argument('--num-seeds', type=int,   default=len(SEEDS),
                        help='Number of independent runs')
    parser.add_argument('--alpha',     type=float, default=None,
                        help='Learning rate override')
    parser.add_argument('--gamma',     type=float, default=None,
                        help='Discount factor override')
    parser.add_argument('--threshold', type=int,   default=None,
                        help='Stick threshold override')
    args = parser.parse_args()

    # choose hyperparameter lists
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

    # Prepare directories for models and results
    model_dir   = os.path.join('models',  'models_td0')
    results_dir = os.path.join('results', 'results_td0')
    os.makedirs(model_dir,   exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    for alpha, gamma, thresh in itertools.product(alphas, gammas, thresholds):
        print(f"\nTraining TD(0): α={alpha}, γ={gamma}, θ={thresh}, seeds={len(seeds)}, eps={episodes}")
        all_returns = np.zeros((len(seeds), episodes))

        for i, seed in enumerate(seeds):
            env   = make_env(seed=seed)
            agent = TD0Agent(nA=2, alpha=alpha, gamma=gamma, threshold=thresh)
            ep_returns = []
            for ep in range(episodes):
                G = run_episode(env, agent)
                ep_returns.append(G)
            all_returns[i, :] = ep_returns

            # Only save seed 0's V-table
            if i == 0:
                model_path = os.path.join(
                    model_dir,
                    f"td0_alpha{alpha}_gamma{gamma}_th{thresh}_seed{seed}.pkl"
                )
                with open(model_path, 'wb') as f:
                    pickle.dump(agent.V, f)
                print(f"  Saved V-table for seed {seed} to {model_path}")

        # Plot win-rate curve
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
        plt.title(f"TD(0) Win Rate (α={alpha}, γ={gamma}, θ={thresh})")
        plt.legend()
        plt.tight_layout()

        plot_fname = f"td0_winrate_{episodes}eps_{len(seeds)}seeds_a{alpha}_g{gamma}_th{thresh}.png"
        out_path   = os.path.join(results_dir, plot_fname)
        plt.savefig(out_path)
        print(f"  Saved curve to {out_path}")
        plt.close()

if __name__ == '__main__':
    main()
