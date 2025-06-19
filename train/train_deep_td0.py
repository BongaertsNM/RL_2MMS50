# experiments/train_td0.py

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os

from configs.atari_configs import TD0_CONFIG
from train_logic.train_deep_td0 import train_deep_td0

def main():
    parser = argparse.ArgumentParser(description="Train Deep TD(0) on Atari")
    parser.add_argument('--episodes',   type=int,   default=TD0_CONFIG['num_episodes'])
    parser.add_argument('--num-seeds',  type=int,   default=len(TD0_CONFIG['seeds']))
    parser.add_argument('--lr',         type=float, default=None)
    parser.add_argument('--gamma',      type=float, default=None)
    parser.add_argument('--render',     action='store_true',
                        help='Enable rendering during training')
    args = parser.parse_args()

    # Build config
    config = TD0_CONFIG.copy()
    config['num_episodes'] = args.episodes
    config['seeds']        = list(range(args.num_seeds))
    config['render']       = args.render
    if args.lr    is not None: config['lr']    = args.lr
    if args.gamma is not None: config['gamma'] = args.gamma

    # Ensure models/ exists
    os.makedirs('models', exist_ok=True)

    # Train and save models
    all_returns, agents = train_deep_td0(config)
    # each model saved inside train_deep_td0 as:
    #   models/deep_td0_seed{seed}.pth

    # Plot training curve
    mean_returns = np.mean(all_returns, axis=0)
    plt.figure(figsize=(8,4))
    plt.plot(mean_returns, alpha=0.2, label='Raw Return')
    window = min(100, len(mean_returns))
    if len(mean_returns) >= window:
        smooth = np.convolve(mean_returns, np.ones(window)/window, mode='valid')
        x = np.arange(window - 1, len(mean_returns))
        if len(smooth) == len(x):
            plt.plot(x, smooth, linewidth=2, label=f'Smoothed (w={window})')

    plt.xlabel('Episode')
    plt.ylabel('Average Return')
    plt.title(f"Deep TD(0) on {config['env_id']} | lr={config['lr']}, γ={config['gamma']}")
    plt.legend()
    plt.tight_layout()

    outname = f"deep_td0_train_{args.episodes}eps_{args.num_seeds}seeds.png"
    plt.savefig(outname)
    print(f"Saved training curve to {outname}")
    plt.close()

if __name__ == '__main__':
    main()
