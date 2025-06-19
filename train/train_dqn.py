# train/train_dqn.py

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from configs.atari_configs import DQN_CONFIG
from train_logic.dqn_trainer import train_dqn

def main():
    parser = argparse.ArgumentParser(
        description="Train DQN on Atari and plot episodic returns for one seed"
    )
    parser.add_argument('--env-id',   type=str,   default=DQN_CONFIG['env_id'],
                        help='Gymnasium env ID (e.g. ALE/Boxing-v5)')
    parser.add_argument('--episodes', type=int,   default=DQN_CONFIG['num_episodes'],
                        help='Number of episodes to run')
    parser.add_argument('--lr',       type=float, default=None,
                        help='Learning rate override')
    parser.add_argument('--gamma',    type=float, default=None,
                        help='Discount factor override')
    parser.add_argument('--render',   action='store_true',
                        help='Enable env rendering during training')
    args = parser.parse_args()

    # Build config for a single seed
    config = DQN_CONFIG.copy()
    config['env_id']       = args.env_id
    config['num_episodes'] = args.episodes
    config['seeds']        = [0]            # only seed 0
    config['render']       = args.render
    if args.lr    is not None:
        config['lr']    = args.lr
    if args.gamma is not None:
        config['gamma'] = args.gamma

    # Ensure models/ exists (trainer will save the trained model)
    os.makedirs('models', exist_ok=True)

    # Train
    all_returns, agents = train_dqn(config)

    # Extract the single seed's returns
    returns = all_returns[0]  # shape (episodes,)

    # Plot raw episodic returns
    plt.figure(figsize=(8, 4))
    plt.plot(returns, label='Episodic Return')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f"DQN Returns on {args.env_id} (seed=0)")
    plt.legend()
    plt.tight_layout()

    outname = f"dqn_returns_{args.episodes}eps_seed0.png"
    plt.savefig(outname)
    print(f"Saved episodic return plot to {outname}")
    plt.close()

if __name__ == '__main__':
    main()
