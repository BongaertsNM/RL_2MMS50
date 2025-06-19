# experiments/train_deep_td0.py

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from configs.atari_configs import TD0_CONFIG
from train_logic.deep_td0_trainer import train_deep_td0

def main():
    parser = argparse.ArgumentParser(
        description="Train Deep TD(0) on Atari and plot episodic returns for one seed"
    )
    parser.add_argument('--env-id',   type=str,
                        default=TD0_CONFIG['env_id'],
                        help='Gymnasium env ID (e.g. ALE/Boxing-v5)')
    parser.add_argument('--episodes', type=int,
                        default=TD0_CONFIG['num_episodes'],
                        help='Number of episodes to run')
    parser.add_argument('--lr',       type=float,
                        default=None,
                        help='Learning rate override')
    parser.add_argument('--gamma',    type=float,
                        default=None,
                        help='Discount factor override')
    parser.add_argument('--render',   action='store_true',
                        help='Enable env rendering during training')
    args = parser.parse_args()

    # Build config for a single seed
    config = TD0_CONFIG.copy()
    config['env_id']       = args.env_id
    config['num_episodes'] = args.episodes
    config['seeds']        = [0]            # only seed 0
    config['render']       = args.render
    if args.lr    is not None:
        config['lr']    = args.lr
    if args.gamma is not None:
        config['gamma'] = args.gamma

    # Prepare model & results directories
    model_dir   = os.path.join('models',  'models_deep_td0')
    results_dir = os.path.join('results', 'results_deep_td0')
    os.makedirs(model_dir,   exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Train and collect returns + agents
    all_returns, agents = train_deep_td0(config)

    # Save the trained value network for seed 0
    agent     = agents[0]
    env_name  = args.env_id.replace('/', '_')
    model_path = os.path.join(
        model_dir,
        f"deep_td0_{env_name}_seed0.pth"
    )
    torch.save(agent.value_net.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Plot episodic returns for seed 0
    returns = all_returns[0]
    plt.figure(figsize=(8, 4))
    plt.plot(returns, label='Episodic Return')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f"Deep TD(0) Returns on {args.env_id} (seed=0)")
    plt.legend()
    plt.tight_layout()

    plot_name = f"deep_td0_{env_name}_{args.episodes}eps_seed0.png"
    plot_path = os.path.join(results_dir, plot_name)
    plt.savefig(plot_path)
    print(f"Saved episodic return plot to {plot_path}")
    plt.close()

if __name__ == '__main__':
    main()
