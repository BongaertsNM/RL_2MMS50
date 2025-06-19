# train/train_deep_sarsa.py

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os

from configs.atari_configs import DQN_CONFIG
from train_logic.deep_sarsa_trainer import train_deep_sarsa

def main():
    parser = argparse.ArgumentParser(
        description="Train Deep SARSA on Atari and plot episodic returns for one seed"
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
    config['seeds']        = [0]            # only one seed: 0
    config['render']       = args.render
    if args.lr    is not None: config['lr']    = args.lr
    if args.gamma is not None: config['gamma'] = args.gamma

    # Prepare directories
    model_dir   = os.path.join('models',  'models_deep_sarsa')
    results_dir = os.path.join('results', 'results_deep_sarsa')
    os.makedirs(model_dir,   exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Train
    all_returns, agents = train_deep_sarsa(config)

    # Save the trained model (policy network) for seed 0
    agent = agents[0]
    model_path = os.path.join(
        model_dir,
        f"deep_sarsa_{args.env_id.replace('/', '_')}_seed0.pth"
    )
    torch.save(agent.q_net.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Extract the single seed's returns
    returns = all_returns[0]  # shape (episodes,)

    # Plot raw episodic returns
    plt.figure(figsize=(8, 4))
    plt.plot(returns, label='Episodic Return')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f"Deep SARSA Returns on {args.env_id} (seed=0)")
    plt.legend()
    plt.tight_layout()

    outname = f"deep_sarsa_returns_{args.env_id.replace('/', '_')}_{args.episodes}eps_seed0.png"
    outpath = os.path.join(results_dir, outname)
    plt.savefig(outpath)
    print(f"Saved episodic return plot to {outpath}")
    plt.close()

if __name__ == '__main__':
    main()
