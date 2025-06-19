# experiments/evaluate_dqn.py

import argparse
import os
import torch
import numpy as np

from agents.dqn_agent import DQNAgent
from environments.atari_env import make_atari_env
from configs.atari_configs import DQN_CONFIG

def evaluate(model_path, env_id, num_trials=500):
    """
    Load the DQNAgent policy_net from model_path,
    then run num_trials episodes greedily and return
    the fraction with total_reward > 0.
    """
    # Build dummy env to get shapes
    env = make_atari_env(env_id, seed=0, render=False)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    env.close()

    # Create agent and load weights
    agent = DQNAgent(input_shape=obs_shape,
                     num_actions=n_actions,
                     config=DQN_CONFIG)
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.epsilon = 0.0  # force greedy

    successes = 0
    for _ in range(num_trials):
        env = make_atari_env(env_id, seed=None, render=False)
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(obs)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += r

        if total_reward > 0:
            successes += 1
        env.close()

    return successes / num_trials

def main():
    parser = argparse.ArgumentParser(description="Evaluate all trained DQN models")
    parser.add_argument('--env-id', type=str, default=DQN_CONFIG['env_id'],
                        help="Gymnasium env ID (e.g. ALE/Boxing-v5)")
    parser.add_argument('--trials', type=int, default=500,
                        help="Number of episodes per model for evaluation")
    args = parser.parse_args()

    model_dir   = os.path.join('models', 'models_dqn')
    results_dir = os.path.join('results', 'results_dqn')
    os.makedirs(results_dir, exist_ok=True)

    model_files = sorted(f for f in os.listdir(model_dir) if f.endswith('.pth'))
    if not model_files:
        print(f"No DQN model files found in {model_dir}")
        return

    for fname in model_files:
        model_path = os.path.join(model_dir, fname)
        win_rate = evaluate(model_path, args.env_id, num_trials=args.trials)

        base   = os.path.splitext(fname)[0]
        out_txt = os.path.join(results_dir, f"{base}_winrate.txt")
        with open(out_txt, 'w') as f:
            f.write(f"{win_rate:.4f}\n")
        print(f"Evaluated {fname}: success rate={win_rate:.4f} → {out_txt}")

if __name__ == '__main__':
    main()
