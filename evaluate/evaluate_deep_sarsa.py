# experiments/evaluate_sarsa.py

import argparse
import numpy as np
import torch

from agents.deep_sarsa_agent import DeepSARSAAgent
from environments.atari_env import make_atari_env
from configs.atari_configs import DQN_CONFIG

def evaluate(model_path, env_id, num_trials=500):
    """
    Load the DeepSARSAAgent from model_path, then run it greedily
    for num_trials episodes on env_id. Success if total_reward > 0.
    """
    # Temporary config just for gamma, lr, epsilon params:
    config = DQN_CONFIG.copy()
    # Build a dummy env to get shapes
    env = make_atari_env(env_id, seed=0, render=False)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    env.close()

    # Create agent and load weights
    agent = DeepSARSAAgent(input_shape=obs_shape,
                            num_actions=n_actions,
                            config=config)
    agent.q_net.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.epsilon = 0.0  # greedy

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
    parser = argparse.ArgumentParser(description="Evaluate trained Deep SARSA model")
    parser.add_argument('--model',   required=True,
                        help="Path to .pth file, e.g. models/deep_sarsa_seed0.pth")
    parser.add_argument('--env-id',  type=str, default=DQN_CONFIG['env_id'])
    parser.add_argument('--trials',  type=int, default=500,
                        help="Number of episodes for evaluation")
    args = parser.parse_args()

    sr = evaluate(args.model, args.env_id, num_trials=args.trials)
    print(f"Success rate over {args.trials} trials: {sr:.3f}")

if __name__ == '__main__':
    main()
