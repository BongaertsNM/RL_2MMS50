# experiments/evaluate_dqn.py

import argparse
import numpy as np
import torch

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
    parser = argparse.ArgumentParser(description="Evaluate trained DQN model")
    parser.add_argument('--model',   required=True,
                        help="Path to .pth file, e.g. models/dqn_seed0.pth")
    parser.add_argument('--env-id',  type=str, default=DQN_CONFIG['env_id'])
    parser.add_argument('--trials',  type=int, default=500,
                        help="Number of episodes for evaluation")
    args = parser.parse_args()

    sr = evaluate(args.model, args.env_id, num_trials=args.trials)
    print(f"Success rate over {args.trials} trials: {sr:.3f}")

if __name__ == '__main__':
    main()
