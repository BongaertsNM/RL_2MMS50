# train_logic/dqn_trainer.py

import os
import numpy as np
import torch

from environments.atari_env import make_atari_env
from agents.dqn_agent import DQNAgent

def run_dqn_episode(env, agent, max_steps=1000):
    """
    Run one episode with a max step cutoff.
    """
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done and steps < max_steps:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.push_transition(obs, action, reward, next_obs, done)
        agent.update()

        total_reward += reward
        obs = next_obs
        steps += 1

    return total_reward

def train_dqn(config):
    """
    Train a DQN agent over multiple seeds and episodes.
    Saves each seed’s policy network to models/dqn_seed{seed}.pth.

    Returns:
      - all_returns: np.ndarray of shape (num_seeds, num_episodes)
      - trained_agents: list of DQNAgent instances (one per seed)
    """
    env_id       = config["env_id"]
    num_episodes = config["num_episodes"]
    seeds        = config["seeds"]

    # Prepare storage
    all_returns = np.zeros((len(seeds), num_episodes))
    trained_agents = []

    # Make sure models/ exists
    os.makedirs("models", exist_ok=True)

    for i, seed in enumerate(seeds):
        print(f"\nTraining on seed {i+1}/{len(seeds)} with env '{env_id}'")
        env = make_atari_env(env_id, seed, render=config.get("render", False))
        obs_shape = env.observation_space.shape
        n_actions = env.action_space.n

        # Initialize agent
        agent = DQNAgent(input_shape=obs_shape, num_actions=n_actions, config=config)

        # Run episodes
        for ep in range(num_episodes):
            ep_return = run_dqn_episode(env, agent)
            all_returns[i, ep] = ep_return
            print(f"  Seed {i+1}, Episode {ep+1}/{num_episodes} — Return: {ep_return:.2f}")

        # Save this seed’s trained policy network
        model_path = f"models/dqn_seed{seed}.pth"
        torch.save(agent.policy_net.state_dict(), model_path)
        print(f"Saved model to {model_path}")

        trained_agents.append(agent)

    return all_returns, trained_agents
