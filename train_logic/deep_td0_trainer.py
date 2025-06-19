# train_logic/deep_td0_trainer.py

import os
import numpy as np
import torch

from environments.atari_env import make_atari_env
from agents.deep_td0_agent import DeepTD0Agent

def run_td0_episode(env, agent):
    """
    Run one episode using the given DeepTD0Agent.
    """
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # TD-0 update
        agent.update(obs, reward, next_obs, done)

        total_reward += reward
        obs = next_obs

    return total_reward

def train_deep_td0(config):
    """
    Train a Deep TD(0) agent over multiple seeds and episodes.
    Returns:
      - all_returns: np.ndarray of shape (num_seeds, num_episodes)
      - trained_agents: list of DeepTD0Agent instances (one per seed)
    Also saves each agent’s value network to models/deep_td0_seed{seed}.pth
    """
    env_id       = config["env_id"]
    num_episodes = config["num_episodes"]
    seeds        = config["seeds"]

    # Prepare storage & output dir
    all_returns    = np.zeros((len(seeds), num_episodes))
    trained_agents = []
    os.makedirs("models", exist_ok=True)

    for i, seed in enumerate(seeds):
        print(f"\nTraining on seed {i+1}/{len(seeds)} with env '{env_id}'")
        env = make_atari_env(env_id, seed, render=config.get("render", False))
        obs_shape = env.observation_space.shape
        n_actions = env.action_space.n

        # Initialize your agent with num_actions
        agent = DeepTD0Agent(
            input_shape=obs_shape,
            num_actions=n_actions,
            config=config
        )

        # Run episodes
        for ep in range(num_episodes):
            ep_return = run_td0_episode(env, agent)
            all_returns[i, ep] = ep_return
            print(f"  Seed {i+1}, Episode {ep+1}/{num_episodes}: Return = {ep_return:.2f}")

        trained_agents.append(agent)

    return all_returns, trained_agents
