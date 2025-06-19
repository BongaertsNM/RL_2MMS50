# train_logic/deep_sarsa_trainer.py

import os
import numpy as np
import torch

from environments.atari_env import make_atari_env
from agents.deep_sarsa_agent import DeepSARSAAgent

def run_sarsa_episode(env, agent):
    """
    Run one episode using the given DeepSARSAAgent.
    """
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    action = agent.select_action(obs)

    while not done:
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_action = agent.select_action(next_obs)

        agent.push_transition(obs, action, reward, next_obs, next_action, done)
        agent.update()

        total_reward += reward
        obs, action = next_obs, next_action

    return total_reward

def train_deep_sarsa(config):
    """
    Train a Deep SARSA agent over multiple seeds and episodes.
    Returns:
      - all_returns: np.ndarray of shape (num_seeds, num_episodes)
      - agents: list of trained DeepSARSAAgent instances (one per seed)
    Also saves each agent’s Q-network state_dict to models/deep_sarsa_seed{seed}.pth
    """
    env_id       = config["env_id"]
    num_episodes = config["num_episodes"]
    seeds        = config["seeds"]

    # Prepare output arrays and folder
    all_returns = np.zeros((len(seeds), num_episodes))
    trained_agents = []
    os.makedirs("models", exist_ok=True)

    for i, seed in enumerate(seeds):
        print(f"\nTraining on seed {i+1}/{len(seeds)} with env '{env_id}'")
        env = make_atari_env(env_id, seed, render=config.get("render", False))
        obs_shape = env.observation_space.shape
        n_actions = env.action_space.n

        agent = DeepSARSAAgent(
            input_shape=obs_shape,
            num_actions=n_actions,
            config=config
        )

        for ep in range(num_episodes):
            ep_return = run_sarsa_episode(env, agent)
            all_returns[i, ep] = ep_return
            print(f"  Seed {i+1}, Episode {ep+1}/{num_episodes}: Return = {ep_return:.2f}")

        # Save this seed’s trained model
        model_path = os.path.join("models", f"deep_sarsa_seed{seed}.pth")
        torch.save(agent.q_net.state_dict(), model_path)
        print(f"Saved model to {model_path}")

        trained_agents.append(agent)

    return all_returns, trained_agents
