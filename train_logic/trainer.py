import numpy as np
from environments.blackjack_env import make_env
from collections import defaultdict


def run_episode(env, agent):
    """
    Run a single episode in the environment using the given agent.

    Args:
        env: Gymnasium environment.
        agent: An instance of Agent.

    Returns:
        total_reward (float): Sum of rewards collected in the episode.
    """
    state, _ = env.reset()
    action = agent.select_action(state)
    total_reward = 0.0

    while True:
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        if hasattr(agent, 'update') and agent is not None:
            if 'next_action' in agent.update.__code__.co_varnames:
                # On-policy: SARSA needs next_action
                next_action = agent.select_action(next_state)
                agent.update(state, action, reward, next_state, next_action, done)
            else:
                # Off-policy or TD0: update without next_action
                agent.update(state, action, reward, next_state, None, done)
                next_action = agent.select_action(next_state)
        else:
            next_action = agent.select_action(next_state)

        state, action = next_state, next_action
        if done:
            break

    return total_reward


def experiment(agent_class, env_name='Blackjack-v1', num_episodes=50000,
               seeds=None, **agent_kwargs):
    """
    Run multiple independent experiment runs for a given agent.

    Args:
        agent_class: Agent class to instantiate.
        env_name (str): Gymnasium env ID.
        num_episodes (int): Episodes per run.
        seeds (list[int], optional): Random seeds for each run.
        agent_kwargs: Parameters to pass to agent constructor.

    Returns:
        all_returns (np.ndarray): shape (n_runs, num_episodes) of episode returns.
    """
    if seeds is None:
        seeds = list(range(30))

    n_runs = len(seeds)
    all_returns = np.zeros((n_runs, num_episodes))

    for i, seed in enumerate(seeds):
        env = make_env(env_name, seed)
        agent = agent_class(**agent_kwargs)
        returns = []
        for ep in range(num_episodes):
            G = run_episode(env, agent)
            returns.append(G)
        all_returns[i, :] = returns

    return all_returns
