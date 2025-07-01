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