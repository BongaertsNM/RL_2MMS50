import gymnasium as gym


def make_env(env_name: str = 'Blackjack-v1', seed: int = None):
    """
    Create and optionally seed the Blackjack environment.

    Args:
        env_name (str): Gymnasium environment ID. Default is 'Blackjack-v1'.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        env: A Gymnasium environment instance.
    """
    env = gym.make(env_name)
    if seed is not None:
        # Reset with seed for reproducibility
        env.reset(seed=seed)
    return env
