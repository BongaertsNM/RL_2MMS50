# environments/atari_env.py

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack

def make_atari_env(env_id: str, seed: int, render: bool = False):
    """
    Creates and returns a preprocessed Atari environment with proper frame skipping.
    """
    env = gym.make(env_id, frameskip=1, render_mode="human" if render else None)

    env = AtariPreprocessing(
        env,
        grayscale_obs=True,
        scale_obs=True,
        frame_skip=12,
        screen_size=84
    )

    env = FrameStack(env, num_stack=4)
    env.reset(seed=seed)
    return env
