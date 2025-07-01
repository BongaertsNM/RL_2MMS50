DQN_CONFIG = {
    "env_id": "ALE/Boxing-v5",  # or another valid ALE env you installed via AutoROM
    "num_episodes": 1000,
    "seeds": [0],
    "lr": 1e-4,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 10000,
    "buffer_size": 10000,
    "batch_size": 32,
    "target_update_freq": 1000,
}

# Inside configs/atari_configs.py

TD0_CONFIG = {
    "env_id": "ALE/Boxing-v5",
    "num_episodes": 1000,
    "seeds": [0],
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.1,
    "epsilon_decay_steps": 100000,
    "lr": 1e-4,
    "buffer_size": 10000,
    "batch_size": 32,
    "render": False
}
