# Hyperparameter grids for Blackjack RL experiments

# Common settings
NUM_EPISODES = 10000
SEEDS = list(range(5))

# TD0 prediction hyperparameters
TD0_GRID = {
    'alpha': [0.1, 0.01],        # learning rates
    'gamma': [0.9, 1.0],         # discount factors
    'threshold': [18, 19, 20]    # stick thresholds for the policy
}

# Q-Learning control hyperparameters
QL_GRID = {
    'alpha': [0.1, 0.01],        # learning rates
    'gamma': [0.9, 1.0],         # discount factors
    'epsilon': [0.1, 0.01],      # exploration rates
}

# SARSA control hyperparameters
SARSA_GRID = {
    'alpha': [0.1, 0.01],        # learning rates
    'gamma': [0.9, 1.0],         # discount factors
    'epsilon': [0.1, 0.01],      # exploration rates
}

# Consolidated grid for convenience
HYPERPARAMETER_GRID = {
    'TD0Agent': TD0_GRID,
    'QLearningAgent': QL_GRID,
    'SARSAgent': SARSA_GRID,
}
