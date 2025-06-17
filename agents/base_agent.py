import numpy as np
from abc import ABC, abstractmethod

class Agent(ABC):
    """
    Abstract base class for RL agents.

    Attributes:
        nA (int): Number of actions.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate for epsilon-greedy policies.
    """

    def __init__(self, nA=2, alpha=0.1, gamma=1.0, epsilon=0.1):
        self.nA = nA
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        """
        Epsilon-greedy action selection.

        Args:
            state: The current state.

        Returns:
            action (int): Selected action.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nA)
        return self._best_action(state)

    @abstractmethod
    def _best_action(self, state):  # pragma: no cover
        """
        Returns the best action for a given state under the agent's current estimate.
        Must be implemented by subclasses.

        Args:
            state: The current state.

        Returns:
            action (int): Greedy action.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, state, action, reward, next_state, next_action=None, done=False):
        """
        Update the agent's internal estimates based on observed transition.

        Args:
            state: Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state: Next state.
            next_action (int): Action selected in next state (for on-policy).
            done (bool): Whether the episode is done.
        """
        raise NotImplementedError
