import numpy as np
from collections import defaultdict
from .base_agent import Agent

class QLearningAgent(Agent):
    """
    Off-policy Q-learning control agent.

    Learns the optimal action-value function Q*(s,a) via the Q-learning update.
    Uses an epsilon-greedy behavior policy and a greedy target policy.
    """

    def __init__(self, nA=2, alpha=0.1, gamma=1.0, epsilon=0.1):
        """
        Args:
            nA (int): Number of actions.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration rate for epsilon-greedy policy.
        """
        super().__init__(nA=nA, alpha=alpha, gamma=gamma, epsilon=epsilon)
        # Initialize Q(s,a) to zero for all state-action pairs
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def _best_action(self, state):
        """
        Return the greedy action for the current state under Q.
        """
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, next_action=None, done=False):
        """
        Perform the Q-learning update:
        Q(s,a) <- Q(s,a) + alpha [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        q_sa = self.Q[state][action]
        # Compute maximum Q for next_state (0 if terminal)
        q_next = 0.0 if done else np.max(self.Q[next_state])
        td_target = reward + self.gamma * q_next
        td_error = td_target - q_sa
        self.Q[state][action] += self.alpha * td_error
