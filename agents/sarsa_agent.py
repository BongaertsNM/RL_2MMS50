import numpy as np
from collections import defaultdict
from .base_agent import Agent

class SARSAgent(Agent):
    """
    On-policy SARSA control agent.

    Learns the action-value function Q(s,a) under the current epsilon-greedy policy.
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
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def _best_action(self, state):
        """
        Return the greedy action for the current state under Q.
        """
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, next_action, done=False):
        """
        Perform the SARSA update:
        Q(s,a) <- Q(s,a) + alpha [r + gamma * Q(s',a') - Q(s,a)]
        """
        q_sa = self.Q[state][action]
        q_next = 0.0 if done else self.Q[next_state][next_action]
        td_target = reward + self.gamma * q_next
        td_error = td_target - q_sa
        self.Q[state][action] += self.alpha * td_error
