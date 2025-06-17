from collections import defaultdict
from .base_agent import Agent

class TD0Agent(Agent):
    """
    TD(0) prediction agent for a fixed policy in Blackjack.

    Learns state-value function V(s) using the TD(0) update rule under
    a simple heuristic policy (hit if player sum < threshold, else stick).
    """

    def __init__(self, nA=2, alpha=0.1, gamma=1.0, threshold=20):
        """
        Args:
            nA (int): Number of actions (default 2: stick or hit).
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            threshold (int): Player sum threshold for stick/hit policy.
        """
        super().__init__(nA=nA, alpha=alpha, gamma=gamma, epsilon=0.0)
        self.V = defaultdict(float)
        self.threshold = threshold

    def _best_action(self, state):
        """
        Heuristic policy: hit if player sum is below threshold, else stick.
        """
        player_sum, _, _ = state
        return 1 if player_sum < self.threshold else 0

    def update(self, state, action, reward, next_state, next_action=None, done=False):
        """
        Perform TD(0) update for the state-value function.

        V(s) <- V(s) + alpha [r + gamma V(s') - V(s)]
        """
        v = self.V[state]
        v_next = 0.0 if done else self.V[next_state]
        td_error = reward + self.gamma * v_next - v
        self.V[state] += self.alpha * td_error
