# experiments/evaluate_td0.py

import argparse
import pickle
import numpy as np

from environments.blackjack_env import make_env
from agents.td0_agent import TD0Agent

def evaluate(V_table, threshold, num_trials=1000):
    """
    Given a V(s) table and stick threshold, run num_trials episodes
    with the fixed threshold policy and return fraction of wins.
    """
    agent = TD0Agent(nA=2, alpha=0.0, gamma=1.0, threshold=threshold)
    agent.V = V_table  # load learned values (not used for action, but stored)

    wins = 0
    for t in range(num_trials):
        env = make_env(seed=100000 + t)
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # policy: hit if player sum < threshold else stick
            action = agent._best_action(state)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward

        if total_reward == 1:
            wins += 1
        env.close()

    return wins / num_trials

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate tabular TD(0) on Blackjack"
    )
    parser.add_argument('--model',     required=True,
                        help="Path to V-table pickle, e.g. models/td0_alpha0.1_gamma1.0_th20_seed0.pkl")
    parser.add_argument('--threshold', type=int, required=True,
                        help="Stick threshold used in training (must match model)")
    parser.add_argument('--trials',    type=int, default=1000,
                        help="Number of evaluation episodes")
    args = parser.parse_args()

    # load V-table
    with open(args.model, 'rb') as f:
        V = pickle.load(f)

    sr = evaluate(V, args.threshold, num_trials=args.trials)
    print(f"Win rate over {args.trials} trials: {sr:.3f}")

if __name__ == '__main__':
    main()
