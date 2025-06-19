# experiments/evaluate_sarsa.py

import argparse
import pickle

import numpy as np

from environments.blackjack_env import make_env

def evaluate(q_table, num_trials=1000):
    """
    Runs num_trials episodes greedily using the provided Q-table.
    Returns fraction of episodes ending in a win (reward == +1).
    """
    wins = 0
    for t in range(num_trials):
        env = make_env(seed=100000 + t)
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Greedy action
            action = int(np.argmax(q_table[state]))
            state, reward, done, _, _ = env.step(action)
            total_reward += reward

        if total_reward == 1:
            wins += 1
        env.close()

    return wins / num_trials

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SARSA Q-table on Blackjack"
    )
    parser.add_argument('--model',  required=True,
                        help="Path to Q-table pickle, e.g. models/sarsa_q_alpha0.1_gamma1.0_seed0.pkl")
    parser.add_argument('--trials', type=int, default=1000,
                        help="Number of evaluation episodes")
    args = parser.parse_args()

    # Load Q-table
    with open(args.model, 'rb') as f:
        Q = pickle.load(f)

    success_rate = evaluate(Q, num_trials=args.trials)
    print(f"Success rate over {args.trials} trials: {success_rate:.3f}")

if __name__ == '__main__':
    main()
