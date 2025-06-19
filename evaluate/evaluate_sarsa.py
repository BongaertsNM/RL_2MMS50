# experiments/evaluate_sarsa.py

import argparse
import os
import pickle

import numpy as np

from environments.blackjack_env import make_env

def evaluate(q_table, num_trials=1000):
    """
    Runs num_trials episodes greedily using the provided Q-table.
    Returns fraction of episodes ending in a win (reward == +1).
    """
    # Precompute a zero‐Q fallback of the right size
    try:
        sample_q = next(iter(q_table.values()))
        zero_q   = np.zeros_like(sample_q)
    except StopIteration:
        # empty Q-table => no actions known; assume 2 actions
        zero_q = np.zeros(2)

    wins = 0
    for t in range(num_trials):
        env = make_env(seed=100000 + t)
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Safely fetch Q-values, fallback to zero_q
            q_vals = q_table.get(state, zero_q)
            action = int(np.argmax(q_vals))

            state, reward, done, _, _ = env.step(action)
            total_reward += reward

        if total_reward == 1:
            wins += 1
        env.close()

    return wins / num_trials

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all trained SARSA Q-tables on Blackjack"
    )
    parser.add_argument('--trials', type=int, default=1000,
                        help="Number of episodes per model for evaluation")
    args = parser.parse_args()

    model_dir   = os.path.join('models', 'models_sarsa')
    results_dir = os.path.join('results', 'results_sarsa')
    os.makedirs(results_dir, exist_ok=True)

    model_files = sorted(f for f in os.listdir(model_dir) if f.endswith('.pkl'))
    if not model_files:
        print(f"No SARSA model files found in {model_dir}")
        return

    for fname in model_files:
        model_path = os.path.join(model_dir, fname)
        with open(model_path, 'rb') as f:
            Q = pickle.load(f)

        win_rate = evaluate(Q, num_trials=args.trials)

        base    = os.path.splitext(fname)[0]
        out_txt = os.path.join(results_dir, f"{base}_winrate.txt")
        with open(out_txt, 'w') as f:
            f.write(f"{win_rate:.4f}\n")
        print(f"Evaluated {fname}: win rate={win_rate:.4f} → {out_txt}")

if __name__ == '__main__':
    main()
