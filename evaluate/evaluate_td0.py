# experiments/evaluate_td0.py

import argparse
import os
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
    agent.V = V_table  # load learned values (policy depends only on threshold)

    wins = 0
    for t in range(num_trials):
        env = make_env(seed=100000 + t)
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent._best_action(state)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward

        if total_reward == 1:
            wins += 1
        env.close()

    return wins / num_trials

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all trained TD(0) V-tables on Blackjack"
    )
    parser.add_argument('--trials', type=int, default=1000,
                        help="Number of episodes per model for evaluation")
    args = parser.parse_args()

    model_dir   = os.path.join('models', 'models_td0')
    results_dir = os.path.join('results', 'results_td0')
    os.makedirs(results_dir, exist_ok=True)

    # find all .pkl files
    model_files = sorted(f for f in os.listdir(model_dir) if f.endswith('.pkl'))
    if not model_files:
        print(f"No TD(0) model files found in {model_dir}")
        return

    for fname in model_files:
        # parse threshold from filename: td0_alpha{a}_gamma{g}_th{t}_seed{s}.pkl
        parts = fname.rstrip('.pkl').split('_')
        # find the segment starting with 'th'
        thresh_part = next(seg for seg in parts if seg.startswith('th'))
        threshold = int(thresh_part[2:])

        model_path = os.path.join(model_dir, fname)
        with open(model_path, 'rb') as f:
            V = pickle.load(f)

        win_rate = evaluate(V, threshold, num_trials=args.trials)

        base = os.path.splitext(fname)[0]
        out_txt = os.path.join(results_dir, f"{base}_winrate.txt")
        with open(out_txt, 'w') as f:
            f.write(f"{win_rate:.4f}\n")
        print(f"Evaluated {fname}: win rate={win_rate:.4f} → {out_txt}")

if __name__ == '__main__':
    main()
