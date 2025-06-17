import argparse
import sys

from experiments.run_td_experiment import main as run_td_experiment
from experiments.run_q_learning import main as run_q_learning
from experiments.run_sarsa import main as run_sarsa

def main():
    parser = argparse.ArgumentParser(
        description="Main entrypoint for Blackjack RL experiments."
    )
    # Single positional argument for algorithm choice
    parser.add_argument(
        'command',
        choices=['td', 'qlearning', 'sarsa'],
        help='Which agent to run: td | qlearning | sarsa'
    )
    # Shared hyperparameters (with defaults)
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Exploration rate (for Q-Learning & SARSA)')
    parser.add_argument('--threshold', type=int, default=20,
                        help='Stick threshold (for TD(0))')
    parser.add_argument('--show', action='store_true',
                        help='Show the learning curve interactively')

    args = parser.parse_args()

    if args.command == 'td':
        run_td_experiment()
    elif args.command == 'qlearning':
        run_q_learning()
    elif args.command == 'sarsa':
        run_sarsa()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
