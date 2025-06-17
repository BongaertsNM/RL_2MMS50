import argparse
import sys

from experiments.run_td_experiment import main as run_td_experiment
from experiments.run_q_learning import main as run_q_learning
from experiments.run_sarsa import main as run_sarsa


def main():
    parser = argparse.ArgumentParser(
        description="Main entrypoint for Blackjack RL experiments."
    )
    subparsers = parser.add_subparsers(dest='command', required=True,
                                       help='Choose experiment to run')

    # TD(0) subcommand
    parser_td = subparsers.add_parser('td', help='Run TD(0) prediction experiments')
    parser_td.add_argument('--alpha', type=float, help='Learning rate')
    parser_td.add_argument('--gamma', type=float, help='Discount factor')
    parser_td.add_argument('--threshold', type=int, help='Stick threshold')
    parser_td.add_argument('--show', action='store_true', help='Show plots interactively')

    # Q-learning subcommand
    parser_ql = subparsers.add_parser('qlearning', help='Run Q-Learning control experiments')
    parser_ql.add_argument('--alpha', type=float, help='Learning rate')
    parser_ql.add_argument('--gamma', type=float, help='Discount factor')
    parser_ql.add_argument('--epsilon', type=float, help='Exploration rate')
    parser_ql.add_argument('--show', action='store_true', help='Show plots interactively')

    # SARSA subcommand
    parser_sarsa = subparsers.add_parser('sarsa', help='Run SARSA control experiments')
    parser_sarsa.add_argument('--alpha', type=float, help='Learning rate')
    parser_sarsa.add_argument('--gamma', type=float, help='Discount factor')
    parser_sarsa.add_argument('--epsilon', type=float, help='Exploration rate')
    parser_sarsa.add_argument('--show', action='store_true', help='Show plots interactively')

    args = parser.parse_args()

    # Dispatch to appropriate experiment
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
