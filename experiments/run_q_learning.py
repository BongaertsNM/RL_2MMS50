import argparse
import itertools
import matplotlib.pyplot as plt

from configs.grid_configs import QL_GRID, NUM_EPISODES, SEEDS
from train_logic.trainer import experiment
from agents.q_learning_agent import QLearningAgent
from utils.plotting import plot_learning_curves


def main():
    """
    Run Q-learning control experiments over hyperparameter grid or specific settings.
    """
    parser = argparse.ArgumentParser(description="Run Q-Learning Blackjack experiments.")
    parser.add_argument('--alpha', type=float, help='Learning rate')
    parser.add_argument('--gamma', type=float, help='Discount factor')
    parser.add_argument('--epsilon', type=float, help='Exploration rate')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    args = parser.parse_args()

    # Determine grid to use
    if args.alpha is not None and args.gamma is not None and args.epsilon is not None:
        alphas = [args.alpha]
        gammas = [args.gamma]
        epsilons = [args.epsilon]
    else:
        alphas = QL_GRID['alpha']
        gammas = QL_GRID['gamma']
        epsilons = QL_GRID['epsilon']

    # Loop through hyperparameter combinations
    for alpha, gamma, epsilon in itertools.product(alphas, gammas, epsilons):
        print(f"Running Q-Learning with alpha={alpha}, gamma={gamma}, epsilon={epsilon}")
        agent_kwargs = {'alpha': alpha, 'gamma': gamma, 'epsilon': epsilon}
        returns = experiment(
            QLearningAgent,
            num_episodes=NUM_EPISODES,
            seeds=SEEDS,
            **agent_kwargs
        )

        # Plot learning curve
        plt.figure(figsize=(8,4))
        plot_learning_curves(returns, label=f"Q-Learning α={alpha}, γ={gamma}, ε={epsilon}")
        plt.title('Q-Learning Control Learning Curve')
        plt.tight_layout()

        # Save figure
        filename = f"q_learning_alpha{alpha}_gamma{gamma}_eps{epsilon}.png"
        plt.savefig(filename)
        print(f"Saved plot to {filename}")

        if args.show:
            plt.show()
        plt.close()


if __name__ == '__main__':
    main()