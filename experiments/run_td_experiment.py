import argparse
import itertools
import matplotlib.pyplot as plt

from configs.grid_configs import TD0_GRID, NUM_EPISODES, SEEDS
from train_logic.trainer import experiment
from agents.td0_agent import TD0Agent
from utils.plotting import plot_learning_curves


def main():
    """
    Run TD(0) prediction experiments over hyperparameter grid or specific settings.
    """
    parser = argparse.ArgumentParser(description="Run TD(0) Blackjack experiments.")
    parser.add_argument('--alpha', type=float, help='Learning rate')
    parser.add_argument('--gamma', type=float, help='Discount factor')
    parser.add_argument('--threshold', type=int, help='Stick threshold')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    args = parser.parse_args()

    # Determine grid to use
    if args.alpha is not None and args.gamma is not None and args.threshold is not None:
        alphas = [args.alpha]
        gammas = [args.gamma]
        thresholds = [args.threshold]
    else:
        alphas = TD0_GRID['alpha']
        gammas = TD0_GRID['gamma']
        thresholds = TD0_GRID['threshold']

    # Loop through hyperparameter combinations
    for alpha, gamma, threshold in itertools.product(alphas, gammas, thresholds):
        print(f"Running TD(0) with alpha={alpha}, gamma={gamma}, threshold={threshold}")
        agent_kwargs = {'alpha': alpha, 'gamma': gamma, 'threshold': threshold}
        returns = experiment(
            TD0Agent,
            num_episodes=NUM_EPISODES,
            seeds=SEEDS,
            **agent_kwargs
        )

        # Plot learning curve
        plt.figure(figsize=(8,4))
        plot_learning_curves(returns, label=f"\nTD(0) α={alpha}, γ={gamma}, θ={threshold}")
        plt.title('TD(0) Prediction Learning Curve')
        plt.tight_layout()

        # Save figure
        filename = f"td0_alpha{alpha}_gamma{gamma}_th{threshold}.png"
        plt.savefig(filename)
        print(f"Saved plot to {filename}")

        if args.show:
            plt.show()
        plt.close()


if __name__ == '__main__':
    main()