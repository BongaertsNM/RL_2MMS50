# Blackjack RL Project

A modular framework for running reinforcement learning experiments on the Gymnasium Blackjack environment. This repository implements three classic RL agents—TD(0) prediction, Q-Learning, and SARSA—along with training loops, evaluation utilities, and easy-to-use scripts.

## Project Structure

```
RL_blackjack/
├── agents/                 # Agent implementations
│   ├── base_agent.py       # Abstract Agent class
│   ├── td0_agent.py        # TD(0) prediction agent
│   ├── q_learning_agent.py # Off-policy Q-Learning agent
│   ├── sarsa_agent.py      # On-policy SARSA agent
│   └── __init__.py
│
├── environments/           # Environment factory
│   ├── blackjack_env.py    # Gymnasium Blackjack env setup and seeding
│   └── __init__.py
│
├── train_logic/            # Core training routines
│   ├── trainer.py          # `run_episode` and `experiment`
│   └── __init__.py
│
├── utils/                  # Helpers for plotting & metrics
│   ├── plotting.py         # Learning-curve plotting
│   ├── metrics.py          # Win rates and confidence intervals
│   └── __init__.py
│
├── experiments/            # Scripts to run each agent
│   ├── run_td_experiment.py
│   ├── run_q_learning.py
│   ├── run_sarsa.py
│   └── __init__.py
│
├── evaluation_metrics.py   # Summarize saved returns arrays
├── train_main.py           # Unified CLI for all experiments
├── requirements.txt        # Python dependencies
└── README.md               # This document
```

## Installation

1. Create a Python virtual environment (e.g., `venv` or `conda`).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Unified CLI

Use `train_main.py` to run any experiment:

```bash
python train_main.py <command> [options]
```

Available commands:
- `td`       : TD(0) prediction experiment
- `qlearning`: Q-Learning control experiment
- `sarsa`    : SARSA control experiment

Each command supports optional flags to override the default hyperparameters:
- `--alpha <float>`      : Learning rate (default 0.1)
- `--gamma <float>`      : Discount factor (default 1.0)
- `--epsilon <float>`    : Exploration rate (only for Q-Learning/SARSA, default 0.1)
- `--threshold <int>`    : Stick threshold (only for TD(0), default 20)
- `--show`               : Display the learning curve interactively

Example:
```bash
python train_main.py qlearning --alpha 0.05 --gamma 0.99 --epsilon 0.05 --show
```

### Direct Scripts

You can also invoke each script directly:
```bash
python experiments/run_td_experiment.py [--alpha ALPHA] [--gamma GAMMA] [--threshold THRESHOLD] [--show]
python experiments/run_q_learning.py [--alpha ALPHA] [--gamma GAMMA] [--epsilon EPSILON] [--show]
python experiments/run_sarsa.py [--alpha ALPHA] [--gamma GAMMA] [--epsilon EPSILON] [--show]
```

## Outputs and Analysis

- Plots: Each run saves a PNG file named according to the algorithm and parameter values (e.g., `q_learning_alpha0.1_gamma1.0_eps0.1.png`).
- Return Arrays: You can modify scripts to save episode returns as `.npy` files for offline analysis.
- Summary CLI: Use `evaluation_metrics.py` to load a saved returns array and print summary statistics:
  ```bash
  python evaluation_metrics.py path/to/returns.npy
  ```

## Customization

- To change default parameters, edit the agent constructors in `agents/` or pass flags at runtime.
- For advanced logging or metrics, modify `utils/plotting.py`, `utils/metrics.py`, or `evaluation_metrics.py`.

## License

This project is licensed under the MIT License.

