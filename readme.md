# RL\_2MMS50

A collection of reinforcement‑learning agents (tabular & deep) implemented for **Blackjack** and **Atari** games. This repository provides scripts to train and evaluate various agents:

- **Tabular agents** for Blackjack: Q‑learning, SARSA, TD(0).
- **Deep agents** for Atari: DQN, Deep SARSA, Deep TD(0).

---

## 🚀 Quickstart

1. **Clone** this repository and `cd` into it:

   ```bat
   git clone <repo_url>
   cd RL_2MMS50
   ```

2. **Optional:** create and activate a virtual environment:

   ```bat
   python -m venv .venv
   .\.venv\Scripts\activate.bat
   python -m pip install --upgrade pip setuptools wheel
   ```

3. **Install core Python dependencies**:

   ```bat
   pip install -r requirements.txt
   ```

4. **Install GPU‑enabled PyTorch** (if available):

   ```bat
   pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
       torch==2.5.1+cu121 \
       torchvision==0.20.1+cu121 \
       torchaudio==2.5.1+cu121
   ```

5. **Accept Atari ROM license** (required for Gymnasium/ALE):

   ```bat
   python -m autorom --accept-license
   ```

6. **Run all training & evaluation** via batch script:

   ```bat
   .\run_all.bat
   ```

   This executes all train & evaluate modules in sequence.

---

## 📂 Directory Structure

```
RL_2MMS50/
├── agents/                # Agent implementations (tabular & deep)
│   ├── q_learning_agent.py
│   ├── sarsa_agent.py
│   ├── td0_agent.py
│   ├── dqn_agent.py
│   ├── deep_sarsa_agent.py
│   └── deep_td0_agent.py
├── configs/               # Hyperparameter configs
│   ├── grid_configs.py
│   └── atari_configs.py
├── environments/          # Gym wrappers
│   ├── blackjack_env.py
│   └── atari_env.py
├── train_logic/           # Shared training loops
│   ├── trainer.py
│   ├── dqn_trainer.py
│   ├── deep_sarsa_trainer.py
│   ├── deep_td0_trainer.py
│   └── ...
├── train/                 # `python -m train.*` entry scripts
│   ├── train_q_learning.py
│   ├── train_sarsa.py
│   ├── train_td0.py
│   ├── train_dqn.py
│   ├── train_deep_sarsa.py
│   └── train_deep_td0.py
├── evaluate/              # `python -m evaluate.*` entry scripts
│   ├── evaluate_q_learning.py
│   ├── evaluate_sarsa.py
│   ├── evaluate_td0.py
│   ├── evaluate_dqn.py
│   ├── evaluate_deep_sarsa.py
│   └── evaluate_deep_td0.py
├── models/                # Trained models by category
│   ├── models_q_learning/
│   ├── models_sarsa/
│   ├── models_td0/
│   ├── models_dqn/
│   ├── models_deep_sarsa/
│   └── models_deep_td0/
├── results/               # Plots & evaluation outputs
│   ├── results_q_learning/
│   ├── results_sarsa/
│   ├── results_td0/
│   ├── results_dqn/
│   ├── results_deep_sarsa/
│   └── results_deep_td0/
├── requirements.txt       # Python dependencies
├── run_all.bat            # Batch script to run all
└── README.md              # This file
```

---

## 📈 Training & Evaluation

- **Train** each agent:

  ```bat
  python -m train.train_<agent> [--episodes N] [--lr LR] [--gamma GAMMA] [--render]
  ```

  Models saved under `models/models_<agent>/` and plots under `results/results_<agent>/`.

- **Evaluate** each trained agent:

  ```bat
  python -m evaluate.evaluate_<agent> --model <path> [--env-id <ID>] [--trials N]
  ```

  Win rates saved as `.txt` files in `results/results_<agent>/` when run in batch mode.

Use `-h` on any script for detailed options.

---

## 📚 Acknowledgments

- **Gymnasium** & **ALE-py** for environments
- **PyTorch** for deep learning
- Course assignment & guidance by teaching staff

Happy Reinforcement Learning! 🎉

