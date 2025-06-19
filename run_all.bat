@echo off
REM ──────── TRAINING ─────────
python -m train.train_q_learning
python -m train.train_sarsa
python -m train.train_td0
python -m train.train_dqn
python -m train.train_deep_sarsa
python -m train.train_deep_td0

REM ──────── EVALUATION ───────—
python -m evaluate.evaluate_q_learning
python -m evaluate.evaluate_sarsa
python -m evaluate.evaluate_td0
python -m evaluate.evaluate_dqn
python -m evaluate.evaluate_deep_sarsa
python -m evaluate.evaluate_deep_td0

echo.
echo All jobs finished.
pause
