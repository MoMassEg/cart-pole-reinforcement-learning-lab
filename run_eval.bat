@echo off
title Evaluate CartPole DQN Agent
call ".\.venv\Scripts\python.exe" evaluate.py cartpole_dqn.npz 10
pause
