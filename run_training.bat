@echo off
title Train CartPole DQN Agent
echo Starting CartPole DQN Training...
call ".\.venv\Scripts\python.exe" -u train.py
pause
