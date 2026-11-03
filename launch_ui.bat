@echo off
title CartPole DQN Interactive UI
echo Launching CartPole DQN Interactive UI...
call ".\.venv\Scripts\python.exe" ui.py cartpole_dqn.npz
if errorlevel 1 (
    echo.
    echo Execution failed. Please check Python installation.
    pause
)
