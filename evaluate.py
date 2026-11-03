"""
Standalone evaluation script for trained CartPole DQN agent.
Runs deterministic greedy evaluation over N episodes and prints detailed metrics.
"""

import sys
import os
import numpy as np
import gymnasium as gym

from dqn_agent import DQNAgent


def evaluate(model_path: str = "cartpole_dqn.npz", num_episodes: int = 20, seed: int = 100):
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".npz"):
        print(f"Error: Model file '{model_path}' not found! Run train.py first to train the agent.")
        sys.exit(1)

    print("=" * 65)
    print("      CARTPOLE-v1 DQN AGENT EVALUATION (GREEDY POLICY)     ")
    print("=" * 65)
    print(f"Model: {model_path}")
    print(f"Number of Evaluation Episodes: {num_episodes}")
    print(f"Exploration: Disabled (Deterministic Policy, eps = 0.0)")
    print("=" * 65)

    env = gym.make("CartPole-v1")
    agent = DQNAgent(state_dim=4, action_dim=2, epsilon_start=0.0, epsilon_min=0.0)
    agent.load(model_path)

    scores = []
    for ep in range(1, num_episodes + 1):
        state, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False
        steps = 0

        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        scores.append(ep_reward)
        status = "SOLVED (MAX 500 STEPS)" if ep_reward == 500.0 else f"{ep_reward:.0f} steps"
        print(f"Episode {ep:2d}/{num_episodes}: Reward = {ep_reward:5.1f} / 500.0 | Status: {status}")

    env.close()

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    perfect_count = sum(s == 500.0 for s in scores)
    success_rate = (perfect_count / num_episodes) * 100.0

    print("=" * 65)
    print("EVALUATION SUMMARY:")
    print(f"  - Mean Reward   : {mean_score:.2f} +/- {std_score:.2f}")
    print(f"  - Min / Max     : {min_score:.1f} / {max_score:.1f}")
    print(f"  - Perfect (500) : {perfect_count} / {num_episodes} episodes")
    print(f"  - Success Rate  : {success_rate:.1f}%")
    print("=" * 65)

    return {
        "mean": mean_score,
        "std": std_score,
        "success_rate": success_rate,
        "scores": scores
    }


if __name__ == "__main__":
    path = "cartpole_dqn.npz"
    episodes = 10
    if len(sys.argv) > 1:
        path = sys.argv[1]
    if len(sys.argv) > 2:
        episodes = int(sys.argv[2])
    evaluate(path, num_episodes=episodes)
