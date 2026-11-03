import os
import sys
import json
import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from dqn_agent import DQNAgent


def evaluate_policy(env, agent, num_episodes: int = 10, seed_base: int = 8000):
    """Run deterministic evaluation episodes without exploration."""
    eval_rewards = []
    for i in range(num_episodes):
        state, _ = env.reset(seed=seed_base + i)
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        eval_rewards.append(ep_reward)
    return float(np.mean(eval_rewards)), float(np.std(eval_rewards)), eval_rewards


def train_dqn(
    max_episodes: int = 400,
    target_eval_reward: float = 475.0,
    save_path: str = "cartpole_dqn.npz",
    plot_path: str = "reward_plot.png",
    history_path: str = "training_history.json"
):
    """
    Train DQN on Gymnasium CartPole-v1.
    CartPole-v1 max steps: 500. Solved benchmark: Average reward >= 475.
    """
    print("=" * 68)
    print("      DEEP Q-NETWORK (DQN) TRAINING ON GYMNASIUM CARTPOLE-v1     ")
    print("=" * 68)
    print(f"Goal: Evaluation Average Reward >= {target_eval_reward:.1f} / 500.0")
    print(f"Max Training Episodes: {max_episodes}")
    print(f"Algorithm: Double DQN with Analytical Backprop & Adam")
    print("=" * 68)

    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    agent = DQNAgent(
        state_dim=4,
        action_dim=2,
        hidden_dim=128,
        lr=7e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        target_update_freq=200,
        buffer_capacity=50000,
        batch_size=64
    )

    rewards_history = []
    moving_avg_history = []
    eval_history = []
    eval_episodes = []
    epsilon_history = []
    recent_rewards = deque(maxlen=20)
    best_eval_reward = 0.0
    solved_at_episode = None

    start_time = time.time()

    for ep in range(1, max_episodes + 1):
        state, _ = env.reset(seed=ep)
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Bellman backup: only mask out future bootstrap on failure (terminated), not truncation
            done_mask = terminated and not truncated

            agent.step(state, action, reward, next_state, done_mask)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        recent_rewards.append(total_reward)
        moving_avg = float(np.mean(recent_rewards))
        moving_avg_history.append(moving_avg)
        epsilon_history.append(agent.epsilon)

        # Periodic evaluation & logging
        if ep % 20 == 0 or ep == 1:
            mean_eval, std_eval, _ = evaluate_policy(eval_env, agent, num_episodes=10, seed_base=8000 + ep)
            eval_history.append(mean_eval)
            eval_episodes.append(ep)

            # Checkpoint best model
            if mean_eval > best_eval_reward:
                best_eval_reward = mean_eval
                agent.save(save_path)

            print(
                f"Ep {ep:3d}/{max_episodes} | "
                f"Train R: {total_reward:5.1f} | "
                f"20-Ep MA: {moving_avg:5.1f} | "
                f"Eval (10 ep): {mean_eval:5.1f} +/- {std_eval:4.1f} (Best: {best_eval_reward:5.1f}) | "
                f"Eps: {agent.epsilon:.3f}",
                flush=True
            )

            # Check solved threshold
            if mean_eval >= target_eval_reward and ep >= 100:
                print(f"\n[SOLVED] CartPole-v1 at Episode {ep}! Evaluation Score: {mean_eval:.1f} / 500.0", flush=True)
                solved_at_episode = ep
                agent.save(save_path)
                break

    env.close()
    eval_env.close()
    elapsed = time.time() - start_time

    # Ensure best model is saved
    if not os.path.exists(save_path) or best_eval_reward >= target_eval_reward:
        agent.save(save_path)

    print("=" * 68)
    print(f"Training completed in {elapsed:.2f} seconds.")
    print(f"Best model saved to '{save_path}' (Best Eval Score: {best_eval_reward:.1f}).")

    # Save training history JSON
    history = {
        "max_episodes": max_episodes,
        "completed_episodes": len(rewards_history),
        "rewards": rewards_history,
        "moving_avg_20": moving_avg_history,
        "eval_episodes": eval_episodes,
        "eval_scores": eval_history,
        "epsilon_history": epsilon_history,
        "best_eval_reward": best_eval_reward,
        "solved_at_episode": solved_at_episode,
        "elapsed_seconds": elapsed
    }
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to '{history_path}'.")

    # Generate and save reward-versus-episodes plot
    plot_training_results(
        rewards_history,
        moving_avg_history,
        eval_episodes,
        eval_history,
        target_eval_reward,
        plot_path
    )

    # Final evaluation run of best saved model
    print("\n" + "=" * 68)
    print("FINAL EVALUATION OF BEST SAVED MODEL (Deterministic Greedy Policy)")
    print("=" * 68)
    final_eval = final_evaluation(save_path, num_episodes=10)

    return agent, history, final_eval


def plot_training_results(
    rewards: list,
    moving_avg: list,
    eval_episodes: list,
    eval_scores: list,
    target_reward: float,
    save_path: str = "reward_plot.png"
):
    """Generate high-quality reward-versus-episodes plot."""
    fig, ax = plt.subplots(figsize=(11, 6), dpi=200)

    episodes = range(1, len(rewards) + 1)

    # Shaded raw episode rewards
    ax.plot(episodes, rewards, label="Training Episode Reward", color="#3498db", alpha=0.35, linewidth=1.2)
    ax.fill_between(episodes, rewards, alpha=0.08, color="#3498db")

    # 20-episode moving average
    ax.plot(episodes, moving_avg, label="20-Episode Moving Average", color="#e74c3c", linewidth=2.4)

    # Evaluation markers
    if eval_episodes and eval_scores:
        ax.plot(
            eval_episodes,
            eval_scores,
            label="Greedy Eval Score (10 eps)",
            color="#9b59b6",
            marker="o",
            markersize=6,
            linewidth=2.0,
            linestyle="-."
        )

    # Solved benchmark line (475/500)
    ax.axhline(
        y=target_reward,
        color="#2ecc71",
        linestyle="--",
        linewidth=2.0,
        label=f"Solved Benchmark ({target_reward:.0f} pts)"
    )

    ax.set_title("CartPole-v1 DQN Training: Reward vs. Episodes", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Episode Number", fontsize=12, fontweight="medium")
    ax.set_ylabel("Total Reward (Timesteps Balanced)", fontsize=12, fontweight="medium")
    ax.set_ylim(-10, 520)
    ax.set_xlim(1, max(len(rewards), 10))
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#bdc3c7", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Reward plot saved to '{save_path}'.")


def final_evaluation(model_path: str = "cartpole_dqn.npz", num_episodes: int = 10, seed: int = 4200):
    """Evaluate best saved checkpoint over test episodes."""
    env = gym.make("CartPole-v1")
    agent = DQNAgent(state_dim=4, action_dim=2, epsilon_start=0.0, epsilon_min=0.0)
    agent.load(model_path)

    scores = []
    print(f"Testing {num_episodes} episodes using deterministic greedy policy (eps = 0.0):")
    print("-" * 55)

    for i in range(num_episodes):
        state, _ = env.reset(seed=seed + i)
        total_r = 0.0
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, term, trunc, _ = env.step(action)
            total_r += reward
            done = term or trunc
        scores.append(total_r)
        status = "SOLVED (MAX 500)" if total_r == 500.0 else f"{total_r:.0f} steps"
        print(f"  Test Episode {i+1:2d}: Reward = {total_r:5.1f} / 500.0 | Status: {status}")

    env.close()
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    success_rate = (sum(s == 500.0 for s in scores) / num_episodes) * 100.0

    print("-" * 55)
    print(f"Summary: Mean Score = {mean_score:.2f} +/- {std_score:.2f}")
    print(f"Success Rate (500/500 Max Reward): {success_rate:.1f}%")
    print("=" * 68)

    return {
        "scores": scores,
        "mean_score": mean_score,
        "std_score": std_score,
        "success_rate": success_rate
    }


if __name__ == "__main__":
    train_dqn()
