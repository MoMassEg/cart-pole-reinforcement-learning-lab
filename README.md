# CartPole-v1 Reinforcement Learning with Deep Q-Network (DQN)

A complete, self-contained implementation of the **Deep Q-Network (DQN)** algorithm from first principles to solve OpenAI / Farama **Gymnasium CartPole-v1**.

The algorithm, neural network backpropagation, Huber loss, Adam optimizer, experience replay, and double Q-learning target computations are implemented from scratch without relying on external deep learning frameworks.

---

## 🚀 Key Features

1. **Algorithm Implementation from Scratch**:
   - **Multi-Layer Perceptron ($Q(s, a)$)**: He/Kaiming initialization with 2 hidden layers (128 $\times$ 128 neurons, ReLU activations).
   - **Double DQN (DDQN)**: Decoupled action selection ($a^* = \arg\max_a Q(s', a; \theta)$) from value evaluation ($Q(s', a^*; \theta^-)$) to eliminate maximization bias.
   - **Experience Replay Memory**: Circular transition buffer with uniform mini-batch sampling.
   - **Analytical Backpropagation & Adam Optimizer**: Exact analytical gradients with Adam momentum, RMSProp variance scaling, and bias correction.
   - **Huber Loss Gradient Clipping**: Robust loss handling that switches from quadratic to linear error when $| \delta | > 1.0$.

2. **Benchmarking & Evaluation**:
   - CartPole-v1 max limit is **500 steps**.
   - Solved threshold benchmark: Average reward $\ge 475.0$ over test episodes.
   - **Achieved Performance**: **500.0 / 500.0 (100% Perfect Success Rate)** on greedy evaluation.
   - Generates a reward-versus-episodes plot (`reward_plot.png`) with raw rewards, 20-episode moving average, greedy evaluation markers, and solved threshold.

3. **Interactive Graphical User Interface (UI)**:
   - Built with native desktop `tkinter` for high performance and responsiveness (60 FPS).
   - Real-time physics canvas showing the cart, wheels, pole, pivot hinge, rail boundaries ($\pm 2.4\text{ m}$), and dynamic action force arrows.
   - Real-time telemetry dashboard:
     - Step counter and reward progress (`500.0 / 500`).
     - State vector: Cart position ($x$), velocity ($v$), pole angle ($\theta$ in degrees and radians), and angular velocity ($\omega$).
     - Real-time Q-values: $Q(s, \text{PUSH\_LEFT})$ and $Q(s, \text{PUSH\_RIGHT})$ with visual advantage indicator.
   - Controls:
     - `Play / Pause` toggle (Spacebar shortcut).
     - `Step Once` single-frame inspection mode.
     - `Speed Slider` (10 FPS to 120 FPS).
     - `Perturbation / Exploration Slider` ($\epsilon$ noise injection).
     - `Push Left` and `Push Right` impulse buttons: injects external disturbances to test the agent's balancing recovery.
     - `View Training Plot`: opens `reward_plot.png`.

---

## 📁 Repository Structure

```
deep rainforcement/
│── dqn_agent.py              # DQNAgent, QNetwork, AdamOptimizer, ReplayBuffer
│── train.py                  # Training pipeline, checkpointing & plot generation
│── evaluate.py               # Standalone deterministic evaluation script
│── ui.py                     # Interactive Tkinter GUI showcase
│── cartpole_dqn.npz          # Trained model weights (100% success rate, 500/500)
│── reward_plot.png           # High-resolution reward-vs-episodes plot
│── training_history.json     # Saved training metrics and episode logs
│── launch_ui.bat             # 1-click launcher for the interactive UI
│── run_training.bat          # 1-click launcher to re-train the agent
│── run_eval.bat              # 1-click launcher to evaluate the agent
│── requirements.txt          # Python dependencies (gymnasium, matplotlib, numpy)
└── README.md                 # Project documentation
```

---

## 🏃 Quick Start

### 1. Launch the Interactive GUI
Run the GUI to watch the trained agent play CartPole in real-time:

```powershell
.\launch_ui.bat
```
or via Python:
```powershell
.\.venv\Scripts\python.exe ui.py
```

### 2. Run Evaluation
Evaluate the trained agent across 10 deterministic test episodes:

```powershell
.\run_eval.bat
```
or via Python:
```powershell
.\.venv\Scripts\python.exe evaluate.py cartpole_dqn.npz 10
```

### 3. Re-Train the Agent
To re-train the model from scratch and generate new plots:

```powershell
.\run_training.bat
```
or via Python:
```powershell
.\.venv\Scripts\python.exe train.py
```

---

## 📊 Training Results

- **Environment**: Farama Gymnasium `CartPole-v1`
- **Solved at Episode**: **380**
- **Evaluation Score**: **500.0 / 500.0**
- **Deterministic Test Success Rate**: **100.0%**
- **Plot**: See `reward_plot.png` for the training curve.
