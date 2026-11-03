import sys
import os
import math
import time
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

# Ensure root dir is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import gymnasium as gym
except ImportError:
    gym = None

from dqn_agent import DQNAgent


class CartPoleApp:
    def __init__(self, root: tk.Tk, model_path: str = "cartpole_dqn.npz"):
        self.root = root
        self.root.title("CartPole-v1 Deep Q-Network (DQN) Interactive Showcase")
        self.root.geometry("980x740")
        self.root.minsize(880, 680)
        self.root.configure(bg="#1e1e2e")

        self.model_path = model_path
        if not os.path.exists(self.model_path) and os.path.exists("cartpole_dqn.npz"):
            self.model_path = "cartpole_dqn.npz"
        self.env = None
        self.agent = None
        self.state = None
        self.episode = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.is_running = False
        self.manual_mode = False
        self.manual_action = 0
        self.fps = 50
        self.noise_prob = 0.0
        self.last_q_values = np.array([0.0, 0.0])
        self.last_action = 0

        # Try initializing environment
        if gym is not None:
            self.env = gym.make("CartPole-v1")
        else:
            messagebox.showerror("Error", "Gymnasium is not installed in the current environment.")

        # Try loading agent
        self.init_agent()

        # Build GUI
        self._build_ui()

        # Reset environment
        self.reset_env()

        # Key bindings
        self.root.bind("<Left>", lambda e: self.on_manual_push(0))
        self.root.bind("<Right>", lambda e: self.on_manual_push(1))
        self.root.bind("<space>", lambda e: self.toggle_run())

    def init_agent(self):
        """Initialize the DQN agent and load trained weights if available."""
        self.agent = DQNAgent(
            state_dim=4,
            action_dim=2,
            epsilon_start=0.0,
            epsilon_min=0.0
        )
        if os.path.exists(self.model_path):
            try:
                self.agent.load(self.model_path)
                self.model_status_text = f" Model Loaded: {os.path.basename(self.model_path)}"
                self.model_status_color = "#27ae60"
            except Exception as e:
                self.model_status_text = f" Failed loading model: {e}"
                self.model_status_color = "#e74c3c"
        else:
            self.model_status_text = f" Model not found ({self.model_path}). Train model first!"
            self.model_status_color = "#f39c12"

    def _build_ui(self):
        # 1. Top Header Frame
        header = tk.Frame(self.root, bg="#181825", pady=10, padx=20)
        header.pack(fill=tk.X)

        title_lbl = tk.Label(
            header,
            text="CARTPOLE-v1 DEEP Q-NETWORK (DQN)",
            font=("Segoe UI", 16, "bold"),
            fg="#cdd6f4",
            bg="#181825"
        )
        title_lbl.pack(side=tk.LEFT)

        self.lbl_status = tk.Label(
            header,
            text=self.model_status_text,
            font=("Segoe UI", 10, "bold"),
            fg=self.model_status_color,
            bg="#181825"
        )
        self.lbl_status.pack(side=tk.RIGHT)

        # 2. Main Content (Canvas on Left, Telemetry & Controls on Right)
        content = tk.Frame(self.root, bg="#1e1e2e", padx=15, pady=10)
        content.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(content, bg="#1e1e2e")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Canvas for CartPole rendering
        self.canvas_w = 580
        self.canvas_h = 360
        self.canvas = tk.Canvas(
            left_frame,
            width=self.canvas_w,
            height=self.canvas_h,
            bg="#11111b",
            highlightthickness=1,
            highlightbackground="#45475a"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom info under canvas
        self.lbl_action_banner = tk.Label(
            left_frame,
            text="Current Action: NONE",
            font=("Segoe UI", 11, "bold"),
            fg="#a6adc8",
            bg="#181825",
            pady=6
        )
        self.lbl_action_banner.pack(fill=tk.X, pady=(8, 0))

        # Right frame (Telemetry & Controls)
        right_frame = tk.Frame(content, bg="#181825", width=340, padx=15, pady=15)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_frame.pack_propagate(False)

        # Telemetry Group
        tk.Label(
            right_frame,
            text="REAL-TIME TELEMETRY",
            font=("Segoe UI", 11, "bold"),
            fg="#89b4fa",
            bg="#181825"
        ).pack(anchor="w")

        telemetry_box = tk.Frame(right_frame, bg="#11111b", padx=10, pady=10, highlightthickness=1, highlightbackground="#313244")
        telemetry_box.pack(fill=tk.X, pady=(5, 12))

        self.lbl_reward = tk.Label(telemetry_box, text="Reward: 0.0 / 500", font=("Segoe UI", 13, "bold"), fg="#a6e3a1", bg="#11111b")
        self.lbl_reward.pack(anchor="w")

        self.lbl_step = tk.Label(telemetry_box, text="Step: 0", font=("Segoe UI", 10), fg="#cdd6f4", bg="#11111b")
        self.lbl_step.pack(anchor="w")

        self.lbl_pos = tk.Label(telemetry_box, text="Cart Pos (x): 0.00 m", font=("Segoe UI", 10), fg="#bac2de", bg="#11111b")
        self.lbl_pos.pack(anchor="w")

        self.lbl_vel = tk.Label(telemetry_box, text="Cart Vel (v): 0.00 m/s", font=("Segoe UI", 10), fg="#bac2de", bg="#11111b")
        self.lbl_vel.pack(anchor="w")

        self.lbl_angle = tk.Label(telemetry_box, text="Pole Angle (θ): 0.00°", font=("Segoe UI", 10), fg="#bac2de", bg="#11111b")
        self.lbl_angle.pack(anchor="w")

        self.lbl_ang_vel = tk.Label(telemetry_box, text="Pole Ang Vel (ω): 0.00 rad/s", font=("Segoe UI", 10), fg="#bac2de", bg="#11111b")
        self.lbl_ang_vel.pack(anchor="w")

        # Q-Values Display
        tk.Label(
            right_frame,
            text="Q-VALUES & DECISION",
            font=("Segoe UI", 11, "bold"),
            fg="#f9e2af",
            bg="#181825"
        ).pack(anchor="w")

        q_box = tk.Frame(right_frame, bg="#11111b", padx=10, pady=10, highlightthickness=1, highlightbackground="#313244")
        q_box.pack(fill=tk.X, pady=(5, 12))

        self.lbl_q_left = tk.Label(q_box, text="Q(s, PUSH_LEFT):  --", font=("Consolas", 10), fg="#cdd6f4", bg="#11111b")
        self.lbl_q_left.pack(anchor="w")

        self.lbl_q_right = tk.Label(q_box, text="Q(s, PUSH_RIGHT): --", font=("Consolas", 10), fg="#cdd6f4", bg="#11111b")
        self.lbl_q_right.pack(anchor="w")

        self.lbl_q_diff = tk.Label(q_box, text="Advantage: --", font=("Consolas", 9), fg="#94e2d5", bg="#11111b")
        self.lbl_q_diff.pack(anchor="w")

        # Controls Group
        tk.Label(
            right_frame,
            text="CONTROLS & PERTURBATION",
            font=("Segoe UI", 11, "bold"),
            fg="#cba6f7",
            bg="#181825"
        ).pack(anchor="w")

        btn_box = tk.Frame(right_frame, bg="#181825")
        btn_box.pack(fill=tk.X, pady=(6, 10))

        self.btn_run = tk.Button(
            btn_box,
            text="▶ Play",
            font=("Segoe UI", 10, "bold"),
            bg="#a6e3a1",
            fg="#11111b",
            activebackground="#94e2d5",
            width=8,
            command=self.toggle_run
        )
        self.btn_run.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_step = tk.Button(
            btn_box,
            text="⏭ Step",
            font=("Segoe UI", 10),
            bg="#89b4fa",
            fg="#11111b",
            width=8,
            command=self.step_once
        )
        self.btn_step.pack(side=tk.LEFT, padx=5)

        self.btn_reset = tk.Button(
            btn_box,
            text="↺ Reset",
            font=("Segoe UI", 10),
            bg="#fab387",
            fg="#11111b",
            width=8,
            command=self.reset_env
        )
        self.btn_reset.pack(side=tk.LEFT, padx=5)

        # Simulation Speed Slider
        speed_box = tk.Frame(right_frame, bg="#181825")
        speed_box.pack(fill=tk.X, pady=4)
        tk.Label(speed_box, text="Speed (FPS):", font=("Segoe UI", 9), fg="#bac2de", bg="#181825").pack(side=tk.LEFT)
        self.speed_slider = tk.Scale(
            speed_box,
            from_=10,
            to=120,
            orient=tk.HORIZONTAL,
            bg="#181825",
            fg="#cdd6f4",
            highlightthickness=0,
            troughcolor="#313244",
            command=self.on_speed_change
        )
        self.speed_slider.set(self.fps)
        self.speed_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))

        # Perturbation / Noise Slider
        noise_box = tk.Frame(right_frame, bg="#181825")
        noise_box.pack(fill=tk.X, pady=4)
        tk.Label(noise_box, text="Noise / ε:", font=("Segoe UI", 9), fg="#bac2de", bg="#181825").pack(side=tk.LEFT)
        self.noise_slider = tk.Scale(
            noise_box,
            from_=0.0,
            to=0.5,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            bg="#181825",
            fg="#cdd6f4",
            highlightthickness=0,
            troughcolor="#313244",
            command=self.on_noise_change
        )
        self.noise_slider.set(0.0)
        self.noise_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))

        # Push / Impulse Buttons
        impulse_box = tk.Frame(right_frame, bg="#181825")
        impulse_box.pack(fill=tk.X, pady=(6, 8))
        btn_push_l = tk.Button(
            impulse_box,
            text="◀ Push Left",
            font=("Segoe UI", 9),
            bg="#45475a",
            fg="#cdd6f4",
            command=lambda: self.apply_impulse(-0.15)
        )
        btn_push_l.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        btn_push_r = tk.Button(
            impulse_box,
            text="Push Right ▶",
            font=("Segoe UI", 9),
            bg="#45475a",
            fg="#cdd6f4",
            command=lambda: self.apply_impulse(0.15)
        )
        btn_push_r.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # View Training Plot Button
        btn_plot = tk.Button(
            right_frame,
            text="📈 View Training Reward Plot",
            font=("Segoe UI", 10),
            bg="#313244",
            fg="#89b4fa",
            relief=tk.FLAT,
            command=self.show_reward_plot
        )
        btn_plot.pack(fill=tk.X, pady=(8, 0))

    def on_speed_change(self, val):
        self.fps = int(val)

    def on_noise_change(self, val):
        self.noise_prob = float(val)

    def reset_env(self):
        """Reset environment and state."""
        if self.env is None:
            return
        self.state, _ = self.env.reset()
        self.step_count = 0
        self.total_reward = 0.0
        self.last_action = 0
        self.last_q_values = np.array([0.0, 0.0])
        self.update_telemetry()
        self.render_cartpole()

    def apply_impulse(self, delta_omega: float):
        """Perturb the pole angular velocity to test agent recovery."""
        if self.state is not None and self.env is not None:
            # Modify environment state (CartPole env.unwrapped.state is [x, x_dot, theta, theta_dot])
            curr = list(self.env.unwrapped.state)
            curr[3] += delta_omega
            self.env.unwrapped.state = np.array(curr, dtype=np.float32)
            self.state = np.array(curr, dtype=np.float32)
            self.update_telemetry()
            self.render_cartpole()

    def toggle_run(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.btn_run.config(text="⏸ Pause", bg="#f38ba8")
            self.run_loop()
        else:
            self.btn_run.config(text="▶ Play", bg="#a6e3a1")

    def run_loop(self):
        if not self.is_running:
            return

        done = self.step_simulation()
        if done:
            self.is_running = False
            self.btn_run.config(text="▶ Play Again", bg="#a6e3a1")
            return

        delay = max(1, int(1000 / self.fps))
        self.root.after(delay, self.run_loop)

    def step_once(self):
        if not self.is_running:
            self.step_simulation()

    def step_simulation(self) -> bool:
        if self.env is None or self.state is None:
            return True

        # Get Q values
        self.last_q_values = self.agent.get_q_values(self.state)

        # Decide action (Agent or random noise)
        if np.random.rand() < self.noise_prob:
            action = np.random.choice([0, 1])
        else:
            action = int(np.argmax(self.last_q_values))

        self.last_action = action
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        self.state = next_state
        self.step_count += 1
        self.total_reward += reward
        done = terminated or truncated

        self.update_telemetry(terminated, truncated)
        self.render_cartpole()

        return done

    def update_telemetry(self, terminated: bool = False, truncated: bool = False):
        if self.state is None:
            return

        x, x_dot, theta, theta_dot = self.state
        theta_deg = math.degrees(theta)

        # Update labels
        self.lbl_reward.config(text=f"Reward: {self.total_reward:.1f} / 500")
        self.lbl_step.config(text=f"Step: {self.step_count}")
        self.lbl_pos.config(text=f"Cart Pos (x): {x:+.3f} m")
        self.lbl_vel.config(text=f"Cart Vel (v): {x_dot:+.3f} m/s")
        self.lbl_angle.config(text=f"Pole Angle (θ): {theta_deg:+.2f}° ({theta:+.3f} rad)")
        self.lbl_ang_vel.config(text=f"Pole Ang Vel (ω): {theta_dot:+.3f} rad/s")

        # Q values
        q0 = self.last_q_values[0]
        q1 = self.last_q_values[1]
        best_act = "PUSH_LEFT" if q0 >= q1 else "PUSH_RIGHT"
        action_name = "PUSH_LEFT (0)" if self.last_action == 0 else "PUSH_RIGHT (1)"

        self.lbl_q_left.config(
            text=f"Q(s, PUSH_LEFT):  {q0:+.3f} {'◄ SELECTED' if best_act == 'PUSH_LEFT' else ''}",
            fg="#a6e3a1" if best_act == "PUSH_LEFT" else "#cdd6f4"
        )
        self.lbl_q_right.config(
            text=f"Q(s, PUSH_RIGHT): {q1:+.3f} {'◄ SELECTED' if best_act == 'PUSH_RIGHT' else ''}",
            fg="#a6e3a1" if best_act == "PUSH_RIGHT" else "#cdd6f4"
        )
        self.lbl_q_diff.config(text=f"Advantage ΔQ: {abs(q0 - q1):.4f}")

        if terminated:
            self.lbl_action_banner.config(text=f" TERMINATED: Pole Fell / Boundary Reached! (Final Reward: {self.total_reward:.0f})", fg="#f38ba8")
        elif truncated:
            self.lbl_action_banner.config(text=" GOAL REACHED! Solved Full 500 Steps!", fg="#a6e3a1")
        else:
            self.lbl_action_banner.config(text=f"Action: {action_name}", fg="#89b4fa")

    def render_cartpole(self):
        """Draw smooth physics rendering on the canvas."""
        self.canvas.delete("all")
        if self.state is None:
            return

        w = self.canvas.winfo_width() or self.canvas_w
        h = self.canvas.winfo_height() or self.canvas_h
        x, _, theta, _ = self.state

        # World coordinates to screen coordinates
        # CartPole x limit is ~2.4, y axis points down in screen coords
        world_width = 4.8  # -2.4 to 2.4
        scale = w / world_width
        cart_y = h * 0.72

        # Screen center
        screen_x = (x + 2.4) * scale

        # 1. Draw track line
        self.canvas.create_line(0, cart_y + 18, w, cart_y + 18, fill="#585b70", width=3)
        # Boundary markers (-2.4 and +2.4 limits)
        self.canvas.create_line(10, cart_y - 20, 10, cart_y + 30, fill="#f38ba8", width=2, dash=(4, 4))
        self.canvas.create_line(w - 10, cart_y - 20, w - 10, cart_y + 30, fill="#f38ba8", width=2, dash=(4, 4))
        self.canvas.create_text(25, cart_y + 35, text="-2.4m", fill="#6c7086", font=("Segoe UI", 9))
        self.canvas.create_text(w - 25, cart_y + 35, text="+2.4m", fill="#6c7086", font=("Segoe UI", 9))
        self.canvas.create_text(w / 2, cart_y + 35, text="0.0m (Center)", fill="#6c7086", font=("Segoe UI", 9))

        # 2. Draw Cart
        cart_w = 70
        cart_h = 36
        cx1 = screen_x - cart_w / 2
        cy1 = cart_y - cart_h / 2
        cx2 = screen_x + cart_w / 2
        cy2 = cart_y + cart_h / 2

        self.canvas.create_rectangle(cx1, cy1, cx2, cy2, fill="#89b4fa", outline="#b4befe", width=2)

        # Wheels
        wheel_r = 7
        self.canvas.create_oval(cx1 + 8, cy2 - 2, cx1 + 8 + 2 * wheel_r, cy2 - 2 + 2 * wheel_r, fill="#313244", outline="#cdd6f4")
        self.canvas.create_oval(cx2 - 8 - 2 * wheel_r, cy2 - 2, cx2 - 8, cy2 - 2 + 2 * wheel_r, fill="#313244", outline="#cdd6f4")

        # 3. Draw Pole
        pole_len = 130
        pole_w = 9
        # In CartPole, theta=0 is straight up, positive theta is clockwise (tilted right)
        # Screen angle: up is -pi/2, so angle on screen is -pi/2 + theta
        pole_end_x = screen_x + pole_len * math.sin(theta)
        pole_end_y = cart_y - pole_len * math.cos(theta)

        # Draw pole line/rectangle
        self.canvas.create_line(screen_x, cart_y, pole_end_x, pole_end_y, fill="#fab387", width=pole_w, capstyle=tk.ROUND)

        # Hinge pin on cart
        self.canvas.create_oval(screen_x - 5, cart_y - 5, screen_x + 5, cart_y + 5, fill="#f38ba8", outline="#ffffff", width=1)
        # Tip circle
        self.canvas.create_oval(pole_end_x - 6, pole_end_y - 6, pole_end_x + 6, pole_end_y + 6, fill="#f9e2af", outline="#ffffff", width=1)

        # 4. Action arrow indicator
        if self.last_action == 0:  # Push Left
            arrow_x = cx1 - 25
            self.canvas.create_line(cx1, cart_y, arrow_x, cart_y, arrow=tk.LAST, fill="#a6e3a1", width=5)
        else:  # Push Right
            arrow_x = cx2 + 25
            self.canvas.create_line(cx2, cart_y, arrow_x, cart_y, arrow=tk.LAST, fill="#a6e3a1", width=5)

    def on_manual_push(self, action: int):
        """Allow manual push with arrow keys."""
        if self.env is not None and self.state is not None:
            self.last_action = action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.state = next_state
            self.step_count += 1
            self.total_reward += reward
            self.last_q_values = self.agent.get_q_values(self.state)
            self.update_telemetry(terminated, truncated)
            self.render_cartpole()

    def show_reward_plot(self):
        """Display reward plot if it exists."""
        plot_path = "reward_plot.png"
        if not os.path.exists(plot_path):
            messagebox.showinfo("Plot Not Found", f"Training plot '{plot_path}' has not been generated yet. Run train.py first.")
            return

        # Open in default system image viewer or a top level window
        try:
            import os
            if os.name == 'nt':
                os.startfile(plot_path)
            else:
                top = tk.Toplevel(self.root)
                top.title("DQN CartPole Reward Plot")
                img = tk.PhotoImage(file=plot_path)
                lbl = tk.Label(top, image=img)
                lbl.image = img
                lbl.pack()
        except Exception as e:
            messagebox.showerror("Error", f"Could not display plot: {e}")


def launch_gui(model_path: str = "cartpole_dqn.npz"):
    root = tk.Tk()
    app = CartPoleApp(root, model_path=model_path)
    root.mainloop()


if __name__ == "__main__":
    model = "cartpole_dqn.npz"
    if len(sys.argv) > 1:
        model = sys.argv[1]
    launch_gui(model)
