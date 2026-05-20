"""
Environment-agnostic DQN training, evaluation, and plotting utilities.
"""
import os
import numpy as np
import torch

from src.agent_factory import DQNAgent, get_device


def train(env, agent=None, episodes=800, verbose=True, model_dir="models"):
    """Generic DQN training loop. Returns (agent, reward_history, loss_history)."""
    if agent is None:
        agent = DQNAgent(env.get_state_size(), env.get_action_size())

    reward_history = []
    loss_history = []
    best_reward = float("-inf")

    if verbose:
        print(f"\n{'Episode':>6}  {'Reward':>10}  {'Avg(100)':>10}  {'Epsilon':>8}  {'Loss':>10}")
        print("-" * 60)

    for ep in range(1, episodes + 1):
        env.reset()
        state = env.get_state()
        total_reward = 0.0
        losses = []

        while not env.is_done():
            action = agent.act(state)
            reward = env.step(action)
            next_state = env.get_state()
            done = env.is_done()

            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss > 0:
                losses.append(loss)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        if ep % agent.target_update == 0:
            agent._sync_target()

        reward_history.append(total_reward)
        if losses:
            loss_history.append(np.mean(losses))

        if total_reward > best_reward:
            best_reward = total_reward
            os.makedirs(model_dir, exist_ok=True)
            torch.save(agent.q_network.state_dict(), os.path.join(model_dir, "model_best.pth"))

        if verbose and ep % 100 == 0:
            avg = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)
            avg_loss = np.mean(loss_history[-50:]) if loss_history else 0
            print(f"{ep:>6}  {total_reward:>10.2f}  {avg:>10.2f}  {agent.epsilon:>8.4f}  {avg_loss:>10.6f}")

    if verbose:
        avg = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)
        print(f"\nTraining complete. Best reward: {best_reward:.2f}, Final avg(100): {avg:.2f}")

    return agent, reward_history, loss_history


def evaluate_function_fit(env, agent, n_samples=1000):
    """
    Evaluate how well the agent has learned the underlying function.
    Returns (x_vals, y_true, y_pred, mae).
    """
    xs = np.linspace(0, 1, n_samples, dtype=np.float32)
    device = get_device()

    # Get agent predictions (greedy action → y value)
    states = torch.FloatTensor(xs).unsqueeze(1).to(device)
    q_vals = agent.q_values(states)
    best_actions = q_vals.argmax(dim=1).cpu().numpy()

    y_min = env.get_y_min()
    y_max = env.get_y_max()
    n_actions = env.get_action_size()
    bin_width = (y_max - y_min) / (n_actions - 1)
    y_pred = y_min + best_actions * bin_width

    # Get ground truth from Python function registry
    from src.piecewise_functions import PY_FUNCTIONS
    import function_env

    # Find which function the env is using by checking each one
    y_true = None
    func_name = env.get_function_name()
    for fid, py_func in PY_FUNCTIONS.items():
        e = function_env.FunctionFittingEnv(fid, n_actions=2, episode_length=1)
        if e.get_function_name() == func_name:
            y_true = np.array([py_func(float(x)) for x in xs])
            break

    if y_true is None:
        y_true = np.zeros_like(xs)

    mae = np.mean(np.abs(y_true - y_pred))
    return xs, y_true, y_pred, mae


def plot_progress(rewards, losses, save_path="training_progress.png"):
    """Plot training reward and loss curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(rewards, alpha=0.3, color="tab:blue", label="per episode")
        if len(rewards) >= 100:
            smooth = np.convolve(rewards, np.ones(100) / 100, mode="valid")
            ax1.plot(np.arange(len(smooth)) + 99, smooth, color="tab:red", label="100-ep moving avg")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.set_title("Training Rewards")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if losses:
            ax2.plot(losses, color="tab:green")
            ax2.set_xlabel("Training step (episodes with loss)")
            ax2.set_ylabel("MSE Loss")
            ax2.set_title("Training Loss")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    except ImportError:
        pass


def plot_function_fit(xs, y_true, y_pred, mae, func_name, save_path):
    """Plot true function vs agent predictions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(xs, y_true, color="black", linewidth=2, label="True function")
        ax.scatter(xs[::20], y_pred[::20], color="tab:red", s=8, alpha=0.6, label="Agent prediction")
        ax.fill_between(xs, y_true, y_pred, alpha=0.15, color="tab:red")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{func_name}  (MAE = {mae:.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    except ImportError:
        pass
