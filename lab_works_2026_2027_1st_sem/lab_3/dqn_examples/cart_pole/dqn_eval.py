import os
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np


# Neural network model for approximating Q-values (matching dqn_training.py)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def load_model(model_path=None):
    if model_path is None:
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        model_path = os.path.join(models_dir, "dqn_cartpole_torch.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"PyTorch model file not found at '{model_path}'. "
            "Please run dqn_training.py first to train and save the model."
        )

    # Temporary environment to extract observation and action space dimensions
    temp_env = gym.make("CartPole-v1")
    input_dim = temp_env.observation_space.shape[0]
    output_dim = temp_env.action_space.n
    temp_env.close()

    model = DQN(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print(f"Loaded PyTorch model from: {model_path}")
    return model


def evaluate_model(episodes=5, render=True, model_path=None):
    model = load_model(model_path)

    render_mode = "human" if render else None
    env = gym.make("CartPole-v1", render_mode=render_mode)

    episode_rewards = []
    print(f"Starting evaluation on CartPole-v1 for {episodes} episodes...\n")

    for episode in range(episodes):
        state, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_tensor)
                action = int(torch.argmax(q_values).item())

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += float(reward)

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{episodes} - Reward: {episode_reward:.1f}")

    env.close()

    avg_reward = float(np.mean(episode_rewards))
    print(f"\nEvaluation finished. Average Reward: {avg_reward:.2f}")
    return episode_rewards


if __name__ == "__main__":
    evaluate_model(episodes=5, render=True)
