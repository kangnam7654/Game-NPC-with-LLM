import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm.clients.ollama_client import OllamaClient
from rl.environments.maze_env import MazeEnv


# Create log dir
log_dir = "rl/logs/"
os.makedirs(log_dir, exist_ok=True)

# Dummy LLM client for the environment
llm_client = OllamaClient(model="gpt-oss:20b")

# Create and wrap the environment
env = MazeEnv(llm_client=llm_client)
env = Monitor(env, log_dir)

# Instantiate the agent
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
)

# Train the agent
model.learn(total_timesteps=100000)

# Save the agent
model.save("rl/ppo_maze")

print("Training finished and model saved.")
