import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from llm.clients.ollama_client import OllamaClient
from rl.environments.maze_env import MazeEnv


class RenderCallback(BaseCallback):
    """
    A custom callback that renders the environment every N steps.
    """

    def __init__(self, render_freq: int, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        # Render the environment
        # self.env.render() is not working as expected with vectorized env
        # We need to call the render method of the underlying environment
        self.training_env.get_attr("render")[0]()
        return True


# Create log dir
log_dir = "rl/logs/"
os.makedirs(log_dir, exist_ok=True)

# Dummy LLM client for the environment
llm_client = OllamaClient(model="gpt-oss:20b")

# Create and wrap the environment
env = MazeEnv(llm_client=llm_client, render_mode="ansi")
env = Monitor(env, log_dir)

# Instantiate the agent
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)

# Create callback
render_callback = RenderCallback(render_freq=1)

# Train the agent
model.learn(total_timesteps=500000, callback=render_callback)

# Save the agent
model.save("rl/ppo_maze")

print("Training finished and model saved.")