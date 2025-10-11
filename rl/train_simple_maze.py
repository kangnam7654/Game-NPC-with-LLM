import os
import sys

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl.environments.simple_maze_env import SimpleMazeEnv

# Create log dir
log_dir = "rl/logs/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = SimpleMazeEnv()
env = Monitor(env, log_dir)

# Instantiate the agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=0.0003,
    n_steps=512,
    ent_coef=0.005,
)

# Train the agent
model.learn(total_timesteps=100000)

# Save the agent
model.save("rl/dqn_simple_maze")

print("Training finished and model saved.")
