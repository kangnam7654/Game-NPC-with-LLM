import os
import sys
import time

from stable_baselines3 import DQN, PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl.environments.simple_maze_env import SimpleMazeEnv

# Create the SimpleMazeEnv environment with human render mode
env = SimpleMazeEnv(render_mode="human")

# Load the trained PPO agent
model = PPO.load("rl/dqn_simple_maze.zip")

# Run the agent for a few episodes
for episode in range(10):
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        time.sleep(0.1)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()
