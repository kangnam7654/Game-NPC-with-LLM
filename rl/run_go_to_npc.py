import os
import sys

import time
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl.environments.go_to_npc_env import GoToNpcEnv

# Create the GoToNpcEnv environment with human render mode
env = GoToNpcEnv(render_mode="human")

# Load the trained PPO agent
model = PPO.load("rl/ppo_go_to_npc.zip")

# Run the agent for a few episodes
for episode in range(10):
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        time.sleep(0.2)
        total_reward += reward
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()
