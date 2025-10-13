import os
import sys
import time

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl.environments.full_quest_env_pass import FullQuestEnv

# Create the environment with human render mode
env = FullQuestEnv(render_mode="human")
env = TimeLimit(env, max_episode_steps=100)

# Load the trained PPO agent
model_path = "rl/ppo_full_quest.zip"
if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    print("Please train the agent first by running train_full_quest.py")
    sys.exit(1)

model = PPO.load(model_path)

# Run the agent for a few episodes
for episode in range(10):
    obs, info = env.reset()
    done = False
    total_reward = 0
    step = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        time.sleep(0.5)
        total_reward += reward
        step += 1
        if truncated:
            print("Episode truncated")
            break
    print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps = {step}")

env.close()
