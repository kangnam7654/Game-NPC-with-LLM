
import gymnasium as gym
from stable_baselines3 import PPO

# Create the CartPole environment with human render mode
env = gym.make("CartPole-v1", render_mode="human")

# Load the trained PPO agent
model = PPO.load("rl/ppo_cartpole.zip")

# Run the agent for a few episodes
for episode in range(10):
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()
