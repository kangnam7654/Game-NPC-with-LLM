
import gymnasium as gym
from stable_baselines3 import PPO

# Create the CartPole environment
env = gym.make("CartPole-v1")

# Instantiate the PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
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
model.save("rl/ppo_cartpole")

print("Training finished and model saved.")
