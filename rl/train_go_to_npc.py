
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl.environments.go_to_npc_env import GoToNpcEnv


# Create log dir
log_dir = "rl/logs/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = GoToNpcEnv()
env = Monitor(env, log_dir)

# Instantiate the agent
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=0.0001,
    n_steps=512,
    batch_size=64,
    n_epochs=20,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.005,
)

# Train the agent
model.learn(total_timesteps=2000000)

# Save the agent
model.save("rl/ppo_go_to_npc")

print("Training finished and model saved.")
