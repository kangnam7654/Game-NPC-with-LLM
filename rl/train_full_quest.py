import os
import sys

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl.environments.full_quest_env_pass import FullQuestEnv

# Create log dir
log_dir = "rl/logs/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = FullQuestEnv()
env = TimeLimit(env, max_episode_steps=2000)  # 에피소드 최대 스텝 수 제한
env = Monitor(env, log_dir)

# Instantiate the agent
# This is a complex task, so hyperparameters might need tuning.
# n_steps is increased for better exploration before an update.
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=0.0001,
    n_steps=1024,  # Increased from 512
    batch_size=64,
    n_epochs=20,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.1,  # Slightly increased for more exploration
)

# Train the agent
# This is a very complex task and may require many more timesteps.
model.learn(total_timesteps=10000000)

# Save the agent
model.save("rl/ppo_full_quest")

print("Training finished and model saved as rl/ppo_full_quest.zip")
