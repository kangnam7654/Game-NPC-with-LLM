import os
import sys

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl.environments.full_quest_env_pass import FullQuestEnv

# Create log dir
log_dir = "rl/logs/"
os.makedirs(log_dir, exist_ok=True)
best_model_save_path = os.path.join(log_dir, "best_model")

# Create and wrap the environment
env = FullQuestEnv()
env = TimeLimit(env, max_episode_steps=1024)  # 에피소드 최대 스텝 수 제한
env = Monitor(env, log_dir)

# Create a separate evaluation environment
eval_env = FullQuestEnv()
eval_env = TimeLimit(eval_env, max_episode_steps=1024)
eval_env = Monitor(eval_env, os.path.join(log_dir, "eval"))

# Setup callback for model checkpointing and early stopping
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=best_model_save_path,
    log_path=log_dir,
    eval_freq=10240,  # Evaluate every 10240 steps (1024 * 10)
    n_eval_episodes=5,
    deterministic=True,
    render=False,
)

model_path = "/Users/kangnam/projects/simulator/rl/ppo_full_quest.zip"

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
    ent_coef=0.1,
)

if os.path.exists(model_path):
    print(f"Loading compatible weights from {model_path} and continuing training.")
    # 그리드 크기 변경으로 인해 observation space가 달라져 전체 모델 로딩이 불가한 경우,
    # zip 파일을 열어 policy 가중치만 직접 로드하고, 현재 모델과 호환되는 부분만 적용합니다.
    # 이렇게 하면 신경망 구조가 다른 부분(주로 CNN 및 첫 번째 MLP 레이어)을 제외하고
    # 학습된 가중치를 최대한 재사용할 수 있습니다.
    try:
        from zipfile import ZipFile

        import torch

        with ZipFile(model_path, "r") as zip_ref:
            # policy.pth 파일만 메모리로 읽어옵니다.
            policy_data = zip_ref.read("policy.pth")

        # BytesIO를 사용하여 메모리 내의 데이터를 파일처럼 다룹니다.
        from io import BytesIO

        old_policy_state_dict = torch.load(
            BytesIO(policy_data), map_location=model.device
        )

        new_policy_state_dict = model.policy.state_dict()
        for k, v in old_policy_state_dict.items():
            if k in new_policy_state_dict and v.shape == new_policy_state_dict[k].shape:
                new_policy_state_dict[k] = v
        model.policy.load_state_dict(new_policy_state_dict)
        print("Successfully loaded compatible weights.")
    except Exception as e:
        print(f"Could not load weights due to an error: {e}")
        print("Starting training from scratch.")


# Train the agent
# This is a very complex task and may require many more timesteps.
model.learn(
    total_timesteps=20000000,
    progress_bar=True,
    callback=eval_callback,
    reset_num_timesteps=False,  # 이어서 학습할 때 타임스텝을 초기화하지 않음
)

# Save the agent
model.save("rl/ppo_full_quest")

print("Training finished and model saved as rl/ppo_full_quest.zip")
