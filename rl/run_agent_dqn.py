import os
import sys
import time

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm.clients.ollama_client import OllamaClient
from rl.environments.maze_env import MazeEnv


def main() -> None:
    """Initializes and runs the main game loop with a trained RL agent."""
    llm_client = OllamaClient(model="gpt-oss:20b")

    # Wrap the environment in a DummyVecEnv for consistency
    env = DummyVecEnv([lambda: MazeEnv(llm_client=llm_client, render_mode="human")])

    # Load the trained agent
    try:
        # The environment is automatically set when loading
        model = DQN.load("rl/dqn_maze.zip", env=env)
    except FileNotFoundError:
        print("Error: Model file 'rl/dqn_maze.zip' not found.")
        print("Please run train_dqn.py to create a model file first.")
        return

    obs = env.reset()

    num_episodes = 10
    for episode in range(num_episodes):
        total_reward = 0
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated = env.step(action)
            total_reward += reward[0]

            # The render method in the environment handles all Pygame updates and events
            env.render()

            # Add a delay to make the agent's actions observable
            # time.sleep(0.1)
            print(f"Action taken: {action}, Reward received: {reward[0]}")
            done = terminated[0] or truncated[0].get("TimeLimit.truncated", False)
        print(f"Episode {episode + 1} finished. Final reward: {total_reward}")
        print("Resetting environment for the next episode.")


if __name__ == "__main__":
    main()
