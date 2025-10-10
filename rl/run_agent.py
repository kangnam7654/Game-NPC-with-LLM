import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

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
        model = PPO.load("rl/ppo_maze", env=env)
    except FileNotFoundError:
        print("Error: Model file 'rl/ppo_maze.zip' not found.")
        print("Please run train.py to create a model file first.")
        return

    obs = env.reset()

    num_episodes = 10
    for episode in range(num_episodes):
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated = env.step(action)

            # The render method in the environment handles all Pygame updates and events
            env.render()

            # Add a delay to make the agent's actions observable
            time.sleep(0.5)

            done = terminated[0] or truncated[0].get("TimeLimit.truncated", False)

        print(f"Episode {episode + 1} finished. Final reward: {reward[0]}")
        print("Resetting environment for the next episode.")


if __name__ == "__main__":
    main()