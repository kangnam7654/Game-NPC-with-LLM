
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from configs import config
from game.games.game import Game
from game.games.states import GameState
from game.renderers.renderer import Renderer


class SimpleMazeEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        # The game object is now much simpler
        self.game = Game(llm_client=None)  # No LLM client needed
        self.render_mode = render_mode

        self.screen = None
        self.clock = None
        self.renderer = None

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
            )
            pygame.display.set_caption("Simple Maze - Training")
            self.clock = pygame.time.Clock()
            self.renderer = Renderer(self.screen)

        # Action space is just movement
        self.action_space = spaces.Discrete(4)
        self.action_map = {0: "up", 1: "down", 2: "left", 3: "right"}

        # Observation space is just the grid
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(config.GRID_HEIGHT, config.GRID_WIDTH, 3),
            dtype=np.uint8,
        )

    def _get_obs(self):
        grid_obs = np.zeros(
            (config.GRID_HEIGHT, config.GRID_WIDTH, 3), dtype=np.uint8
        )
        grid_np = np.array(self.game.grid)

        # Walls
        grid_obs[:, :, 0] = grid_np == 1

        # Player
        px, py = self.game.player_pos
        grid_obs[py, px, 1] = 1

        # Exit
        ex, ey = self.game.exit_pos
        grid_obs[ey, ex, 2] = 1

        return grid_obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        observation = self._get_obs()
        info = {}
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()

        self.prev_pos = self.game.player_pos
        action_name = self.action_map[action]
        self.game.step(action_name)

        reward = -0.01  # Time penalty

        # Penalty for hitting a wall
        if self.game.player_pos == self.prev_pos:
            reward -= 0.1

        # Reward for reaching the exit
        terminated = False
        if self.game.player_pos == self.game.exit_pos:
            reward += 100
            terminated = True

        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            if self.renderer is None:
                return
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pass

            self.renderer.draw(self.game)
            pygame.display.flip()
            if self.clock:
                self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
