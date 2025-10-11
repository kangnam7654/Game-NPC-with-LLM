import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from configs import config
from game.games.game import Game
from game.renderers.renderer import Renderer


class GoToNpcEnv(gym.Env):
    """An environment where the agent must go to an NPC before going to the exit."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.game = Game(llm_client=None)
        self.render_mode = render_mode

        self.screen = None
        self.clock = None
        self.renderer = None

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
            )
            pygame.display.set_caption("Go to NPC Maze - Training")
            self.clock = pygame.time.Clock()
            self.renderer = Renderer(self.screen)

        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.action_map = {0: "up", 1: "down", 2: "left", 3: "right"}

        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0,
                    high=1,
                    shape=(config.GRID_HEIGHT, config.GRID_WIDTH, 4),
                    dtype=np.uint8,
                ),
                "has_info": spaces.Discrete(2),
            }
        )

    def _get_obs(self):
        grid_obs = np.zeros((config.GRID_HEIGHT, config.GRID_WIDTH, 4), dtype=np.uint8)
        grid_np = np.array(self.game.grid)

        grid_obs[:, :, 0] = grid_np == 1  # Walls
        px, py = self.game.player_pos
        grid_obs[py, px, 1] = 1  # Player

        # For simplicity, the first NPC is the location informant
        nx, ny = self.game.npcs[0].pos
        grid_obs[ny, nx, 2] = 1  # NPC

        ex, ey = self.game.exit_pos
        grid_obs[ey, ex, 3] = 1  # Exit

        return {"grid": grid_obs, "has_info": int(self.game.knows_location)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        # In this simplified env, we'll say the player doesn't know the location
        self.game.knows_location = False
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

        if self.game.player_pos == self.prev_pos:
            reward -= 0.1  # Wall penalty

        # Check if agent is at the NPC position
        if (
            not self.game.knows_location
            and self.game.player_pos == self.game.npcs[0].pos
        ):
            self.game.knows_location = True
            reward += 50  # Reward for getting info

        terminated = False
        # Check if agent is at the exit
        if self.game.player_pos == self.game.exit_pos:
            if self.game.knows_location:
                reward += 100  # Reward for reaching exit with info
                terminated = True
            else:
                reward -= 1  # Penalty for reaching exit without info

        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            # Rendering logic from SimpleMazeEnv
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
