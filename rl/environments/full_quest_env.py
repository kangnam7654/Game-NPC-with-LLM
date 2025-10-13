import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from configs import config
from game.games.game import Game
from game.renderers.renderer import Renderer


class FullQuestEnv(gym.Env):
    """An environment for the full quest: NPC1 -> Treasure -> NPC2 -> Treasure -> Exit."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.game = Game(llm_client=None)
        self.render_mode = render_mode

        # Custom state for the quest progression
        self.visited_treasure_first = False

        self.screen = None
        self.clock = None
        self.renderer = None

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
            )
            pygame.display.set_caption("Full Quest Maze - Training")
            self.clock = pygame.time.Clock()
            self.renderer = Renderer(self.screen)

        self.action_space = spaces.Discrete(5)  # up, down, left, right
        self.action_map = {0: "up", 1: "down", 2: "left", 3: "right", 4: "interact"}

        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0,
                    high=1,
                    shape=(config.GRID_HEIGHT, config.GRID_WIDTH, 6),
                    dtype=np.uint8,
                ),
                "has_location_info": spaces.Discrete(2),
                "visited_treasure_first": spaces.Discrete(2),
                "has_password_info": spaces.Discrete(2),
                "treasure_opened": spaces.Discrete(2),
                "password": spaces.Discrete(10000)
            }
        )

    def _get_obs(self):
        grid_obs = np.zeros((config.GRID_HEIGHT, config.GRID_WIDTH, 6), dtype=np.uint8)
        grid_np = np.array(self.game.grid)

        grid_obs[:, :, 0] = grid_np == 1  # Walls
        px, py = self.game.player_pos
        grid_obs[py, px, 1] = 1  # Player

        loc_npc_pos = self.game.npcs[0].pos
        grid_obs[loc_npc_pos[1], loc_npc_pos[0], 2] = 1  # Location NPC

        pw_npc_pos = self.game.npcs[1].pos
        grid_obs[pw_npc_pos[1], pw_npc_pos[0], 3] = 1  # Password NPC

        tx, ty = self.game.treasure_pos
        grid_obs[ty, tx, 4] = 1  # Treasure

        ex, ey = self.game.exit_pos
        grid_obs[ey, ex, 5] = 1  # Exit

        return {
            "grid": grid_obs,
            "has_location_info": int(self.game.knows_location),
            "visited_treasure_first": int(self.visited_treasure_first),
            "has_password_info": int(self.game.knows_password),
            "treasure_opened": int(self.game.treasure_opened),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self.game.knows_location = False
        self.game.knows_password = False
        self.game.treasure_opened = False
        self.visited_treasure_first = False

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
        player_pos = self.game.player_pos

        if player_pos == self.prev_pos:
            reward -= 0.1  # Wall penalty or Not Moving

        # Quest progression
        # 1. Visit Location NPC
        if (
            not self.game.knows_location
            and self.game.is_adjacent(player_pos, self.game.npcs[0].pos)
            and action == 4
        ):
            self.game.knows_location = True
            reward += 50

        # 2. Visit Treasure (first time)
        elif (
            self.game.knows_location
            and not self.visited_treasure_first
            and self.game.is_adjacent(player_pos, self.game.treasure_pos)
            and action == 4
        ):
            self.visited_treasure_first = True
            self.game.treasure_visible = True
            reward += 50

        # 3. Visit Password NPC
        elif (
            self.visited_treasure_first
            and not self.game.knows_password
            and self.game.is_adjacent(player_pos, self.game.npcs[1].pos)
            and action == 4
        ):
            self.game.knows_password = True
            reward += 50

        # 4. Open Treasure (second visit)
        elif (
            self.game.knows_password
            and not self.game.treasure_opened
            and self.game.is_adjacent(player_pos, self.game.treasure_pos)
            and action == 4
        ):
            self.game.treasure_opened = True
            reward += 50

        # Penalty for out-of-order actions
        elif (
            self.game.is_adjacent(player_pos, self.game.npcs[0].pos)
            or self.game.is_adjacent(player_pos, self.game.npcs[1].pos)
            or self.game.is_adjacent(player_pos, self.game.treasure_pos)
        ) and action == 4:
            reward -= 0.5

        # 5. Reach Exit
        terminated = False
        if player_pos == self.game.exit_pos:
            if self.game.treasure_opened:
                reward += 100
                terminated = True
            else:
                reward -= 1  # Penalty for reaching exit without opening treasure

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
