import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from configs import config
from game.games.game import Game
from game.games.states import GameState
from game.renderers.renderer import Renderer


class MazeEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 30}

    def __init__(self, llm_client, render_mode=None):
        super().__init__()
        self.game = Game(llm_client=llm_client)
        self.render_mode = render_mode

        self.screen = None
        self.clock = None
        self.renderer = None
        self.visited_cells = set()

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
            )
            pygame.display.set_caption("절차적 퀘스트 미로 - 학습 중")
            self.clock = pygame.time.Clock()
            self.renderer = Renderer(self.screen)

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0,
                    high=1,
                    shape=(config.GRID_HEIGHT, config.GRID_WIDTH, 5),
                    dtype=np.uint8,
                ),
                "knowledge": spaces.MultiBinary(2),
            }
        )

    def _get_obs(self):
        grid_obs = np.zeros((config.GRID_HEIGHT, config.GRID_WIDTH, 5), dtype=np.uint8)
        grid_np = np.array(self.game.grid)

        grid_obs[:, :, 0] = grid_np == 1
        px, py = self.game.player_pos
        grid_obs[py, px, 1] = 1
        for npc in self.game.npcs:
            nx, ny = npc.pos
            grid_obs[ny, nx, 2] = 1
        if self.game.treasure_visible:
            tx, ty = self.game.treasure_pos
            grid_obs[ty, tx, 3] = 1
        ex, ey = self.game.exit_pos
        grid_obs[ey, ex, 4] = 1

        knowledge_obs = np.array(
            [self.game.knows_location, self.game.knows_password], dtype=np.int8
        )

        return {"grid": grid_obs, "knowledge": knowledge_obs}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self.visited_cells.clear()
        self.visited_cells.add(self.game.player_pos)
        observation = self._get_obs()
        info = {}
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()

        prev_knows_location = self.game.knows_location
        prev_knows_password = self.game.knows_password
        prev_treasure_opened = self.game.treasure_opened
        prev_pos = self.game.player_pos

        action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
        reward = -0.1  # Time penalty

        if action < 4:  # Move action
            self.game.step(action_map[action])
            new_pos = self.game.player_pos
            if prev_pos == new_pos:  # Bumped into a wall
                reward -= 1
            elif new_pos not in self.visited_cells:
                reward += 0.2  # Exploration bonus
                self.visited_cells.add(new_pos)

        else:  # Interact action
            interacted = False
            for npc in self.game.npcs:
                if self.game.is_adjacent(self.game.player_pos, npc.pos):
                    if npc.name == "위치 정보원":
                        self.game.knows_location = True
                    elif npc.name == "암호 전문가":
                        self.game.knows_password = True
                    interacted = True
                    break
            if (
                self.game.treasure_visible
                and self.game.player_pos == self.game.treasure_pos
                and self.game.knows_password
            ):
                self.game.treasure_opened = True
                interacted = True

            if not interacted:
                reward -= 0.5  # Penalty for useless interaction

        # Rewards for gaining info / opening treasure
        if not prev_knows_location and self.game.knows_location:
            reward += 50
        if not prev_knows_password and self.game.knows_password:
            reward += 50
        if not prev_treasure_opened and self.game.treasure_opened:
            reward += 100
            self.game.objective = "목표: 출구로 탈출하기"

        # Check for game termination
        terminated = False
        if self.game.treasure_opened and self.game.player_pos == self.game.exit_pos:
            self.game.state = GameState.GAME_OVER

        if self.game.state == GameState.GAME_OVER:
            terminated = True
            reward += 500

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

        elif self.render_mode == "ansi":
            # ... (omitted for brevity, same as before)
            pass

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
