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

        self.action_space = spaces.Discrete(6)
        self.action_map = {
            0: "up",
            1: "down",
            2: "left",
            3: "right",
            4: "interact",
        }
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0,
                    high=1,
                    shape=(config.GRID_HEIGHT, config.GRID_WIDTH, 6),
                    dtype=np.uint8,
                ),
                "knowledge": spaces.MultiBinary(2),
            }
        )
        self._prev_move = None
        self._num_steps = 0
        self._distance = np.inf
        self._n_meetings = 0

    def _get_obs(self):
        grid_obs = np.zeros((config.GRID_HEIGHT, config.GRID_WIDTH, 6), dtype=np.uint8)
        grid_np = np.array(self.game.grid)

        # Walls
        grid_obs[:, :, 0] = grid_np == 1

        # Player
        px, py = self.game.player_pos
        grid_obs[py, px, 1] = 1

        # NPCs
        npc1 = self.game.npcs[0]
        npc2 = self.game.npcs[1]
        nx1, ny1 = npc1.pos
        nx2, ny2 = npc2.pos
        grid_obs[ny1, nx1, 2] = 1
        grid_obs[ny2, nx2, 3] = 1

        # Treasure and Exit
        if self.game.treasure_visible:
            tx, ty = self.game.treasure_pos
            grid_obs[ty, tx, 4] = 1
        ex, ey = self.game.exit_pos
        grid_obs[ey, ex, 5] = 1

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
        self._prev_move = None
        self._distance = np.inf
        self._n_meetings = 0
        self._num_steps = 0
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
        self.prev_pos = self.game.player_pos

        reward = -0.1  # Time penalty

        # Moving
        if action < 4:  # Move action
            action_name = self.action_map[action]
            self.game.step(action_name)

        else:  # Interact action
            interacted = False
            for npc in self.game.npcs:
                if self.game.is_adjacent(self.game.player_pos, npc.pos):
                    if npc.name == "위치 정보원":
                        self.game.knows_location = True
                    elif npc.name == "암호 전문가" and self.game.knows_location:
                        self.game.knows_password = True
                    interacted = True
                    break

            if (
                self.game.treasure_visible
                and self.game.knows_password
                and self.game.is_adjacent(self.game.player_pos, self.game.treasure_pos)
            ):
                self.game.treasure_opened = True
                interacted = True

            if not interacted:
                reward -= 0.1  # Penalty for useless interaction

        # Make treasure visible if player is near and knows the location
        if self.game.knows_location and not self.game.treasure_visible:
            if (
                self.game.is_adjacent(self.game.player_pos, self.game.treasure_pos)
                or self.game.player_pos == self.game.treasure_pos
            ):
                self.game.treasure_visible = True

        reward += self.compute_reward(
            prev_knows_location, prev_knows_password, prev_treasure_opened, action
        )

        # Check for game termination
        terminated = False
        if self.game.treasure_opened and self.game.player_pos == self.game.exit_pos:
            self.game.state = GameState.GAME_OVER
            terminated = True
            reward += 200

        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        self._num_steps += 1
        if self._num_steps >= config.MAX_STEPS_PER_EPISODE:
            terminated = True
            info["TimeLimit.truncated"] = True
            self._num_steps = 0
        return observation, reward, terminated, False, info

    def compute_reward(
        self, prev_knows_location, prev_knows_password, prev_treasure_opened, action
    ):
        reward = 0

        # Big rewards for achieving goals
        if not prev_knows_location and self.game.knows_location:
            reward += 100
            self._distance = np.inf  # Reset distance for the next phase
        if not prev_knows_password and self.game.knows_password:
            reward += 100
            self._distance = np.inf  # Reset distance for the next phase
        if not prev_treasure_opened and self.game.treasure_opened:
            reward += 100
            self.game.objective = "목표: 출구로 탈출하기"
            self._distance = np.inf  # Reset distance for the next phase

        if action < 4:  # Only apply movement-based rewards if the agent moved
            # Penalty for hitting a wall
            # if self.game.player_pos == self.prev_pos:
            #     reward -= 1  # Penalty for hitting a wall

            # Exploration bonus
            if self.game.player_pos not in self.visited_cells:
                self.visited_cells.add(self.game.player_pos)
                reward += 1  # Reward for visiting a new cell

        return reward

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

    def l2_norm(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def l1_norm(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.sum(np.abs(a - b))
