import logging

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from configs import config
from game.games.game import Game
from game.renderers.renderer import Renderer

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


class FullQuestEnv(gym.Env):
    """An environment for the full quest: NPC1 -> Treasure -> NPC2 -> Treasure -> Exit."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.game = Game(llm_client=None)
        self.render_mode = render_mode

        # Custom state for the quest progression
        self.visited_treasure_first = False
        # State for password input
        self.password_input_mode = False
        self.entered_password = ""

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

        # 0-3: move, 4: interact, 5-14: digits 0-9
        self.action_space = spaces.Discrete(15)
        self.action_map = {
            0: "up",
            1: "down",
            2: "left",
            3: "right",
            4: "interact",
            **{i + 5: f"digit_{i}" for i in range(10)},
        }
        self.digit_actions = {f"digit_{i}": str(i) for i in range(10)}

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
                "password_input_mode": spaces.Discrete(2),
                "treasure_opened": spaces.Discrete(2),
                "treasure_password": spaces.Discrete(10000),
                "entered_password": spaces.Discrete(10000),
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
            "password_input_mode": int(self.password_input_mode),
            "treasure_password": (
                int(self.game.password) if self.game.knows_password else 0
            ),
            "entered_password": (
                int(self.entered_password) if self.entered_password else 0
            ),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self.game.knows_location = False
        self.game.knows_password = False
        self.game.treasure_opened = False
        self.visited_treasure_first = False
        self.password_input_mode = False
        self.entered_password = ""

        observation = self._get_obs()
        info = {}
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()

        action_name = self.action_map[action]
        reward = -0.1  # Time penalty
        player_pos = self.game.player_pos
        # =======================
        # | Password Input Mode |
        # =======================
        if self.password_input_mode:
            if action_name in self.digit_actions:
                self.entered_password += str(self.digit_actions[action_name])
                reward += 0.05  # Small reward for correct action type

                if len(self.entered_password) == 4:
                    # if self.entered_password == self.game.password:
                    n_correct = self.calculate_score(
                        int(self.game.password), int(self.entered_password)
                    )
                    if n_correct == 4:
                        self.game.treasure_opened = True
                        reward += 50  # Big reward for opening the treasure
                    elif n_correct == 3:
                        reward += 0.04
                    elif n_correct == 2:
                        reward += 0.01
                    elif n_correct == 1:
                        reward += 0.005
                    # Reset password mode
                    logger.debug(f"Password entered: {self.entered_password}")
                    self.password_input_mode = False
                    self.entered_password = ""
            else:
                # Penalty for doing non-digit action in password mode
                reward -= 0.1
                # Exit password mode if player moves away or interacts again
                self.password_input_mode = False
                self.entered_password = ""

            observation = self._get_obs()
            terminated = False
            return observation, reward, terminated, False, {}

        # ===============
        # | Normal Mode |
        # ===============
        self.prev_pos = self.game.player_pos
        if action < 4:  # Move actions
            self.game.step(action_name)
            player_pos = self.game.player_pos  # Update player_pos after move

        if player_pos == self.prev_pos:
            reward -= 0.01  # Wall penalty or Not Moving

        # Quest progression
        # interact action
        if action == 4:
            # 1. Visit Location NPC
            if not self.game.knows_location and self.game.is_adjacent(
                player_pos, self.game.npcs[0].pos
            ):
                self.game.knows_location = True
                logger.debug("LOCATION NPC INTERACTION")
                reward += 50

            # 2. Visit Treasure (first time)
            elif (
                self.game.knows_location
                and not self.visited_treasure_first
                and not self.game.knows_password
                and self.game.is_adjacent(player_pos, self.game.treasure_pos)
            ):
                self.visited_treasure_first = True
                self.game.treasure_visible = True
                logger.debug("TREASURE FOUND")
                reward += 50  # Reduced reward, as this is just a step

            # 3. Visit Password NPC
            elif (
                self.visited_treasure_first
                and not self.game.knows_password
                and self.game.is_adjacent(player_pos, self.game.npcs[1].pos)
            ):
                self.game.knows_password = True
                logger.debug("PASSWORD NPC INTERACTION")
                reward += 50  # Now agent has all info

            # 4. Open Treasure (second visit)
            # 4. Try to Open Treasure (enter password mode)
            elif (
                self.game.knows_password
                and not self.game.treasure_opened
                and self.game.is_adjacent(player_pos, self.game.treasure_pos)
            ):
                self.password_input_mode = True
                self.entered_password = ""

            # Penalty for out-of-order actions
            elif (  # interact action
                self.game.is_adjacent(player_pos, self.game.npcs[0].pos)
                or self.game.is_adjacent(player_pos, self.game.npcs[1].pos)
                or self.game.is_adjacent(player_pos, self.game.treasure_pos)
            ):
                reward -= 0.05

        # 5. Reach Exit
        terminated = False
        if player_pos == self.game.exit_pos:
            if self.game.treasure_opened:
                reward += 100
                terminated = True
            else:
                reward -= 0.05  # Penalty for reaching exit without opening treasure

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

    def calculate_score(self, answer, guess):
        score = 0
        # 숫자를 네 자리 문자열로 변환하여 각 자리를 쉽게 비교합니다.
        # f-string의 '04d' 포맷은 4자리 미만일 경우 앞에 0을 채워줍니다. (예: 4 -> '0004')
        answer_str = f"{answer:04d}"
        guess_str = f"{guess:04d}"

        # 네 자리 숫자의 각 위치를 순회합니다 (인덱스 0, 1, 2, 3).
        for i in range(4):
            # 같은 위치(i)의 숫자가 서로 같으면 점수를 1 올립니다.
            if answer_str[i] == guess_str[i]:
                score += 1

        return score
