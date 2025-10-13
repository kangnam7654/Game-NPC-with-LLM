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
        reward = -0.01  # Time penalty
        player_pos = self.game.player_pos
        # =======================
        # | Password Input Mode |
        # =======================
        if self.password_input_mode:
            if action_name in self.digit_actions:
                self.entered_password += str(self.digit_actions[action_name])
                reward += 0.1  # Small reward for correct action type

                if len(self.entered_password) == 1:
                    if self.entered_password[0] == self.game.password[0]:
                        reward += 0.1
                    else:
                        reward -= 0.1
                elif len(self.entered_password) == 2:
                    if self.entered_password == self.game.password[:2]:
                        reward += 0.1
                    else:
                        reward -= 0.1
                elif len(self.entered_password) == 3:
                    if self.entered_password == self.game.password[:3]:
                        reward += 0.1
                    else:
                        reward -= 0.1
                elif len(self.entered_password) == 4:
                    if self.entered_password == self.game.password:
                        self.game.treasure_opened = True
                        reward += 100  # Big reward for opening the treasure
                    else:
                        reward -= 10  # Big penalty for wrong password
                    # Reset password mode
                    self.password_input_mode = False
                    self.entered_password = ""
            else:
                # Penalty for doing non-digit action in password mode
                reward -= 0.5
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
            reward -= 0.1  # Wall penalty or Not Moving

        # Quest progression
        # interact action
        if action == 4:
            # 1. Visit Location NPC
            if not self.game.knows_location and self.game.is_adjacent(
                player_pos, self.game.npcs[0].pos
            ):
                self.game.knows_location = True
                reward += 50

            # 2. Visit Treasure (first time)
            elif (
                self.game.knows_location
                and not self.visited_treasure_first
                and not self.visited_treasure_first
                and not self.game.knows_password
                and self.game.is_adjacent(player_pos, self.game.treasure_pos)
            ):
                self.visited_treasure_first = True
                self.game.treasure_visible = True
                reward += 25  # Reduced reward, as this is just a step

            # 3. Visit Password NPC
            elif (
                self.visited_treasure_first
                and not self.game.knows_password
                and self.game.is_adjacent(player_pos, self.game.npcs[1].pos)
            ):
                self.game.knows_password = True
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
                reward += 0.5

            # Penalty for out-of-order actions
            elif (  # interact action
                self.game.is_adjacent(player_pos, self.game.npcs[0].pos)
                or self.game.is_adjacent(player_pos, self.game.npcs[1].pos)
                or self.game.is_adjacent(player_pos, self.game.treasure_pos)
            ):
                reward -= 0.2

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
