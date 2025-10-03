import random

import pygame

from actors.npc import NPC
from configs import config
from games.states import GameState
from maps.map_generator import generate_connected_map


class Game:
    """Manages all game states and core logic.

    Attributes:
        grid (list[list[int]]): The game map represented as a grid.
        player_pos (tuple[int, int]): The player's current position.
        exit_pos (tuple[int, int]): The exit's position.
        treasure_pos (tuple[int, int]): The treasure's position.
        npcs (list[NPC]): A list of non-player characters in the game.
        password (str): The password for the treasure chest.
        knows_location (bool): Whether the player knows the treasure's location.
        knows_password (bool): Whether the player knows the treasure's password.
        treasure_visible (bool): Whether the treasure is visible on the map.
        treasure_opened (bool): Whether the treasure chest has been opened.
        state (GameState): The current state of the game.
        active_npc (NPC | None): The NPC the player is currently interacting with.
        input_text (str): The text currently being entered by the player.
        editing_text (str): The text being edited in an input field.
        input_prompt (str): The prompt displayed for text input.
        chat_history (list[str]): A history of chat messages with NPCs.
        menu_selection (int): The currently selected item in a menu.
        message (str): A message to be displayed to the player.
        dialogue (str): The current dialogue text.
        objective (str): The player's current objective.
    """

    def __init__(self) -> None:
        """Initializes the game state."""
        self.grid: list[list[int]] = []
        self.player_pos: tuple[int, int] = (0, 0)
        self.exit_pos: tuple[int, int] = (0, 0)
        self.treasure_pos: tuple[int, int] = (0, 0)
        self.npcs: list[NPC] = []
        self.password: str = ""
        self.knows_location: bool = False
        self.knows_password: bool = False
        self.treasure_visible: bool = False
        self.treasure_opened: bool = False
        self.state: GameState = GameState.PLAYING
        self.active_npc: NPC | None = None
        self.input_text: str = ""
        self.editing_text: str = ""
        self.input_prompt: str = ""
        self.chat_history: list[str] = []
        self.menu_selection: int = 0
        self.message: str = ""
        self.dialogue: str = ""
        self.objective: str = ""
        self.reset()

    def _get_random_empty_cells(self, count: int) -> list[tuple[int, int]]:
        """Gets a list of random empty cells from the grid.

        Args:
            count (int): The number of empty cells to return.

        Returns:
            list[tuple[int, int]]: A list of (x, y) tuples for empty cells.
        """
        empty_cells: list[tuple[int, int]] = []
        for r in range(config.GRID_HEIGHT):
            for c in range(config.GRID_WIDTH):
                if self.grid[r][c] == 0:
                    empty_cells.append((c, r))
        random.shuffle(empty_cells)
        return [empty_cells.pop() for _ in range(min(count, len(empty_cells)))]

    def reset(self) -> None:
        """Resets the game to its initial state."""
        self.grid = generate_connected_map()

        self.knows_location = False
        self.knows_password = False
        self.treasure_visible = False
        self.treasure_opened = False

        self.state = GameState.PLAYING

        pos = self._get_random_empty_cells(5)
        self.player_pos = pos[0]
        self.exit_pos = pos[1]
        self.treasure_pos = pos[2]
        self.password = str(random.randint(1000, 9999))
        self.npcs = [
            NPC(name="위치 정보원", pos=pos[3], color=config.NPC_LOC_COLOR, label="L"),
            NPC(name="암호 전문가", pos=pos[4], color=config.NPC_PW_COLOR, label="P"),
        ]

        self.active_npc = None
        self.input_text = ""
        self.editing_text = ""
        self.input_prompt = ""
        self.chat_history = []
        self.menu_selection = 0
        self.message = ""
        self.dialogue = "정보를 가진 NPC들을 찾아 대화하세요. (스페이스 바)"
        self.objective = "목표: 보물상자의 위치를 알아내기"

    def is_adjacent(self, pos1: tuple[int, int], pos2: tuple[int, int]) -> bool:
        """Checks if two positions are adjacent (not diagonally).

        Args:
            pos1 (tuple[int, int]): The first position (x, y).
            pos2 (tuple[int, int]): The second position (x, y).

        Returns:
            bool: True if the positions are adjacent, False otherwise.
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def handle_interaction(self) -> None:
        """Handles player interaction with NPCs and the treasure chest."""
        for npc in self.npcs:
            if self.is_adjacent(self.player_pos, npc.pos):
                self.state = GameState.INTERACTION_MENU
                self.active_npc = npc
                self.menu_selection = 0
                return
        if (
            self.treasure_visible
            and self.player_pos == self.treasure_pos
            and not self.treasure_opened
        ):
            if self.knows_password:
                pygame.key.start_text_input()
                self.state = GameState.TEXT_INPUT
                self.input_prompt = "암호를 입력하세요 (Enter로 확인):"
            else:
                self.dialogue = "상자가 잠겨있다. 암호를 알아내야 한다."
            return
        self.dialogue = "주변에 상호작용할 대상이 없습니다."

    def process_text_input(self) -> None:
        """Processes text input from the user for NPC chat or password entry."""
        if self.active_npc:
            player_msg = f"나: {self.input_text}"
            self.chat_history.append(player_msg)
            if "안녕" in self.input_text:
                response = "안녕하신가, 모험가."
            elif "보물" in self.input_text or "위치" in self.input_text:
                if self.active_npc.name == "위치 정보원":
                    response = f"보물은 ({self.treasure_pos[0]}, {self.treasure_pos[1]}) 좌표에 있다네."
                    self.knows_location = True
                else:
                    response = "나는 암호 전문가일세. 위치는 다른 친구에게 물어보게."
            else:
                response = "흠... 흥미로운 질문이군."
            self.chat_history.append(f"[{self.active_npc.name}]: {response}")
            if len(self.chat_history) > 4:
                self.chat_history.pop(0)
            self.input_text = ""
        elif self.input_prompt:
            if self.input_text == self.password:
                self.treasure_opened = True
                self.dialogue = f"암호 '{self.password}'를 입력했다. 보물상자가 열렸다!"
                self.objective = "목표: 출구로 탈출하기"
            else:
                self.dialogue = "암호가 틀렸습니다."
            self.input_text = ""
            pygame.key.stop_text_input()
            self.state = GameState.PLAYING
            self.input_prompt = ""
            self.active_npc = None

    def step(self, action: str) -> None:
        """Updates the game state based on the player's action.

        Args:
            action (str): The action taken by the player (e.g., "up", "down").
        """
        if self.state != GameState.PLAYING:
            return
        px, py = self.player_pos
        if action == "up":
            py -= 1
        elif action == "down":
            py += 1
        elif action == "left":
            px -= 1
        elif action == "right":
            px += 1
        if (
            0 <= px < config.GRID_WIDTH
            and 0 <= py < config.GRID_HEIGHT
            and self.grid[py][px] == 0
        ):
            self.player_pos = (px, py)
        if self.knows_location and self.player_pos == self.treasure_pos:
            self.treasure_visible = True
        if self.treasure_opened and self.player_pos == self.exit_pos:
            self.state = GameState.GAME_OVER
            self.message = "성공! 보물을 가지고 미로를 탈출했습니다!"