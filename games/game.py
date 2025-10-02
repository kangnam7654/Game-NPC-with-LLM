import random

import pygame

from configs import config
from maps.map_generator import generate_connected_map


class Game:
    """게임의 모든 상태와 핵심 로직을 관리하는 클래스."""

    def __init__(self):
        self.reset()

    def _get_random_empty_cells(self, count):
        cells = [
            (c, r)
            for r in range(config.GRID_HEIGHT)
            for c in range(config.GRID_WIDTH)
            if self.grid[r][c] == 0
        ]
        random.shuffle(cells)
        return [cells.pop() for _ in range(min(count, len(cells)))]

    def reset(self):
        self.grid = generate_connected_map()

        self.knows_location = False
        self.knows_password = False
        self.treasure_visible = False
        self.treasure_opened = False

        self.state = "playing"  # playing, interaction_menu, text_input, game_over

        pos = self._get_random_empty_cells(5)
        self.player_pos = pos[0]
        self.exit_pos = pos[1]
        self.treasure_pos = pos[2]
        self.password = str(random.randint(1000, 9999))
        self.npcs = [
            {
                "name": "위치 정보원",
                "pos": pos[3],
                "color": config.NPC_LOC_COLOR,
                "label": "L",
            },
            {
                "name": "암호 전문가",
                "pos": pos[4],
                "color": config.NPC_PW_COLOR,
                "label": "P",
            },
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

    def is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def handle_interaction(self):
        for npc in self.npcs:
            if self.is_adjacent(self.player_pos, npc["pos"]):
                self.state = "interaction_menu"
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
                self.state = "text_input"
                self.input_prompt = "암호를 입력하세요 (Enter로 확인):"
            else:
                self.dialogue = "상자가 잠겨있다. 암호를 알아내야 한다."
            return
        self.dialogue = "주변에 상호작용할 대상이 없습니다."

    def process_text_input(self):
        if self.active_npc:
            player_msg = f"나: {self.input_text}"
            self.chat_history.append(player_msg)
            if "안녕" in self.input_text:
                response = "안녕하신가, 모험가."
            elif "보물" in self.input_text or "위치" in self.input_text:
                if self.active_npc["name"] == "위치 정보원":
                    response = f"보물은 ({self.treasure_pos[0]}, {self.treasure_pos[1]}) 좌표에 있다네."
                    self.knows_location = True
                else:
                    response = "나는 암호 전문가일세. 위치는 다른 친구에게 물어보게."
            else:
                response = "흠... 흥미로운 질문이군."
            self.chat_history.append(f"[{self.active_npc['name']}]: {response}")
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
            self.state = "playing"
            self.input_prompt = ""
            self.active_npc = None

    def step(self, action):
        if self.state != "playing":
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
            self.state = "game_over"
            self.message = "성공! 보물을 가지고 미로를 탈출했습니다!"
