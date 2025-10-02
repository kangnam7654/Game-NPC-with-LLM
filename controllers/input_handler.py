import pygame


class InputHandler:
    """모든 사용자 입력을 처리하는 클래스."""

    def __init__(self, game):
        self.game = game

    def handle_events(self):
        """Pygame 이벤트를 처리하고 게임 상태를 업데이트합니다."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # 게임 종료 신호

            # 각 게임 상태에 맞는 입력 처리
            if self.game.state == "text_input":
                self._handle_text_input(event)
            elif self.game.state == "interaction_menu":
                self._handle_menu_input(event)
            elif self.game.state == "game_over":
                self._handle_game_over_input(event)
            elif self.game.state == "playing":
                self._handle_playing_input(event)
        return True  # 게임 계속 진행

    def _handle_text_input(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self.game.process_text_input()
            elif event.key == pygame.K_BACKSPACE:
                if not self.game.editing_text:
                    self.game.input_text = self.game.input_text[:-1]
            elif event.key == pygame.K_ESCAPE:
                pygame.key.stop_text_input()
                self.game.state = "playing"
                self.game.active_npc = None
                self.game.chat_history = []
                self.game.editing_text = ""
        elif event.type == pygame.TEXTEDITING:
            self.game.editing_text = event.text
        elif event.type == pygame.TEXTINPUT:
            self.game.editing_text = ""
            self.game.input_text += event.text

    def _handle_menu_input(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.game.menu_selection = (self.game.menu_selection - 1) % 3
            elif event.key == pygame.K_DOWN:
                self.game.menu_selection = (self.game.menu_selection + 1) % 3
            elif event.key == pygame.K_RETURN:
                self._process_menu_selection()
            elif event.key == pygame.K_ESCAPE:
                self.game.state = "playing"
                self.game.active_npc = None

    def _process_menu_selection(self):
        if self.game.menu_selection == 0:  # 정보 확인
            npc = self.game.active_npc
            if npc["name"] == "위치 정보원":
                self.game.knows_location = True
                self.game.dialogue = f"[{npc['name']}]: 보물은 ({self.game.treasure_pos[0]}, {self.game.treasure_pos[1]}) 좌표에 있네."
                self.game.objective = "목표: 보물상자의 암호 알아내기"
            elif npc["name"] == "암호 전문가":
                if self.game.knows_location:
                    self.game.knows_password = True
                    self.game.dialogue = (
                        f"[{npc['name']}]: 암호는 '{self.game.password}'일세."
                    )
                    self.game.objective = "목표: 보물상자를 열기"
                else:
                    self.game.dialogue = f"[{npc['name']}]: 위치부터 알아오게."
            self.game.state = "playing"
            self.game.active_npc = None
        elif self.game.menu_selection == 1:  # 자연어 대화
            pygame.key.start_text_input()
            self.game.state = "text_input"
            self.game.chat_history.append(
                f"[{self.game.active_npc['name']}]: 무엇이 궁금한가? (ESC로 종료)"
            )
        elif self.game.menu_selection == 2:  # 떠나기
            self.game.state = "playing"
            self.game.active_npc = None

    def _handle_game_over_input(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self.game.reset()

    def _handle_playing_input(self, event):
        if event.type == pygame.KEYDOWN:
            action = None
            if event.key in [pygame.K_UP, pygame.K_w]:
                action = "up"
            elif event.key in [pygame.K_DOWN, pygame.K_s]:
                action = "down"
            elif event.key in [pygame.K_LEFT, pygame.K_a]:
                action = "left"
            elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                action = "right"
            elif event.key == pygame.K_SPACE:
                self.game.handle_interaction()
            if action:
                self.game.step(action)
