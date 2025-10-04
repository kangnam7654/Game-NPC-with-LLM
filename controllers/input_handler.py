import pygame

from controllers.interaction_handler import InteractionHandler
from games.game import Game
from games.states import GameState


class InputHandler:
    """Handles all user input for the game."""

    def __init__(self, game: Game, interaction_handler: InteractionHandler) -> None:
        """Initializes the InputHandler.

        Args:
            game (Game): The main game object.
            interaction_handler (InteractionHandler): Handler for game interactions.
        """
        self.game: Game = game
        self.interaction_handler: InteractionHandler = interaction_handler

    def handle_events(self) -> bool:
        """Processes all pending Pygame events and updates the game state.

        Returns:
            bool: False if the game should quit, True otherwise.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Signal to quit the game

            # Delegate to the appropriate handler based on game state
            if self.game.state == GameState.TEXT_INPUT:
                self._handle_text_input(event)
            elif self.game.state == GameState.INTERACTION_MENU:
                self._handle_menu_input(event)
            elif self.game.state == GameState.GAME_OVER:
                self._handle_game_over_input(event)
            elif self.game.state == GameState.PLAYING:
                self._handle_playing_input(event)
        return True  # Signal to continue the game

    def _handle_text_input(self, event: pygame.event.Event) -> None:
        """Handles input during the TEXT_INPUT state."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self.interaction_handler.process_text_input()
            elif event.key == pygame.K_BACKSPACE:
                if not self.game.editing_text:
                    self.game.input_text = self.game.input_text[:-1]
            elif event.key == pygame.K_ESCAPE:
                pygame.key.stop_text_input()
                if self.game.active_npc:
                    self.game.active_npc.chat_history = []
                self.game.state = GameState.PLAYING
                self.game.active_npc = None
                self.game.chat_display_text = ""
                self.game.editing_text = ""
        elif event.type == pygame.TEXTEDITING:
            self.game.editing_text = event.text
        elif event.type == pygame.TEXTINPUT:
            self.game.editing_text = ""
            self.game.input_text += event.text

    def _handle_menu_input(self, event: pygame.event.Event) -> None:
        """Handles input during the INTERACTION_MENU state."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.game.menu_selection = (self.game.menu_selection - 1) % 3
            elif event.key == pygame.K_DOWN:
                self.game.menu_selection = (self.game.menu_selection + 1) % 3
            elif event.key == pygame.K_RETURN:
                self._process_menu_selection()
            elif event.key == pygame.K_ESCAPE:
                self.game.state = GameState.PLAYING
                self.game.active_npc = None
                self.game.chat_display_text = ""

    def _process_menu_selection(self) -> None:
        """Processes the selected option in the interaction menu."""
        if self.game.active_npc is None:
            return

        if self.game.menu_selection == 0:  # Get fixed info
            npc = self.game.active_npc
            if npc.name == "위치 정보원":
                self.game.knows_location = True
                self.game.dialogue = f"[{npc.name}]: 보물은 ({self.game.treasure_pos[0]}, {self.game.treasure_pos[1]}) 좌표에 있네."
                self.game.objective = "목표: 보물상자의 암호 알아내기"
            elif npc.name == "암호 전문가":
                if self.game.knows_location:
                    self.game.knows_password = True
                    self.game.dialogue = (
                        f"[{npc.name}]: 암호는 '{self.game.password}'일세."
                    )
                    self.game.objective = "목표: 보물상자를 열기"
                else:
                    self.game.dialogue = f"[{npc.name}]: 위치부터 알아오게."
            self.game.state = GameState.PLAYING
            self.game.active_npc = None
        elif self.game.menu_selection == 1:  # Start natural language chat
            pygame.key.start_text_input()
            self.game.state = GameState.TEXT_INPUT
            initial_message = "무엇이 궁금한가? (ESC로 종료)"
            if not self.game.active_npc.chat_history:
                self.game.active_npc.chat_history.append(
                    {"role": "assistant", "content": initial_message}
                )
            self.interaction_handler._update_chat_display()

        elif self.game.menu_selection == 2:  # Leave
            self.game.state = GameState.PLAYING
            self.game.active_npc = None
            self.game.chat_display_text = ""

    def _handle_game_over_input(self, event: pygame.event.Event) -> None:
        """Handles input during the GAME_OVER state."""
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self.game.reset()

    def _handle_playing_input(self, event: pygame.event.Event) -> None:
        """Handles input during the PLAYING state."""
        if event.type == pygame.KEYDOWN:
            action: str | None = None
            if event.key in [pygame.K_UP, pygame.K_w]:
                action = "up"
            elif event.key in [pygame.K_DOWN, pygame.K_s]:
                action = "down"
            elif event.key in [pygame.K_LEFT, pygame.K_a]:
                action = "left"
            elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                action = "right"
            elif event.key == pygame.K_SPACE:
                self.interaction_handler.handle_interaction()

            if action:
                self.game.step(action)
