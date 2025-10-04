import pygame

from configs import config
from games.game import Game


class UIManager:
    """Handles drawing all UI overlays for the game (menus, dialogs, etc.).

    Attributes:
        screen (pygame.Surface): The main screen surface to draw on.
        fonts (dict[str, pygame.font.Font]): A dictionary of pre-loaded fonts.
    """

    def __init__(
        self, screen: pygame.Surface, fonts: dict[str, pygame.font.Font]
    ) -> None:
        """Initializes the UIManager.

        Args:
            screen (pygame.Surface): The main screen surface.
            fonts (dict[str, pygame.font.Font]): A dictionary of fonts.
        """
        self.screen: pygame.Surface = screen
        self.fonts: dict[str, pygame.font.Font] = fonts

    def _wrap_text(
        self, text: str, font: pygame.font.Font, max_width: int
    ) -> list[str]:
        """Wraps text to fit within a specified width.

        Args:
            text (str): The text to wrap.
            font (pygame.font.Font): The font to use for measuring text width.
            max_width (int): The maximum width in pixels for a line.

        Returns:
            list[str]: A list of strings, where each string is a wrapped line.
        """
        words = text.split(" ")
        lines = []
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            if font.size(test_line)[0] <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        return lines

    def draw_ui_box(self, rect: pygame.Rect, title: str = "") -> None:
        """Draws a standard UI box with a background, border, and optional title.

        Args:
            rect (pygame.Rect): The rectangle defining the box's position and size.
            title (str, optional): An optional title to display at the top of the box.
        """
        pygame.draw.rect(self.screen, config.UI_BG_COLOR, rect)
        pygame.draw.rect(self.screen, config.UI_BORDER_COLOR, rect, 2)
        if title:
            title_surf = self.fonts["info"].render(title, True, config.WHITE)
            self.screen.blit(title_surf, (rect.x + 10, rect.y + 10))

    def draw_interaction_menu(self, game: Game) -> None:
        """Draws the NPC interaction menu.

        Args:
            game (Game): The main game object containing the game state.
        """
        menu_w, menu_h = 300, 150
        menu_rect = pygame.Rect(
            (config.SCREEN_WIDTH - menu_w) / 2,
            (config.SCREEN_HEIGHT - config.INFO_PANEL_HEIGHT - menu_h) / 2,
            menu_w,
            menu_h,
        )
        if game.active_npc:
            self.draw_ui_box(menu_rect, f"{game.active_npc.name}와(과) 대화")

        options = ["(고정된) 정보 확인", "자유롭게 대화하기", "떠나기 (ESC)"]
        for i, option in enumerate(options):
            prefix = "> " if i == game.menu_selection else "  "
            text_color = (
                config.PLAYER_COLOR if i == game.menu_selection else config.WHITE
            )
            option_text = f"{prefix}{i+1}. {option}"
            opt_surf = self.fonts["info"].render(option_text, True, text_color)
            self.screen.blit(opt_surf, (menu_rect.x + 20, menu_rect.y + 50 + i * 30))

    def draw_text_input(self, game: Game) -> None:
        """Draws the text input interface for chatting or entering passwords.

        Args:
            game (Game): The main game object containing the game state.
        """
        input_h, chat_h = 100, 200
        if game.active_npc:  # Chat window
            ui_rect = pygame.Rect(50, 50, config.SCREEN_WIDTH - 100, chat_h + input_h)
            self.draw_ui_box(ui_rect, "대화하기 (ESC로 종료)")

            # --- Text Wrapping Logic ---
            chat_area_width = ui_rect.width - 20  # Padding
            y_offset = ui_rect.y + 40
            line_height = 25

            all_lines = []
            for message in game.active_npc.chat_history:
                role = message["role"]
                content = message["content"]

                if role == "user":
                    prefix = "Player: "
                    color = config.PLAYER_COLOR
                else:  # assistant
                    prefix = f"{game.active_npc.name}: "
                    color = config.WHITE

                wrapped_lines = self._wrap_text(
                    prefix + content, self.fonts["info"], chat_area_width
                )
                all_lines.extend([(line, color) for line in wrapped_lines])

            # Display only the last few lines that fit in the chat box
            chat_box_height = chat_h - 40  # Top padding
            max_visible_lines = chat_box_height // line_height
            visible_lines = all_lines[-max_visible_lines:]

            for line, color in visible_lines:
                if y_offset + line_height > ui_rect.y + chat_h:
                    break  # Stop if we're about to draw outside the chat area
                line_surf = self.fonts["info"].render(line, True, color)
                self.screen.blit(line_surf, (ui_rect.x + 10, y_offset))
                y_offset += line_height
            # --- End of Text Wrapping Logic ---

            input_rect = pygame.Rect(
                ui_rect.x, ui_rect.y + chat_h, ui_rect.width, input_h
            )
            self.draw_ui_box(input_rect, "내용 입력 (Enter로 전송):")
            full_text = game.input_text + game.editing_text + "|"
            text_surf = self.fonts["info"].render(full_text, True, config.WHITE)
            self.screen.blit(text_surf, (input_rect.x + 10, input_rect.y + 40))
        else:  # Password input
            ui_rect = pygame.Rect(100, 100, config.SCREEN_WIDTH - 200, input_h)
            self.draw_ui_box(ui_rect, game.input_prompt)
            full_text = game.input_text + game.editing_text + "|"
            text_surf = self.fonts["info"].render(full_text, True, config.WHITE)
            self.screen.blit(text_surf, text_surf.get_rect(center=ui_rect.center))

    def draw_game_over(self, game: Game) -> None:
        """Draws the game over screen.

        Args:
            game (Game): The main game object containing the game state.
        """
        text_surf = self.fonts["main"].render(game.message, True, config.WHITE)
        self.screen.blit(
            text_surf,
            text_surf.get_rect(
                center=(
                    config.SCREEN_WIDTH / 2,
                    (config.GRID_HEIGHT * config.GRID_SIZE) / 2,
                )
            ),
        )
        restart_surf = self.fonts["info"].render(
            "Press 'R' to restart", True, config.WHITE
        )
        self.screen.blit(
            restart_surf,
            restart_surf.get_rect(
                center=(
                    config.SCREEN_WIDTH / 2,
                    (config.GRID_HEIGHT * config.GRID_SIZE) / 2 + 40,
                )
            ),
        )
