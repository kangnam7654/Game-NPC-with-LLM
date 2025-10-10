import pygame

from configs import config
from game.games.game import Game


class UIManager:
    """Handles drawing all UI overlays for the game (menus, dialogs, etc.).

    Attributes:
        screen (pygame.Surface): The main screen surface to draw on.
        fonts (dict[str, pygame.font.Font]): A dictionary of pre-loaded fonts.
    """

    def __init__(
        self, screen: pygame.Surface, fonts: dict[str, pygame.font.Font]
    ) -> None:
        """Initializes the UIManager."""
        self.screen: pygame.Surface = screen
        self.fonts: dict[str, pygame.font.Font] = fonts

    def _wrap_text(
        self, text: str, font: pygame.font.Font, max_width: int
    ) -> list[str]:
        """Wraps text to fit within a specified width using character-by-character wrapping."""
        lines = []
        current_line = ""
        if max_width <= 0:
            return [text]

        for char in text:
            # Handle explicit newlines
            if char == '\n':
                lines.append(current_line)
                current_line = ""
                continue

            if font.size(current_line + char)[0] <= max_width:
                current_line += char
            else:
                lines.append(current_line)
                current_line = char
        lines.append(current_line)
        return lines

    def draw_ui_box(self, rect: pygame.Rect, title: str = "") -> None:
        """Draws a standard UI box with a background, border, and optional title."""
        pygame.draw.rect(self.screen, config.UI_BG_COLOR, rect)
        pygame.draw.rect(self.screen, config.UI_BORDER_COLOR, rect, 2)
        if title:
            title_surf = self.fonts["info"].render(title, True, config.WHITE)
            self.screen.blit(title_surf, (rect.x + 10, rect.y + 10))

    def draw_interaction_menu(self, game: Game) -> None:
        """Draws the NPC interaction menu."""
        # --- Dynamic Size Calculation ---
        menu_w = min(config.SCREEN_WIDTH * 0.9, 400)
        line_height = self.fonts["info"].get_linesize()
        title_h = line_height
        options_h = 3 * (line_height + 5)
        padding = 40
        menu_h = title_h + options_h + padding

        # --- Centering and Drawing Box ---
        game_area_h = config.SCREEN_HEIGHT - config.INFO_PANEL_HEIGHT
        menu_rect = pygame.Rect(
            (config.SCREEN_WIDTH - menu_w) / 2,
            (game_area_h - menu_h) / 2,
            menu_w,
            menu_h,
        )
        if game.active_npc:
            self.draw_ui_box(menu_rect, f"{game.active_npc.name}와(과) 대화")

        # --- Drawing Options ---
        options = ["(고정된) 정보 확인", "자유롭게 대화하기", "떠나기 (ESC)"]
        y_start = menu_rect.y + title_h + 20

        for i, option in enumerate(options):
            prefix = "> " if i == game.menu_selection else "  "
            text_color = (
                config.PLAYER_COLOR if i == game.menu_selection else config.WHITE
            )
            option_text = f"{prefix}{i + 1}. {option}"

            available_width = menu_rect.width - 40
            if self.fonts["info"].size(option_text)[0] > available_width:
                while self.fonts["info"].size(option_text + "..")[0] > available_width:
                    option_text = option_text[:-1]
                option_text += ".."

            opt_surf = self.fonts["info"].render(option_text, True, text_color)
            y_pos = y_start + i * (line_height + 5)
            self.screen.blit(opt_surf, (menu_rect.x + 20, y_pos))

    def draw_text_input(self, game: Game) -> None:
        """Draws the text input interface for chatting or entering passwords."""
        if game.active_npc:  # Chat window
            game_area_h = config.SCREEN_HEIGHT - config.INFO_PANEL_HEIGHT
            ui_w = config.SCREEN_WIDTH * 0.9
            ui_h = int(game_area_h * 0.8)
            ui_x = (config.SCREEN_WIDTH - ui_w) / 2
            ui_y = (game_area_h - ui_h) / 2
            ui_rect = pygame.Rect(ui_x, ui_y, ui_w, ui_h)

            input_h = 100  # Keep user input height fixed
            chat_h = ui_h - input_h
            if chat_h < 0:
                chat_h = 0

            self.draw_ui_box(ui_rect, "대화하기 (ESC로 종료, 위/아래 화살표로 스크롤)")

            chat_area_width = ui_rect.width - 20
            y_offset = ui_rect.y + 40
            line_height = self.fonts["info"].get_linesize()

            all_lines = []
            for message in game.active_npc.chat_history:
                role = message["role"]
                content = message["content"]
                prefix = f"{game.active_npc.name}: " if role != 'user' else 'Player: '
                color = config.WHITE if role != 'user' else config.PLAYER_COLOR
                wrapped_lines = self._wrap_text(prefix + content, self.fonts["info"], chat_area_width)
                all_lines.extend([(line, color) for line in wrapped_lines])

            chat_box_height = chat_h - 40
            max_visible_lines = chat_box_height // line_height if line_height > 0 else 0

            max_scroll_offset = len(all_lines) - max_visible_lines
            if max_scroll_offset < 0:
                max_scroll_offset = 0
            if game.chat_scroll_offset > max_scroll_offset:
                game.chat_scroll_offset = max_scroll_offset
            if game.chat_scroll_offset < 0:
                game.chat_scroll_offset = 0

            start_index = len(all_lines) - max_visible_lines - game.chat_scroll_offset
            end_index = len(all_lines) - game.chat_scroll_offset
            if start_index < 0:
                start_index = 0
            visible_lines = all_lines[start_index:end_index]

            for line, color in visible_lines:
                if y_offset + line_height > ui_rect.y + chat_h:
                    break
                line_surf = self.fonts["info"].render(line, True, color)
                self.screen.blit(line_surf, (ui_rect.x + 10, y_offset))
                y_offset += line_height

            if len(all_lines) > max_visible_lines:
                if end_index < len(all_lines):
                    scroll_down_surf = self.fonts["info"].render("▼", True, config.WHITE)
                    self.screen.blit(scroll_down_surf, (ui_rect.right - 30, ui_rect.y + chat_h - 25))
                if game.chat_scroll_offset > 0:
                    scroll_up_surf = self.fonts["info"].render("▲", True, config.WHITE)
                    self.screen.blit(scroll_up_surf, (ui_rect.right - 30, ui_rect.y + 40))

            input_rect = pygame.Rect(ui_rect.x, ui_rect.y + chat_h, ui_rect.width, input_h)
            self.draw_ui_box(input_rect, "내용 입력 (Enter로 전송):")
            full_text = game.input_text + "|"
            available_width = input_rect.width - 20
            wrapped_lines = self._wrap_text(full_text, self.fonts["info"], available_width)

            y_offset = input_rect.y + 40
            for line in wrapped_lines:
                if y_offset + line_height > input_rect.bottom - 10:
                    break
                text_surf = self.fonts["info"].render(line, True, config.WHITE)
                self.screen.blit(text_surf, (input_rect.x + 10, y_offset))
                y_offset += line_height
        else:  # Password input
            ui_rect = pygame.Rect(100, 100, config.SCREEN_WIDTH - 200, 100)
            self.draw_ui_box(ui_rect, game.input_prompt)
            full_text = game.input_text + "|"

            # Wrap the password input text
            available_width = ui_rect.width - 20
            wrapped_lines = self._wrap_text(full_text, self.fonts["info"], available_width)
            
            y_offset = ui_rect.y + 40
            line_height = self.fonts["info"].get_linesize()
            for line in wrapped_lines:
                if y_offset + line_height > ui_rect.bottom - 10:
                    break
                text_surf = self.fonts["info"].render(line, True, config.WHITE)
                self.screen.blit(text_surf, (ui_rect.x + 10, y_offset))
                y_offset += line_height

    def draw_game_over(self, game: Game) -> None:
        """Draws the game over screen."""
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