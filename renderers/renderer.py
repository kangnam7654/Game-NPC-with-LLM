import os

import pygame

from configs import config
from games.game import Game
from games.states import GameState
from ui.manager import UIManager


class Renderer:
    """Handles drawing all graphical elements of the game to the screen.

    Attributes:
        screen (pygame.Surface): The main screen surface to draw on.
        fonts (dict[str, pygame.font.Font]): A dictionary of pre-loaded fonts.
        ui_manager (UIManager): An instance of the UIManager for drawing UI overlays.
    """

    def __init__(self, screen: pygame.Surface) -> None:
        """Initializes the Renderer.

        Args:
            screen (pygame.Surface): The main screen surface.
        """
        self.screen: pygame.Surface = screen
        self.fonts: dict[str, pygame.font.Font] = self._load_fonts()
        self.ui_manager: UIManager = UIManager(self.screen, self.fonts)

    def _get_korean_font(self) -> str | None:
        """Finds an available Korean font on the system.

        Returns:
            str | None: The path to a found font file, or None if not found.
        """
        font_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        ]
        for path in font_paths:
            if os.path.exists(path):
                return path
        fonts = ["nanumgothic", "malgungothic", "gulim", "dotum", "applegothic"]
        for font_name in fonts:
            try:
                font_path = pygame.font.match_font(font_name)
                if font_path:
                    return font_path
            except Exception:
                continue
        print("\n[경고] 한글 폰트를 찾지 못했습니다. 한글이 깨질 수 있습니다.\n")
        return None

    def _load_fonts(self) -> dict[str, pygame.font.Font]:
        """Loads the fonts required for the game.

        Returns:
            dict[str, pygame.font.Font]: A dictionary mapping font names to Font objects.
        """
        font_path = self._get_korean_font()
        return {
            "main": pygame.font.Font(font_path, 36),
            "info": pygame.font.Font(font_path, 20),
            "label": pygame.font.Font(font_path, 18),
        }

    def _draw_labeled_rect(
        self, pos: tuple[int, int], color: tuple[int, int, int], label: str
    ) -> None:
        """Draws a labeled rectangle on the grid.

        Args:
            pos (tuple[int, int]): The grid position (x, y) of the rectangle.
            color (tuple[int, int, int]): The color of the rectangle.
            label (str): The text label to display in the center of the rectangle.
        """
        rect = pygame.Rect(
            pos[0] * config.GRID_SIZE,
            pos[1] * config.GRID_SIZE,
            config.GRID_SIZE,
            config.GRID_SIZE,
        )
        pygame.draw.rect(self.screen, color, rect)
        surf = self.fonts["label"].render(label, True, config.LABEL_TEXT_COLOR)
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    def draw(self, game: Game) -> None:
        """Draws the entire game screen.

        Args:
            game (Game): The main game object containing the current game state.
        """
        self.screen.fill(config.BLACK)
        # Draw map
        for r in range(config.GRID_HEIGHT):
            for c in range(config.GRID_WIDTH):
                color = config.WALL_COLOR if game.grid[r][c] == 1 else config.PATH_COLOR
                pygame.draw.rect(
                    self.screen,
                    color,
                    (
                        c * config.GRID_SIZE,
                        r * config.GRID_SIZE,
                        config.GRID_SIZE,
                        config.GRID_SIZE,
                    ),
                )

        # Draw objects
        self._draw_labeled_rect(game.exit_pos, config.EXIT_COLOR, "E")
        for npc in game.npcs:
            self._draw_labeled_rect(npc.pos, npc.color, npc.label)
        if game.treasure_visible:
            color =
                (
                    config.TREASURE_OPEN_COLOR
                    if game.treasure_opened
                    else config.TREASURE_COLOR
                )
            self._draw_labeled_rect(game.treasure_pos, color, "T")

        # Draw player
        pygame.draw.rect(
            self.screen,
            config.PLAYER_COLOR,
            (
                game.player_pos[0] * config.GRID_SIZE,
                game.player_pos[1] * config.GRID_SIZE,
                config.GRID_SIZE,
                config.GRID_SIZE,
            ),
        )

        # Information Panel
        info_panel = pygame.Rect(
            0,
            config.GRID_HEIGHT * config.GRID_SIZE,
            config.SCREEN_WIDTH,
            config.INFO_PANEL_HEIGHT,
        )
        pygame.draw.rect(self.screen, config.GRAY, info_panel)
        self.screen.blit(
            self.fonts["info"].render(game.objective, True, config.WHITE),
            (10, config.GRID_HEIGHT * config.GRID_SIZE + 10),
        )
        self.screen.blit(
            self.fonts["info"].render(f"정보: {game.dialogue}", True, config.WHITE),
            (10, config.GRID_HEIGHT * config.GRID_SIZE + 40),
        )
        coords = f"좌표: ({game.player_pos[0]}, {game.player_pos[1]})"
        status = f"위치: {'O' if game.knows_location else 'X'} | 암호: {'O' if game.knows_password else 'X'} | 상자: {'O' if game.treasure_opened else 'X'}"
        self.screen.blit(
            self.fonts["info"].render(coords, True, config.WHITE),
            (10, config.GRID_HEIGHT * config.GRID_SIZE + 70),
        )
        self.screen.blit(
            self.fonts["info"].render(status, True, config.WHITE),
            (200, config.GRID_HEIGHT * config.GRID_SIZE + 70),
        )

        # UI Overlays
        if game.state == GameState.INTERACTION_MENU:
            self.ui_manager.draw_interaction_menu(game)
        elif game.state == GameState.TEXT_INPUT:
            self.ui_manager.draw_text_input(game)
        elif game.state == GameState.GAME_OVER:
            self.ui_manager.draw_game_over(game)

        pygame.display.flip()
