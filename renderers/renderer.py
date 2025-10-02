import os

import pygame

from configs import config
from ui.manager import UIManager


class Renderer:
    """게임의 모든 그래픽 요소를 화면에 그리는 클래스."""

    def __init__(self, screen):
        self.screen = screen
        self.fonts = self._load_fonts()
        self.ui_manager = UIManager(self.screen, self.fonts)

    def _get_korean_font(self):
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
                return pygame.font.match_font(font_name)
            except Exception:
                continue
        print("\n[경고] 한글 폰트를 찾지 못했습니다. 한글이 깨질 수 있습니다.\n")
        return None

    def _load_fonts(self):
        font_path = self._get_korean_font()
        return {
            "main": pygame.font.Font(font_path, 36),
            "info": pygame.font.Font(font_path, 20),
            "label": pygame.font.Font(font_path, 18),
        }

    def _draw_labeled_rect(self, pos, color, label):
        rect = pygame.Rect(
            pos[0] * config.GRID_SIZE,
            pos[1] * config.GRID_SIZE,
            config.GRID_SIZE,
            config.GRID_SIZE,
        )
        pygame.draw.rect(self.screen, color, rect)
        surf = self.fonts["label"].render(label, True, config.LABEL_TEXT_COLOR)
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    def draw(self, game):
        self.screen.fill(config.BLACK)
        # 맵 그리기
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

        # 객체 그리기
        self._draw_labeled_rect(game.exit_pos, config.EXIT_COLOR, "E")
        for npc in game.npcs:
            self._draw_labeled_rect(npc["pos"], npc["color"], npc["label"])
        if game.treasure_visible:
            color = (
                config.TREASURE_OPEN_COLOR
                if game.treasure_opened
                else config.TREASURE_COLOR
            )
            self._draw_labeled_rect(game.treasure_pos, color, "T")

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

        # 정보 패널
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

        # UI 오버레이
        if game.state == "interaction_menu":
            self.ui_manager.draw_interaction_menu(game)
        elif game.state == "text_input":
            self.ui_manager.draw_text_input(game)
        elif game.state == "game_over":
            self.ui_manager.draw_game_over(game)

        pygame.display.flip()
