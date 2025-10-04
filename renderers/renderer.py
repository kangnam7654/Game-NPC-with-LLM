import os

import pygame

from configs import config
from games.game import Game
from games.states import GameState
from ui.manager import UIManager


class Renderer:
    """Handles drawing all graphical elements of the game to the screen."""

    def __init__(self, screen: pygame.Surface) -> None:
        """Initializes the Renderer."""
        self.screen: pygame.Surface = screen
        self.fonts: dict[str, pygame.font.Font] = self._load_fonts()
        self.ui_manager: UIManager = UIManager(self.screen, self.fonts)
        self.sprites: dict[str, pygame.Surface] = self._load_sprites()

    def _get_korean_font(self) -> str | None:
        """Finds an available Korean font on the system."""
        font_paths = ["fonts/NanumGothic.ttf"]
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
        """Loads the fonts required for the game."""
        font_path = self._get_korean_font()
        return {
            "main": pygame.font.Font(font_path, 36),
            "info": pygame.font.Font(font_path, 20),
            "label": pygame.font.Font(font_path, 18),
        }

    def _load_sprites(self) -> dict[str, pygame.Surface]:
        """Loads all sprite images and scales them to the grid size."""
        sprite_paths = {
            "player": "sprites/player.png",
            "wall": "sprites/wall.png",
            "floor": "sprites/floor.png",
            "exit": "sprites/exit.png",
            "treasure": "sprites/treasure.png",
            "treasure_open": "sprites/treasure_open.png",
            "npc_loc": "sprites/npc_loc.png",
            "npc_pw": "sprites/npc_pw.png",
        }
        loaded_sprites = {}
        for name, path in sprite_paths.items():
            try:
                sprite = pygame.image.load(path).convert_alpha()
                loaded_sprites[name] = pygame.transform.scale(
                    sprite, (config.GRID_SIZE, config.GRID_SIZE)
                )
            except pygame.error:
                print(f"[경고] 스프라이트 파일을 찾을 수 없습니다: {path}")
                temp_surface = pygame.Surface((config.GRID_SIZE, config.GRID_SIZE))
                temp_surface.fill(config.GRAY)  # Use a visible placeholder color
                loaded_sprites[name] = temp_surface
        return loaded_sprites

    def draw(self, game: Game) -> None:
        """Draws the entire game screen using sprites."""
        self.screen.fill(config.BLACK)

        # Draw map (floor and walls)
        for r in range(config.GRID_HEIGHT):
            for c in range(config.GRID_WIDTH):
                pos_pixels = (c * config.GRID_SIZE, r * config.GRID_SIZE)
                sprite_name = "wall" if game.grid[r][c] == 1 else "floor"
                self.screen.blit(self.sprites[sprite_name], pos_pixels)

        # Draw objects
        exit_pos_pixels = (game.exit_pos[0] * config.GRID_SIZE, game.exit_pos[1] * config.GRID_SIZE)
        self.screen.blit(self.sprites["exit"], exit_pos_pixels)

        if game.treasure_visible:
            treasure_pos_pixels = (
                game.treasure_pos[0] * config.GRID_SIZE,
                game.treasure_pos[1] * config.GRID_SIZE,
            )
            sprite_name = "treasure_open" if game.treasure_opened else "treasure"
            self.screen.blit(self.sprites[sprite_name], treasure_pos_pixels)

        # Draw NPCs
        for npc in game.npcs:
            npc_pos_pixels = (npc.pos[0] * config.GRID_SIZE, npc.pos[1] * config.GRID_SIZE)
            sprite_name = "npc_loc" if "위치" in npc.name else "npc_pw"
            self.screen.blit(self.sprites[sprite_name], npc_pos_pixels)

        # Draw player
        player_pos_pixels = (
            game.player_pos[0] * config.GRID_SIZE,
            game.player_pos[1] * config.GRID_SIZE,
        )
        self.screen.blit(self.sprites["player"], player_pos_pixels)

        # Information Panel (remains the same)
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

        # UI Overlays (remains the same)
        if game.state == GameState.INTERACTION_MENU:
            self.ui_manager.draw_interaction_menu(game)
        elif game.state == GameState.TEXT_INPUT:
            self.ui_manager.draw_text_input(game)
        elif game.state == GameState.GAME_OVER:
            self.ui_manager.draw_game_over(game)

        pygame.display.flip()
