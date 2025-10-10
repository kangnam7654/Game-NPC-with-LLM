import os

import pygame

from configs import config
from game.games.game import Game
from game.games.states import GameState
from game.ui.manager import UIManager


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
        font_paths = ["game/fonts/NanumGothic.ttf"]
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
        """Loads the fonts required for the game, scaling them based on the final grid size."""
        font_path = self._get_korean_font()

        # --- Font Scaling Logic ---
        # Define a reference grid size that corresponds to the base font sizes.
        REFERENCE_GRID_SIZE = 24
        scaling_factor = config.GRID_SIZE / REFERENCE_GRID_SIZE

        # Calculate scaled font sizes, ensuring a minimum size of 1.
        main_size = max(1, int(config.BASE_FONT_MAIN_SIZE * scaling_factor))
        info_size = max(1, int(config.BASE_FONT_INFO_SIZE * scaling_factor))
        label_size = max(1, int(config.BASE_FONT_LABEL_SIZE * scaling_factor))
        # --- End of Scaling Logic ---

        return {
            "main": pygame.font.Font(font_path, main_size),
            "info": pygame.font.Font(font_path, info_size),
            "label": pygame.font.Font(font_path, label_size),
        }

    def _load_sprites(self) -> dict[str, pygame.Surface]:
        """Loads all sprite images and scales them to the grid size."""
        sprite_paths = {
            "player": "game/sprites/player.png",
            "wall": "game/sprites/wall.png",
            "floor": "game/sprites/floor.png",
            "exit": "game/sprites/exit.png",
            "treasure": "game/sprites/treasure.png",
            "treasure_open": "game/sprites/treasure_open.png",
            "npc_loc": "game/sprites/npc_loc.png",
            "npc_pw": "game/sprites/npc_pw.png",
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
            label_surf = self.fonts["label"].render(npc.name, True, config.WHITE)
            label_rect = label_surf.get_rect(center=(npc_pos_pixels[0] + config.GRID_SIZE // 2, npc_pos_pixels[1] - 10))
            self.screen.blit(label_surf, label_rect)

        # Draw player
        player_pos_pixels = (
            game.player_pos[0] * config.GRID_SIZE,
            game.player_pos[1] * config.GRID_SIZE,
        )
        self.screen.blit(self.sprites["player"], player_pos_pixels)
        label_surf = self.fonts["label"].render("나", True, config.WHITE)
        label_rect = label_surf.get_rect(center=(player_pos_pixels[0] + config.GRID_SIZE // 2, player_pos_pixels[1] - 10))
        self.screen.blit(label_surf, label_rect)

        # --- Information Panel (with text wrapping) ---
        info_panel_rect = pygame.Rect(
            0,
            config.GRID_HEIGHT * config.GRID_SIZE,
            config.SCREEN_WIDTH,
            config.INFO_PANEL_HEIGHT,
        )
        pygame.draw.rect(self.screen, config.GRAY, info_panel_rect)

        line_height = self.fonts["info"].get_linesize()
        y_offset = info_panel_rect.y + 5
        max_width = config.SCREEN_WIDTH - 20 # Padding

        # Wrap and draw Objective
        for line in self.ui_manager._wrap_text(game.objective, self.fonts["info"], max_width):
            if y_offset + line_height > info_panel_rect.bottom: break
            line_surf = self.fonts["info"].render(line, True, config.WHITE)
            self.screen.blit(line_surf, (10, y_offset))
            y_offset += line_height
        
        y_offset += 3 # Spacing

        # Wrap and draw Dialogue
        dialogue_text = f"정보: {game.dialogue}"
        for line in self.ui_manager._wrap_text(dialogue_text, self.fonts["info"], max_width):
            if y_offset + line_height > info_panel_rect.bottom: break
            line_surf = self.fonts["info"].render(line, True, config.WHITE)
            self.screen.blit(line_surf, (10, y_offset))
            y_offset += line_height

        y_offset += 3 # Spacing

        # Draw Coords and Status (on one line if possible)
        coords = f"좌표: ({game.player_pos[0]}, {game.player_pos[1]})"
        status = f"위치: {'O' if game.knows_location else 'X'} | 암호: {'O' if game.knows_password else 'X'} | 상자: {'O' if game.treasure_opened else 'X'}"
        
        coords_surf = self.fonts["info"].render(coords, True, config.WHITE)
        status_surf = self.fonts["info"].render(status, True, config.WHITE)

        # Check if they fit on one line
        if 10 + coords_surf.get_width() + 10 + status_surf.get_width() < config.SCREEN_WIDTH:
            if y_offset + line_height <= info_panel_rect.bottom:
                self.screen.blit(coords_surf, (10, y_offset))
                self.screen.blit(status_surf, (10 + coords_surf.get_width() + 10, y_offset))
        else: # Draw them on separate lines
            if y_offset + line_height <= info_panel_rect.bottom:
                self.screen.blit(coords_surf, (10, y_offset))
                y_offset += line_height
            if y_offset + line_height <= info_panel_rect.bottom:
                self.screen.blit(status_surf, (10, y_offset))

        # UI Overlays (remains the same)
        if game.state == GameState.INTERACTION_MENU:
            self.ui_manager.draw_interaction_menu(game)
        elif game.state == GameState.TEXT_INPUT:
            self.ui_manager.draw_text_input(game)
        elif game.state == GameState.GAME_OVER:
            self.ui_manager.draw_game_over(game)

        pygame.display.flip()