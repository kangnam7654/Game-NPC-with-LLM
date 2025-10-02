import pygame

from configs import config


class UIManager:
    """게임의 모든 UI 오버레이(메뉴, 대화창 등)를 그리는 클래스."""

    def __init__(self, screen, fonts):
        self.screen = screen
        self.fonts = fonts

    def draw_ui_box(self, rect, title=""):
        pygame.draw.rect(self.screen, config.UI_BG_COLOR, rect)
        pygame.draw.rect(self.screen, config.UI_BORDER_COLOR, rect, 2)
        if title:
            title_surf = self.fonts["info"].render(title, True, config.WHITE)
            self.screen.blit(title_surf, (rect.x + 10, rect.y + 10))

    def draw_interaction_menu(self, game):
        menu_w, menu_h = 300, 150
        menu_rect = pygame.Rect(
            (config.SCREEN_WIDTH - menu_w) / 2,
            (config.SCREEN_HEIGHT - config.INFO_PANEL_HEIGHT - menu_h) / 2,
            menu_w,
            menu_h,
        )
        self.draw_ui_box(menu_rect, f"{game.active_npc['name']}와(과) 대화")

        options = ["(고정된) 정보 확인", "자연어 대화 시작", "떠나기 (ESC)"]
        for i, option in enumerate(options):
            prefix = "> " if i == game.menu_selection else "  "
            text_color = (
                config.PLAYER_COLOR if i == game.menu_selection else config.WHITE
            )
            option_text = f"{prefix}{i+1}. {option}"
            opt_surf = self.fonts["info"].render(option_text, True, text_color)
            self.screen.blit(opt_surf, (menu_rect.x + 20, menu_rect.y + 50 + i * 30))

    def draw_text_input(self, game):
        input_h, chat_h = 100, 200
        if game.active_npc:  # 채팅창
            ui_rect = pygame.Rect(50, 50, config.SCREEN_WIDTH - 100, chat_h + input_h)
            self.draw_ui_box(ui_rect, "대화하기 (ESC로 종료)")
            for i, line in enumerate(game.chat_history):
                line_surf = self.fonts["info"].render(line, True, config.WHITE)
                self.screen.blit(line_surf, (ui_rect.x + 10, ui_rect.y + 40 + i * 25))
            input_rect = pygame.Rect(
                ui_rect.x, ui_rect.y + chat_h, ui_rect.width, input_h
            )
            self.draw_ui_box(input_rect, "내용 입력 (Enter로 전송):")
            full_text = game.input_text + game.editing_text + "|"
            text_surf = self.fonts["info"].render(full_text, True, config.WHITE)
            self.screen.blit(text_surf, (input_rect.x + 10, input_rect.y + 40))
        else:  # 비밀번호 입력창
            ui_rect = pygame.Rect(100, 100, config.SCREEN_WIDTH - 200, input_h)
            self.draw_ui_box(ui_rect, game.input_prompt)
            full_text = game.input_text + game.editing_text + "|"
            text_surf = self.fonts["info"].render(full_text, True, config.WHITE)
            self.screen.blit(text_surf, text_surf.get_rect(center=ui_rect.center))

    def draw_game_over(self, game):
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
