import pygame

from configs import config
from controllers.input_handler import InputHandler
from games.game import Game
from renderers.renderer import Renderer


def main():
    """메인 게임 함수"""
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("절차적 퀘스트 미로")
    clock = pygame.time.Clock()

    # 핵심 컴포넌트들 생성
    game = Game()
    renderer = Renderer(screen)
    input_handler = InputHandler(game)

    running = True
    while running:
        # 1. 입력 처리
        # handle_events가 False를 반환하면 (예: 창 닫기) 루프 종료
        running = input_handler.handle_events()

        # 2. 게임 상태 업데이트 (현재는 입력에 의해서만 상태가 변하므로 비워둠)
        # 예: update() 메서드를 만들어 적이 움직이게 하는 등의 로직 추가 가능

        # 3. 화면 그리기
        renderer.draw(game)

        # 4. FPS 조절
        clock.tick(config.FPS)

    pygame.quit()


if __name__ == "__main__":
    main()