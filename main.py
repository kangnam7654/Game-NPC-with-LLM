import pygame

from configs import config
from controllers.input_handler import InputHandler
from games.game import Game
from renderers.renderer import Renderer


def main() -> None:
    """Initializes and runs the main game loop."""
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("절차적 퀘스트 미로")  # Procedural Quest Maze
    clock = pygame.time.Clock()

    # Create core components
    game = Game()
    renderer = Renderer(screen)
    input_handler = InputHandler(game)

    running = True
    while running:
        # 1. Handle input
        # The loop terminates if handle_events returns False (e.g., closing the window)
        running = input_handler.handle_events()

        # 2. Update game state (currently only changes on input, so this is empty)
        # e.g., An update() method could be added here for things like enemy movement.

        # 3. Draw the screen
        renderer.draw(game)

        # 4. Control FPS
        clock.tick(config.FPS)

    pygame.quit()


if __name__ == "__main__":
    main()