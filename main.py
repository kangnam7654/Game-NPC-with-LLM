import pygame
from transformers import AutoModelForCausalLM, AutoTokenizer

from clients.hf_wrapper import HuggingFaceWrapper
from configs import config
from controllers.input_handler import InputHandler
from controllers.interaction_handler import InteractionHandler
from games.game import Game
from renderers.renderer import Renderer


def main() -> None:
    """Initializes and runs the main game loop."""
    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL_NAME, torch_dtype="bfloat16", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME)
    llm_client = HuggingFaceWrapper(model, tokenizer)

    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("절차적 퀘스트 미로")  # Procedural Quest Maze
    clock = pygame.time.Clock()

    # Create core components
    game = Game(llm_client=llm_client)
    renderer = Renderer(screen)
    interaction_handler = InteractionHandler(game)
    input_handler = InputHandler(game, interaction_handler)

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
