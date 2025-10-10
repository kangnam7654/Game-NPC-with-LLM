import pygame
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.clients.hf_wrapper import HuggingFaceWrapper
from llm.clients.ollama_client import OllamaClient
from configs import config
from game.controllers.input_handler import InputHandler
from game.controllers.interaction_handler import InteractionHandler
from game.games.game import Game
from game.renderers.renderer import Renderer


def main() -> None:
    """Initializes and runs the main game loop."""
    # --- Dynamic Grid & Screen Size Calculation ---
    # Check if the configured grid dimensions are smaller than the minimum window size.
    # If so, calculate a new GRID_SIZE to scale up the grid to fill the minimum window.

    # Calculate screen dimensions based on default config
    default_screen_w = config.GRID_WIDTH * config.GRID_SIZE
    default_screen_h = config.GRID_HEIGHT * config.GRID_SIZE + config.INFO_PANEL_HEIGHT

    # Check if we need to scale up
    if default_screen_w < config.MIN_WINDOW_WIDTH or default_screen_h < config.MIN_WINDOW_HEIGHT:
        # Calculate the GRID_SIZE required to meet the minimums
        size_from_w = config.MIN_WINDOW_WIDTH // config.GRID_WIDTH
        size_from_h = (
            config.MIN_WINDOW_HEIGHT - config.INFO_PANEL_HEIGHT
        ) // config.GRID_HEIGHT

        # To keep cells square, use the smaller of the two calculated sizes.
        # This ensures the grid fits within the minimum window dimensions.
        new_grid_size = min(size_from_w, size_from_h)

        # Update the config dynamically. All other modules will now use this larger GRID_SIZE.
        config.GRID_SIZE = new_grid_size

    # Recalculate final screen dimensions with the (potentially new) GRID_SIZE
    config.SCREEN_WIDTH = config.GRID_WIDTH * config.GRID_SIZE
    config.SCREEN_HEIGHT = (
        config.GRID_HEIGHT * config.GRID_SIZE + config.INFO_PANEL_HEIGHT
    )
    # --- End of Dynamic Calculation ---

    # model = AutoModelForCausalLM.from_pretrained(
    #     config.LLM_MODEL_NAME, torch_dtype="bfloat16", device_map="auto"
    # )
    # tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME)
    # llm_client = HuggingFaceWrapper(model, tokenizer)

    llm_client = OllamaClient(model="gpt-oss:20b")
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