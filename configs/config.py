"""Configuration file for game settings and constants."""

# --- Game Settings (Adjustable) ---
MIN_WINDOW_WIDTH: int = 800
MIN_WINDOW_HEIGHT: int = 640
GRID_WIDTH: int = 7  # Maze width
GRID_HEIGHT: int = 7  # Maze height
# GRID_WIDTH: int = 9  # Maze width
# GRID_HEIGHT: int = 9  # Maze height
GRID_SIZE: int = 30  # Default pixel size of each cell (can be scaled up)
WALL_DENSITY: float = 0  # Wall density (higher value means narrower paths)

INFO_PANEL_HEIGHT: int = 180  # Height of the bottom info panel
SCREEN_WIDTH: int = GRID_WIDTH * GRID_SIZE
SCREEN_HEIGHT: int = GRID_HEIGHT * GRID_SIZE + INFO_PANEL_HEIGHT
FPS: int = 30

# --- Color Definitions ---
Color = tuple[int, int, int]

WHITE: Color = (255, 255, 255)
BLACK: Color = (0, 0, 0)
GRAY: Color = (40, 40, 40)
PLAYER_COLOR: Color = (0, 150, 255)
WALL_COLOR: Color = (80, 80, 120)
PATH_COLOR: Color = (20, 20, 40)
EXIT_COLOR: Color = (255, 200, 0)
TREASURE_COLOR: Color = (255, 180, 50)
TREASURE_OPEN_COLOR: Color = (150, 220, 255)
NPC_LOC_COLOR: Color = (200, 0, 200)
NPC_PW_COLOR: Color = (0, 200, 200)
LABEL_TEXT_COLOR: Color = (255, 255, 255)
UI_BG_COLOR: Color = (10, 10, 30)
UI_BORDER_COLOR: Color = (150, 150, 200)


# --- AI/LLM Settings ---
# LLM_MODEL_NAME: str = "LGAI-EXAONE/EXAONE-4.0-1.2B"
LLM_MODEL_NAME: str = "LGAI-EXAONE/EXAONE-4.0-32B-AWQ"

# --- Prompt File Paths ---
NPC_LOC_PROMPT_PATH: str = "llm/prompts/npc_location.md"
NPC_PW_PROMPT_PATH: str = "llm/prompts/npc_password.md"


# --- Font Settings ---
BASE_FONT_MAIN_SIZE: int = 30
BASE_FONT_INFO_SIZE: int = 6
BASE_FONT_LABEL_SIZE: int = 11

MAX_STEPS_PER_EPISODE: int = 200  # Maximum steps per episode
