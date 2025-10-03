from enum import Enum, auto


class GameState(Enum):
    """Represents the different possible states of the game."""

    PLAYING = auto()
    INTERACTION_MENU = auto()
    TEXT_INPUT = auto()
    GAME_OVER = auto()