class NPC:
    """Represents a non-player character (NPC) in the game.

    Attributes:
        name (str): The name of the NPC.
        pos (tuple[int, int]): The (x, y) grid position of the NPC.
        color (tuple[int, int, int]): The RGB color of the NPC.
        label (str): The single-character label to display for the NPC.
    """

    def __init__(
        self, name: str, pos: tuple[int, int], color: tuple[int, int, int], label: str
    ) -> None:
        """Initializes the NPC.

        Args:
            name (str): The name of the NPC.
            pos (tuple[int, int]): The grid position of the NPC.
            color (tuple[int, int, int]): The color of the NPC.
            label (str): The label for the NPC.
        """
        self.name: str = name
        self.pos: tuple[int, int] = pos
        self.color: tuple[int, int, int] = color
        self.label: str = label