import random

from configs import config


def generate_connected_map() -> list[list[int]]:
    """Generates a connected map using the 'Drunkard's Walk' algorithm.

    This algorithm starts from a random point and carves out a path by moving
    randomly, ensuring all paths are connected.

    Returns:
        list[list[int]]: A 2D list representing the map grid, where 0 is a path
                         and 1 is a wall.
    """
    grid: list[list[int]] = [
        [1] * config.GRID_WIDTH for _ in range(config.GRID_HEIGHT)
    ]  # 1: Wall

    total_internal_cells = (config.GRID_WIDTH - 2) * (config.GRID_HEIGHT - 2)
    target_floor_count = int(total_internal_cells * (1 - config.WALL_DENSITY))

    # Start walker at a random position
    cx = random.randint(1, config.GRID_WIDTH - 2)
    cy = random.randint(1, config.GRID_HEIGHT - 2)
    grid[cy][cx] = 0  # 0: Path
    floor_count = 1

    while floor_count < target_floor_count:
        # Choose a random direction
        dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        nx, ny = cx + dx, cy + dy

        # Stay within the inner bounds of the map
        if 1 <= nx < config.GRID_WIDTH - 1 and 1 <= ny < config.GRID_HEIGHT - 1:
            cx, cy = nx, ny
            if grid[cy][cx] == 1:  # If it's a wall, carve a path
                grid[cy][cx] = 0
                floor_count += 1

    return grid
