import random

from configs import config


def generate_connected_map():
    """'Drunkard's Walk' 알고리즘으로 모든 길이 연결된 맵을 생성합니다."""
    grid = [[1] * config.GRID_WIDTH for _ in range(config.GRID_HEIGHT)]  # 1: 벽

    total_internal_cells = (config.GRID_WIDTH - 2) * (config.GRID_HEIGHT - 2)
    target_floor_count = int(total_internal_cells * (1 - config.WALL_DENSITY))

    cx = random.randint(1, config.GRID_WIDTH - 2)
    cy = random.randint(1, config.GRID_HEIGHT - 2)
    grid[cy][cx] = 0  # 0: 길
    floor_count = 1

    while floor_count < target_floor_count:
        dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        nx, ny = cx + dx, cy + dy

        if 1 <= nx < config.GRID_WIDTH - 1 and 1 <= ny < config.GRID_HEIGHT - 1:
            cx, cy = nx, ny
            if grid[cy][cx] == 1:
                grid[cy][cx] = 0
                floor_count += 1

    return grid
