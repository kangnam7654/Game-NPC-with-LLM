import random


def random_coordinate(width, height, margin=0):
    """
    Generate a random coordinate within the specified width and height,
    ensuring that the coordinate is at least 'margin' pixels away from the edges.

    :param width: The width of the area.
    :param height: The height of the area.
    :param margin: The minimum distance from the edges.
    :return: A tuple (x, y) representing the random coordinate.
    """
    x = random.randint(margin, width - margin)
    y = random.randint(margin, height - margin)
    return x, y
