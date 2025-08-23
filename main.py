import sys

import pygame

from configs import FPS, HEIGHT, WIDTH
from levels.level import Level


class Simulator:
    def __init__(self):

        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.level = Level()

    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0  # Convert milliseconds to seconds
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.screen.fill((255, 255, 255))
            self.level.run(dt)
            pygame.display.flip()


if __name__ == "__main__":
    simulator = Simulator()
    simulator.run()
