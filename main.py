import sys

import pygame

from configs import FPS, HEIGHT, WIDTH
from levels.level import Level


class Simulator:
    def __init__(self):

        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.screen.fill((0, 0, 0))
            pygame.display.flip()
            self.clock.tick(FPS)


if __name__ == "__main__":
    simulator = Simulator()
    simulator.run()
