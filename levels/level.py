import random

import pygame

from actors.character import Character


def controller1():
    return 1, 0


def controller2():
    return -1, 0


class Level:
    def __init__(self):
        self.display_surface = pygame.display.get_surface()

        self.visible_sprites = pygame.sprite.Group()
        self.obstacle_sprites = pygame.sprite.Group()

        self.worker1 = Character(
            (100, 120),
            color=(30, 100, 220),
            controller=controller2,
            obstacles=self.obstacle_sprites,
        )
        self.worker2 = Character(
            (150, 120),
            color=(220, 30, 100),
            controller=controller2,
            obstacles=self.obstacle_sprites,
        )

        self.visible_sprites.add(self.worker1)
        self.visible_sprites.add(self.worker2)

        self.obstacle_sprites.add(self.worker1)
        self.obstacle_sprites.add(self.worker2)

    def run(self, dt):
        self.visible_sprites.update(dt)
        self.visible_sprites.draw(self.display_surface)
