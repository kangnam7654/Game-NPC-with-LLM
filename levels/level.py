import pygame

from actors.character import Character


class Level:
    def __init__(self):
        self.display_surface = pygame.display.get_surface()

        self.visible_sprites = pygame.sprite.Group()
        # self.obstacle_sprites = pygame.sprite.Group()
        self.worker1 = Character((100, 120), color=(30, 100, 220))
        self.worker2 = Character((500, 120), color=(220, 30, 100))

        self.visible_sprites.add(self.worker1)
        self.visible_sprites.add(self.worker2)

    def run(self, dt):
        self.visible_sprites.update(dt)
        self.visible_sprites.draw(self.display_surface)
