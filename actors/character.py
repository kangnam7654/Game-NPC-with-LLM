import pygame


class Character(pygame.sprite.Sprite):
    def __init__(
        self, pos, size=(50, 50), color=(0, 0, 0), controller=None, obstacles=None
    ):
        super().__init__()
        self.image = pygame.Surface(size)
        self.image.fill(color)

        self.obstacles = obstacles

        self.rect = self.image.get_rect(topleft=pos)
        self._hitbox_factor = 0.1
        self.hitbox = self.rect.inflate(
            (-int(size[0] * self._hitbox_factor), -int(size[1] * self._hitbox_factor))
        )

        self.pos = pygame.Vector2(self.rect.topleft)
        self.direction = pygame.Vector2()
        self.speed = 50

        self.controller = controller

    def _keyboard_controller(self):
        keys = pygame.key.get_pressed()

        x = (keys[pygame.K_d] or keys[pygame.K_RIGHT]) - (
            keys[pygame.K_a] or keys[pygame.K_LEFT]
        )
        y = (keys[pygame.K_s] or keys[pygame.K_DOWN]) - (
            keys[pygame.K_w] or keys[pygame.K_UP]
        )
        return x, y

    def collision(self, axis):
        # No obstacles to check against
        if not self.obstacles:
            return

        for obstacle in self.obstacles:
            if obstacle is self:  # Check if the obstacle is the character itself
                continue

            obstacle_hitbox = getattr(obstacle, "hitbox", obstacle.rect)
            if self.hitbox.colliderect(obstacle_hitbox):
                if axis == "horizontal":
                    if self.direction.x > 0:  # Moving Right
                        self.hitbox.right = obstacle_hitbox.left
                    elif self.direction.x < 0:  # Moving Left
                        self.hitbox.left = obstacle_hitbox.right

                    self.rect.centerx = self.hitbox.centerx
                    self.pos.x = float(self.rect.x)

                elif axis == "vertical":
                    if self.direction.y > 0:  # Moving Down
                        self.hitbox.bottom = obstacle_hitbox.top
                    elif self.direction.y < 0:  # Moving Up
                        self.hitbox.top = obstacle_hitbox.bottom
                    
                    self.rect.centery = self.hitbox.centery
                    self.pos.y = float(self.rect.y)

    def input(self):
        if self.controller:
            x, y = self.controller()
        else:
            x, y = self._keyboard_controller()
        self.direction.update(x, y)

        if self.direction.length_squared() > 0:
            self.direction.normalize_ip()

    def move(self, dt=1.0):
        if self.direction.length_squared() > 0:
            self.pos += self.direction * self.speed * dt
            self.rect.topleft = (int(self.pos.x), int(self.pos.y))

    def update(self, dt=1.0):
        self.input()
        self.move(dt)
