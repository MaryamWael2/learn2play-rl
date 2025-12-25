import pygame
import os
import math

class UFOBullet:
    def __init__(self, x, y, speed=10):
        self.x = x
        self.y = y
        self.speed = speed
        self.active = True
        self.rock = pygame.image.load(os.path.join(".", "src", "env", "assets", 'stone.png'))
        self.rock = pygame.transform.scale(self.rock, (30, 30))

    def update(self, screen):
        self.y += self.speed
        screen.blit(self.rock, (self.x, self.y))

    def hit_rocket(self, rocket_x, rocket_y):
        sx = self.x + 15
        sy = self.y + 15
        rx = rocket_x + 35
        ry = rocket_y + 35

        dx = sx - rx
        dy = sy - ry
        return dx*dx + dy*dy < 35*35

