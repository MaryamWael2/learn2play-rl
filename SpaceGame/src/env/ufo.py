import pygame
import math
import os

class UFO:
    def __init__(self, ufo_x, ufo_y):
        self.ufo = pygame.image.load(os.path.join(".", "src", "env", "assets", 'ufo.png'))
        self.ufo = pygame.transform.scale(self.ufo, (70, 70))
        self.ufo_x = ufo_x
        self.ufo_y = ufo_y

    def draw(self, screen):
        screen.blit(self.ufo, (self.ufo_x, self.ufo_y))
    
    def is_collision(self, bullet_x, bullet_y):
        sx = self.ufo_x + 35
        sy = self.ufo_y + 35
        rx = bullet_x + 15
        ry = bullet_y + 20

        dx = sx - rx
        dy = sy - ry
        return dx*dx + dy*dy < 35*35
