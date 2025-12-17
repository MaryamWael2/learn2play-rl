import pygame
import random
import math
import os

class UFO:
    def __init__(self, screen_width):
        self.screen_width = screen_width
        self.ufo = pygame.image.load(os.path.join(".", "src", "env", "assets", 'ufo.png'))
        self.ufo = pygame.transform.scale(self.ufo, (70, 70))
        self.ufo_x = random.randint(0, screen_width-70)
        self.ufo_y = random.randint(50, 250)
        self.moveX_ufo = random.uniform(0.1, 2)
        self.moveY_ufo = 20
    
    def update_ufo(self, screen):
        self.ufo_x += self.moveX_ufo
        if self.ufo_x <= 0:
            self.moveX_ufo = abs(self.moveX_ufo)
            self.ufo_y += self.moveY_ufo
        elif self.ufo_x >= self.screen_width-70:
            self.moveX_ufo = -abs(self.moveX_ufo)
            self.ufo_y += self.moveY_ufo
        screen.blit(self.ufo, (self.ufo_x, self.ufo_y))
        
    def is_ufo_killed(self, bullet_x, bullet_y):
        distance = math.sqrt(math.pow(self.ufo_x - bullet_x, 2) + (math.pow(self.ufo_y - bullet_y, 2)))
        if distance < 40:
            return True
        else:
            return False      