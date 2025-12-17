import pygame
import os

class Bullet():
    def __init__(self, rocket_x, rocket_y):
        self.bullet = pygame.image.load(os.path.join(".", "src", "env", "assets",'bullet.png'))
        self.bullet = pygame.transform.scale(self.bullet, (30, 40))
        
        self.bullet_x = rocket_x
        self.bullet_y = rocket_y
        
        self.moveX_bullet = 0
        self.moveY_bullet = -4
                
        self.bullet_fired = False
        
    def update_bullet(self, screen):
        if self.bullet_fired:
            self.bullet_y  += self.moveY_bullet
            screen.blit(self.bullet, (self.bullet_x, self.bullet_y))  
         
        if self.bullet_y <= 0:
            self.bullet_y = 480
            self.bullet_fired = False