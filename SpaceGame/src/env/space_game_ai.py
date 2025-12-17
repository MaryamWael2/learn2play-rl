import pygame
import random
import numpy as np
import os
from .ufo import UFO
from .bullet import Bullet
        
class SpaceGameAI:
    def __init__(self, width=900, height=600, difficulty=4):
        pygame.init()
        self.width = width
        self.height = height
        self.difficulty = difficulty
        
        # Window shown to the human
        self.display_surface = pygame.display.set_mode((width, height))
        self.bg = pygame.image.load(os.path.join(".", "src", "env", "assets",'bg.jpg'))
        self.bg = pygame.transform.scale(self.bg, (width, height))
        pygame.display.set_caption("Space Game")
        pygame.display.set_icon(pygame.image.load(os.path.join(".", "src", "env", "assets",'rocket.png')))
        
        # Off-screen surface used ONLY for the RL model
        self.model_surface = pygame.Surface((width, height))
        
        self.reset()
        
    def reset(self):
        self.score = 0
        
        self.rocket = pygame.image.load(os.path.join(".", "src", "env", "assets",'rocket.png'))
        self.rocket = pygame.transform.scale(self.rocket, (70, 70))
        self.rocket_x = 350
        self.rocket_y = 450
        self.moveX_rocket = 0
                
        self.bullet = Bullet(self.rocket_x, self.rocket_y)
        
        self.ufos = [UFO(self.width) for _ in range(self.difficulty)]
        
    def get_pixels(self):
        return pygame.surfarray.array3d(self.model_surface) 
            
    def update_rocket(self, screen):
        self.rocket_x += self.moveX_rocket
        if self.rocket_x <= 0:
            self.rocket_x = 0
        elif self.rocket_x >= self.width-70:
            self.rocket_x = self.width-70
        screen.blit(self.rocket, (self.rocket_x, self.rocket_y))
 
    def show_score(self, screen):
        font = pygame.font.Font('freesansbold.ttf', 32)
        score = font.render("Score: " + str(self.score), True, (255, 255, 255))
        screen.blit(score, (10, 10))
        
    def move(self, action):
        # [left, right, space, nothing]
        if np.array_equal(action, [1,0,0,0]):
            self.moveX_rocket = -2
        elif np.array_equal(action, [0,1,0,0]):
            self.moveX_rocket = 2
        elif np.array_equal(action, [0,0,1,0]):
            if self.bullet.bullet_fired == False:
                self.bullet.bullet_fired = True
                self.bullet.bullet_x = self.rocket_x
                self.bullet.bullet_y = self.rocket_y
        else:
            self.moveX_rocket = 0

    def play(self, action):
        self.display_surface.fill((0,0,0))
        self.display_surface.blit(self.bg, (0, 0))
        self.model_surface.fill((0,0,0))
        
        reward = 0.1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return reward, True, self.score
        
        self.move(action)
        
        for ufo in self.ufos:
            if ufo.ufo_y >= self.rocket_y-40:
                reward -= 10
                return self.reward, True, self.score
                    
            if ufo.is_ufo_killed(self.bullet.bullet_x, self.bullet.bullet_y):
                self.bullet.bullet_fired = False
                self.score += 1
                reward += 5
                ufo.ufo_x = random.randint(0, self.width-70)
                ufo.ufo_y = random.randint(50, 150)
            ufo.update_ufo(self.display_surface)
            ufo.update_ufo(self.model_surface)
           
        self.update_rocket(self.model_surface)
        self.update_rocket(self.display_surface)
        self.bullet.update_bullet(self.model_surface)
        self.bullet.update_bullet(self.display_surface)
        self.show_score(self.display_surface)
        pygame.display.flip()
        
        return reward, False, self.score
