import pygame
import random
import numpy as np
import os
from .ufo import UFO
from .bullet import Bullet
from .ufo_bullet import UFOBullet
        
class SpaceGameAI:
    def __init__(self, width=900, height=600, difficulty=3):
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
        
        # UFO Bullets
        self.ufo_bullets = []
        self.last_ufo_shot = pygame.time.get_ticks()
        self.ufo_shoot_delay = 2500
        
        # UFOs
        self.ufos = []
        self.ufo_direction = 1
        self.ufo_speed = 1
        self.ufo_drop = 20
        self.spawn_ufos(self.difficulty)
        
    def get_pixels(self):
        return pygame.surfarray.array3d(self.model_surface) 
            
    def update_rocket(self, screen):
        self.rocket_x += self.moveX_rocket
        if self.rocket_x <= 0:
            self.rocket_x = 0
        elif self.rocket_x >= self.width-70:
            self.rocket_x = self.width-70
        screen.blit(self.rocket, (self.rocket_x, self.rocket_y))
        
    def update_ufos(self):
        #respawn ufos if empty
        if not self.ufos:
            self.ufo_speed += 0.3
            self.spawn_ufos(3)
            
        #update ufo position
        hit_edge = False
        for ufo in self.ufos:
            ufo.ufo_x += self.ufo_direction * self.ufo_speed
            if ufo.ufo_x <= 0 or ufo.ufo_x >= self.width - 70:
                hit_edge = True

        if hit_edge:
            self.ufo_direction *= -1
            for ufo in self.ufos:
                ufo.ufo_y += self.ufo_drop
                
        #ufo shooting        
        now = pygame.time.get_ticks()
        if now - self.last_ufo_shot > self.ufo_shoot_delay and self.ufos:
            shooter = random.choice(self.ufos)
            self.ufo_bullets.append(UFOBullet(shooter.ufo_x, shooter.ufo_y))
            self.last_ufo_shot = now
            
    def spawn_ufos(self, rows):
        self.ufos.clear()
        self.ufo_bullets.clear()

        for r in range(rows):
            for c in range(int(self.width // 200)):
                self.ufos.append(UFO(c * 140, r * 90))
 
    def show_score(self, screen):
        font = pygame.font.Font('freesansbold.ttf', 32)
        score = font.render("Score: " + str(self.score), True, (255, 255, 255))
        screen.blit(score, (10, 10))
        
    def move(self, action):
        # [left, right, space, nothing]
        self.moveX_rocket = 0
        if np.array_equal(action, [1,0,0,0]):
            self.moveX_rocket = -10
        elif np.array_equal(action, [0,1,0,0]):
            self.moveX_rocket = 10
        elif np.array_equal(action, [0,0,1,0]):
            if self.bullet.bullet_fired == False:
                self.bullet.bullet_fired = True
                self.bullet.bullet_x = self.rocket_x + 35
                self.bullet.bullet_y = self.rocket_y
        else:
            pass

    def play(self, action):
        self.display_surface.fill((0,0,0))
        self.display_surface.blit(self.bg, (0, 0))
        self.model_surface.fill((0,0,0))
        
        reward = 0.1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return reward, True, self.score
        
        self.move(action)
        
        for ufo in self.ufos[:]:
            if ufo.is_collision(self.rocket_x, self.rocket_y):
                reward -= 10
                return reward, True, self.score
            
            if ufo.ufo_y + 70 >= self.height:
                reward -= 10
                return reward, True, self.score
                    
            if self.bullet.bullet_fired and ufo.is_collision(self.bullet.bullet_x, self.bullet.bullet_y):
                self.bullet.bullet_fired = False
                self.bullet.bullet_x = -1000
                self.bullet.bullet_y = -1000
                self.score += 1
                reward += 1
                self.ufos.remove(ufo)
            ufo.draw(self.display_surface)
            ufo.draw(self.model_surface)
            
        for bullet in self.ufo_bullets[:]:
            if bullet.y > self.height:
                self.ufo_bullets.remove(bullet)
                
            if bullet.hit_rocket(self.rocket_x, self.rocket_y):
                reward -= 10
                return reward, True, self.score
            
            bullet.update(self.model_surface)
            bullet.update(self.display_surface)
           
        self.update_ufos()
        self.update_rocket(self.model_surface)
        self.update_rocket(self.display_surface)
        self.bullet.update_bullet(self.model_surface)
        self.bullet.update_bullet(self.display_surface)
        self.show_score(self.display_surface)
        pygame.display.flip()
        
        return reward, False, self.score
