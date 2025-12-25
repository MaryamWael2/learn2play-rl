import pygame
import random
import os
from .ufo import UFO
from .bullet import Bullet
from .ufo_bullet import UFOBullet

class SpaceGame:
    def __init__(self, width=900, height=600, difficulty=3):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.bg = pygame.image.load(os.path.join(".", "src", "env", "assets",'bg.jpg'))
        self.bg = pygame.transform.scale(self.bg, (width, height))
        pygame.display.set_caption("Space Game")
        pygame.display.set_icon(pygame.image.load(os.path.join(".", "src", "env", "assets",'rocket.png')))
        
        self.score = 0

        # Rocket
        self.rocket = pygame.image.load(os.path.join(".", "src", "env", "assets", "rocket.png"))
        self.rocket = pygame.transform.scale(self.rocket, (70, 70))
        self.rocket_x = 350
        self.rocket_y = 450
        self.moveX_rocket = 0

        self.bullet = Bullet(self.rocket_x, self.rocket_y)

        # UFOs
        self.ufos = []
        self.ufo_direction = 1
        self.ufo_speed = 1
        self.ufo_drop = 20

        # UFO Bullets
        self.ufo_bullets = []
        self.last_ufo_shot = pygame.time.get_ticks()
        self.ufo_shoot_delay = 1500

        self.spawn_ufos(difficulty)

    def spawn_ufos(self, rows):
        self.ufos.clear()
        self.ufo_bullets.clear()

        for r in range(rows):
            for c in range(int(self.width // 200)):
                self.ufos.append(UFO(c * 140, r * 90))

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
            self.ufo_bullets.append(UFOBullet(shooter.ufo_x, shooter.ufo_y, speed=2))
            self.last_ufo_shot = now                

    def update_rocket(self):
        self.rocket_x += self.moveX_rocket
        if self.rocket_x <= 0:
            self.rocket_x = 0
        elif self.rocket_x >= self.width-70:
            self.rocket_x = self.width-70
        self.screen.blit(self.rocket, (self.rocket_x, self.rocket_y))
 
    def show_score(self):
        font = pygame.font.Font('freesansbold.ttf', 32)
        score = font.render("Score: " + str(self.score), True, (255, 255, 255))
        self.screen.blit(score, (10, 10))
        
    def display_game_over(self):
        self.screen.fill((0,0,0))
        over_font = pygame.font.Font('freesansbold.ttf', 64)
        over_text = over_font.render("GAME OVER", True, (255, 255, 255))
        self.screen.blit(over_text, (250, 250))

    def play(self):
        self.screen.blit(self.bg, (0, 0))

        self.moveX_rocket = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True, self.score
             
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.moveX_rocket = -20
                elif event.key == pygame.K_RIGHT:
                    self.moveX_rocket = 20
                elif event.key == pygame.K_SPACE and not self.bullet.bullet_fired:
                    self.bullet.bullet_fired = True
                    self.bullet.bullet_x = self.rocket_x + 35
                    self.bullet.bullet_y = self.rocket_y      
                else:
                    self.moveX_rocket = 0
                    
        for ufo in self.ufos[:]:
            if ufo.is_collision(self.rocket_x, self.rocket_y):
                self.display_game_over()
                return True, self.score
            
            if ufo.ufo_y + 70 >= self.height:
                self.display_game_over()
                return True, self.score

            if self.bullet.bullet_fired and ufo.is_collision(self.bullet.bullet_x, self.bullet.bullet_y):
                self.bullet.bullet_fired = False
                self.bullet.bullet_x = -1000
                self.bullet.bullet_y = -1000
                self.score += 1
                self.ufos.remove(ufo)

            ufo.draw(self.screen)
            
        for bullet in self.ufo_bullets[:]:
            if bullet.y > self.height:
                self.ufo_bullets.remove(bullet)
                
            if bullet.hit_rocket(self.rocket_x, self.rocket_y):
                self.display_game_over()
                return True, self.score
            
            bullet.update(self.screen)
            
        self.update_ufos()
        self.update_rocket()
        self.bullet.update_bullet(self.screen)
        self.show_score()
        pygame.display.flip()
        return False, self.score
