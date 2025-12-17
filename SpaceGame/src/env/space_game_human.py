import pygame
import random
import os
from .ufo import UFO
from .bullet import Bullet

class SpaceGame:
    def __init__(self, width=900, height=600, difficulty=6):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.bg = pygame.image.load(os.path.join(".", "src", "env", "assets",'bg.jpg'))
        self.bg = pygame.transform.scale(self.bg, (width, height))
        pygame.display.set_caption("Space Game")
        pygame.display.set_icon(pygame.image.load(os.path.join(".", "src", "env", "assets",'rocket.png')))
        self.score = 0
        
        self.rocket = pygame.image.load(os.path.join(".", "src", "env", "assets",'rocket.png'))
        self.rocket = pygame.transform.scale(self.rocket, (70, 70))
        self.rocket_x = 350
        self.rocket_y = 450
        self.moveX_rocket = 0
                
        self.bullet = Bullet(self.rocket_x, self.rocket_y)
        
        self.ufos = [UFO(self.width) for _ in range(difficulty)]
        
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
        self.screen.fill((0,0,0))
        self.screen.blit(self.bg, (0, 0))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True, self.score
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.moveX_rocket = -2
                if event.key == pygame.K_RIGHT:
                    self.moveX_rocket = 2
                if event.key == pygame.K_SPACE:
                    if self.bullet.bullet_fired == False:
                        self.bullet.bullet_fired = True
                        self.bullet.bullet_x = self.rocket_x
            else:
                self.moveX_rocket = 0
        
        for ufo in self.ufos:
            if ufo.ufo_y >= self.rocket_y-40:
                self.display_game_over()
                break
                    
            if ufo.is_ufo_killed(self.bullet.bullet_x, self.bullet.bullet_y):
                self.bullet.bullet_y = self.rocket_y
                self.bullet.bullet_fired = False
                self.score += 1
                ufo.ufo_x = random.randint(0, self.width-70)
                ufo.ufo_y = random.randint(50, 150)
            ufo.update_ufo(self.screen)
           
        self.update_rocket()
        self.bullet.update_bullet(self.screen)
        self.show_score()
        pygame.display.flip()
        
        return False, self.score

if __name__ == '__main__':
    
    game = SpaceGame()
    
    while True:  
        quit, score = game.play()
        if quit:
            break

    pygame.quit()  
        