import pygame
import os
from .obstacle import Obstacle
        
class CarGame:
    def __init__(self, width=900, height=600, obstacle_speed=20, max_obstacles=7):
        pygame.init()
        self.width = width
        self.height = height
        self.max_no_of_obstacles = max_obstacles
        self.obstacle_speed = obstacle_speed
        
        self.screen = pygame.display.set_mode((width, height))
        self.bg = pygame.image.load(os.path.join(".", "src", "env", "assets", "road.jpg"))
        self.bg = pygame.transform.scale(self.bg, (width, height))
        
        self.reset()
        
    def reset(self):
        pygame.display.set_caption("Car Game")
        pygame.display.set_icon(pygame.image.load(os.path.join(".", "src", "env", "assets", "car.png")))
        self.car = pygame.image.load(os.path.join(".", "src", "env", "assets", "car.png"))
        self.car = pygame.transform.scale(self.car, (80, 100))
        
        self.score = 0
        self.car_x = 350
        self.car_y = 450
        self.moveX_car = 0
        self.spawn_delay = 800
        
        self.obstacles = []
        self.obstacles.append(Obstacle(self.width, self.obstacle_speed))
        
        self.last_spawn_time = pygame.time.get_ticks()
        
    def update_car(self):
        self.car_x += self.moveX_car
        if self.car_x <= 200:
            self.car_x = 200
        elif self.car_x >= 600:
            self.car_x = 600
        self.screen.blit(self.car, (self.car_x, self.car_y))
 
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
        
        self.moveX_car = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True, self.score
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.moveX_car = -10
                elif event.key == pygame.K_RIGHT:
                    self.moveX_car = 10
                else:
                    self.moveX_car = 0
        
        current_time = pygame.time.get_ticks()
        if current_time - self.last_spawn_time > self.spawn_delay and len(self.obstacles) < self.max_no_of_obstacles:
            self.obstacles.append(Obstacle(self.width, self.obstacle_speed))
            self.last_spawn_time = current_time 
            
        for obstacle in self.obstacles[:]:
            if obstacle.obstacle_y >= self.height:
                self.score += 1
                self.obstacles.remove(obstacle)
                    
            if obstacle.is_collision(self.car_x, self.car_y):
                self.display_game_over()
                return True, self.score
                
            obstacle.update_obstacle(self.screen)
           
        self.update_car()
        self.show_score()
        pygame.display.flip()
        
        return False, self.score
        