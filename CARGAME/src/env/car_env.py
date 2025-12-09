import pygame
import math
import numpy as np
import os
from .obstacle import Obstacle
from .direction import Direction     
        
class CarGameAI:
    def __init__(self, width=900, height=600, obstacle_speed=20, max_obstacles=7):
        pygame.init()
        self.width = width
        self.height = height
        self.max_no_of_obstacles = max_obstacles
        self.obstacle_speed = obstacle_speed
        
        # Window shown to the human
        self.display_surface = pygame.display.set_mode((width, height))
        self.bg = pygame.image.load(os.path.join(".", "src", "env", "assets", "road.jpg"))
        self.bg = pygame.transform.scale(self.bg, (width, height))
        
        # Off-screen surface used ONLY for the RL model
        self.model_surface = pygame.Surface((width, height))
        
        self.reset()
    
    def reset(self):
        pygame.display.set_caption("Car Game")
        pygame.display.set_icon(pygame.image.load(os.path.join(".", "src", "env", "assets", "car.png")))
        self.score = 0
        self.direction = Direction.LEFT
        
        self.car = pygame.image.load(os.path.join(".", "src", "env", "assets", "car.png"))
        self.car = pygame.transform.scale(self.car, (80, 100))
        self.car_x = 350
        self.car_y = 450
        self.moveX_car = 0
        self.spawn_delay = 1000
        
        self.obstacles = []    
        self.obstacles.append(Obstacle(self.width, self.obstacle_speed))
        
        self.last_spawn_time = pygame.time.get_ticks()
        
    def get_pixels(self):
        return pygame.surfarray.array3d(self.model_surface) 
    
    def update_car(self):
        self.car_x += self.moveX_car
        if self.car_x <= 200:
            self.car_x = 200
        elif self.car_x >= 600:
            self.car_x = 600
        self.model_surface.blit(self.car, (self.car_x, self.car_y))
        self.display_surface.blit(self.car, (self.car_x, self.car_y))
        
        
    def calc_distance(self, coordinates1, coordinates2):
        x1, y1 = coordinates1
        x2, y2 = coordinates2
        distance = math.sqrt(math.pow(x1 - x2, 2) + (math.pow(y1 - y2, 2)))
        return distance
    
    def calc_direction(self, coordinates1, coordinates2):
        x1, y1 = coordinates1
        x2, y2 = coordinates2
        direction = math.atan2(y2 - y1, x2 - x1)
        return math.degrees(direction)
 
    def show_score(self):
        font = pygame.font.Font('freesansbold.ttf', 32)
        score = font.render("Score: " + str(self.score), True, (255, 255, 255))
        self.display_surface.blit(score, (10, 10))
        
    def display_game_over(self):
        self.display_surface.fill((0,0,0))
        over_font = pygame.font.Font('freesansbold.ttf', 64)
        over_text = over_font.render("GAME OVER", True, (255, 255, 255))
        self.display_surface.blit(over_text, (250, 250))
        
    def move(self, action):
        #[left, right, no action]
        if np.array_equal(action, [1,0,0]):
            self.direction = Direction.LEFT
            self.moveX_car = -10
        elif np.array_equal(action, [0,1,0]):
            self.direction = Direction.RIGHT
            self.moveX_car = 10
        elif np.array_equal(action, [0,0,1]):
            self.moveX_car = 0

    def play(self, action):
        self.display_surface.fill((0,0,0))
        self.display_surface.blit(self.bg, (0, 0))
        self.model_surface.fill((0,0,0))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return reward, True, self.score
        
        self.move(action)
        
        current_time = pygame.time.get_ticks()
        if current_time - self.last_spawn_time > self.spawn_delay and len(self.obstacles) < self.max_no_of_obstacles:
            self.obstacles.append(Obstacle(self.width, self.obstacle_speed))
            self.last_spawn_time = current_time 
            
        reward = 0.1 
            
        for obstacle in self.obstacles[:]:
            if obstacle.is_collision(self.car_x, self.car_y):
                # self.display_game_over()
                reward -= 10
                self.update_car()
                pygame.display.flip()
                return reward, True, self.score
            
            if obstacle.obstacle_y >= self.height:
                self.score += 1
                reward += 1
                self.obstacles.remove(obstacle)
                    
            obstacle.update_obstacle(self.model_surface)
            obstacle.update_obstacle(self.display_surface)
    
        self.update_car()
        self.show_score()
        pygame.display.flip()
        
        return reward, False, self.score

        