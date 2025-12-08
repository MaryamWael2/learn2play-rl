import pygame
import random
import math
import os

class Obstacle:
    def __init__(self, screen, screen_width, obstacle_speed=20):
        self.screen = screen
        self.screen_width = screen_width
        self.obstacle = pygame.image.load(os.path.join(".", "cargame", "env", "assets", "obstacle.png"))
        self.obstacle = pygame.transform.scale(self.obstacle, (80, 100))
        x_positions = [200, 250, 300, 350, 400, 450, 500, 550, 600]        
        
        self.obstacle_x = x_positions[random.randint(0, len(x_positions)-1)]
        
        self.obstacle_y = 0
        self.moveX_obstacle = 0
        self.moveY_obstacle = obstacle_speed
        
    def update_obstacle(self):
        self.obstacle_y += self.moveY_obstacle
        if self.obstacle_y >= self.screen_width-70:
            self.moveX_obstacle = -abs(self.moveX_obstacle)
            self.obstacle_y += self.moveY_obstacle
        self.screen.blit(self.obstacle, (self.obstacle_x, self.obstacle_y))
        
    def is_collision(self, car_x, car_y):
        distance = math.sqrt(math.pow(self.obstacle_x - car_x, 2) + (math.pow(self.obstacle_y - car_y, 2)))
        if distance < 40:
            return True
        else:
            return False       
         