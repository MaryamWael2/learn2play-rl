from src.env.car_env_human import CarGame
import pygame

if __name__ == '__main__':  
    obstacle_speed=1
    max_obstacles=7 
    game = CarGame(900, 600, obstacle_speed, max_obstacles)
    
    while True:  
        quit, score = game.play()
        if quit:
            print("Final Score:", score)
            break

    pygame.quit()