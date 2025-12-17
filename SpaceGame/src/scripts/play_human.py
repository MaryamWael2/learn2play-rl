from src.env.space_game_human import SpaceGame
import pygame

if __name__ == '__main__':  
    game = SpaceGame(900, 600, difficulty=6)
    
    while True:  
        quit, score = game.play()
        if quit:
            print("Final Score:", score)
            break

    pygame.quit()