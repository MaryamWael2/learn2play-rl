import torch
import random
import numpy as np
from collections import deque
from CarGame.src.scripts.utils import resize_image_nn

class CNNAgent:
    def __init__(self, model, trainer, eps, eps_min, eps_decay, gamma, max_memory, batch_size):
        self.n_games = 0
        self.epsilon = eps   
        self.eps_min = eps_min    
        self.eps_decay = eps_decay 
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_memory)
        self.frame_stack = deque(maxlen=4)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.trainer = trainer
        
    def stack_frames(self, gray_screen):
        self.frame_stack.append(gray_screen)

        while len(self.frame_stack) < 4:
            self.frame_stack.append(self.frame_stack[0].copy())

        stacked_state = np.concatenate(self.frame_stack, axis=0)  # (4, 80, 80)
        return stacked_state

    def get_state(self, game):
        screen_array = game.get_pixels()
        gray_screen = np.dot(screen_array[...,:3], [0.2989, 0.5870, 0.1140])
        gray_screen = resize_image_nn(gray_screen, 80, 80)
        gray_screen = np.expand_dims(gray_screen, axis=0)  # (1, 80, 80)
        # gray_screen = gray_screen / 255.0
        return gray_screen

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        states      = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        actions     = np.array(actions, dtype=np.int64)
        rewards     = np.array(rewards, dtype=np.float32)
        dones       = np.array(dones, dtype=np.bool_)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        state      = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action     = np.array(action, dtype=np.int64)
        reward     = np.array(reward, dtype=np.float32)
        done       = np.array(done, dtype=np.bool_)

        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        final_move = [0, 0, 0]

        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
