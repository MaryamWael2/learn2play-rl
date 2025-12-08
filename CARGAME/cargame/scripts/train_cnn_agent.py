from cargame.agents.cnn_agent import CNNAgent
from cargame.agents.cnn_model import CNN
from cargame.training.trainer import QTrainer
from cargame.env.car_env import CarGameAI
import matplotlib.pyplot as plt
import os

def main():
    epsilon = 1.0
    eps_min = 0.05
    eps_decay = 0.995
    gamma = 0.9
    max_memory = 40_000
    batch_size = 256
    lr =  0.0001
    
    model = CNN(4, 3)
    trainer = QTrainer(model, lr=lr, gamma=gamma)
    agent = CNNAgent(model, trainer, epsilon, eps_min, eps_decay, gamma, max_memory, batch_size)
    game = CarGameAI(obstacle_speed=20, max_obstacles=7)
    
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    while True:
        game.reset()
        agent.frame_stack.clear()

        # get initial stacked state
        first_frame = agent.get_state(game)
        state_old = agent.stack_frames(first_frame)

        done = False
        score = 0

        while not done:
            # choose action
            final_move = agent.get_action(state_old)

            # env step
            reward, done, score = game.play(final_move)

            # get new frame and update stack ONCE per step
            new_frame = agent.get_state(game)
            state_new = agent.stack_frames(new_frame)

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            # shift state
            state_old = state_new

        # episode end
        agent.n_games += 1
        agent.train_long_memory()

        # decay epsilon
        agent.epsilon = max(agent.eps_min, agent.epsilon * agent.eps_decay)

        if score > record:
            record = score
            agent.model.save()

        print('Game:', agent.n_games, 'Score:', score, 'Record:', record)

        plot_scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        plot_mean_scores.append(mean_score)
        print("Mean score:", mean_score)
        
        if agent.n_games % 50 == 0:
            print("-------------------------------------- Over last 50 games --------------------------------------")
            last_50_games = plot_scores[-50:]
            print("Last 50 games mean score:", sum(last_50_games)/50)
            print("Number of zero score games in last 50 games:", last_50_games.count(0))
            print("Size of memory: ", len(agent.memory))
            print("-----------------------------------------------------------------------------------------------------")

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    plt.ion()
    main()
