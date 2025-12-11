import matplotlib.pyplot as plt
import os
from src.agent.cnn_agent import CNNAgent
from src.agent.cnn_model import CNN
from CarGame.src.agent.qtrainer import QTrainer
from src.env.car_env_ai import CarGameAI
from CarGame.src.scripts.utils import save_plot, setup_logger

def main():
    epsilon = 1.0
    eps_min = 0.05
    eps_decay = 0.995
    gamma = 0.9
    max_memory = 40_000
    batch_size = 256
    lr = 0.0001

    model = CNN(4, 3)
    trainer = QTrainer(model, lr=lr, gamma=gamma)
    agent = CNNAgent(model, trainer, epsilon, eps_min, eps_decay, gamma, max_memory, batch_size)
    game = CarGameAI(obstacle_speed=20, max_obstacles=7)

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    logger = setup_logger()
    logger.info("Training started.")

    while True:
        game.reset()
        agent.frame_stack.clear()

        first_frame = agent.get_state(game)
        state_old = agent.stack_frames(first_frame)

        done = False
        score = 0

        while not done:

            final_move = agent.get_action(state_old)

            reward, done, score = game.play(final_move)

            new_frame = agent.get_state(game)
            state_new = agent.stack_frames(new_frame)

            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            state_old = state_new

        agent.n_games += 1
        agent.train_long_memory()

        agent.epsilon = max(agent.eps_min, agent.epsilon * agent.eps_decay)

        if score > record:
            record = score
            agent.model.save()

        logger.info(f"Game {agent.n_games} | Score: {score} | Record: {record}")

        plot_scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        plot_mean_scores.append(mean_score)

        logger.info(f"Mean Score: {mean_score}")
        save_plot(plot_scores, plot_mean_scores)

        if agent.n_games % 50 == 0:
            last_50 = plot_scores[-50:]
            logger.info("----- Last 50 Games Summary -----")
            logger.info(f"Mean score: {sum(last_50)/50}")
            logger.info(f"Zero-score games: {last_50.count(0)}")
            logger.info(f"Memory size: {len(agent.memory)}")
            logger.info("---------------------------------")

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.makedirs("train_logs", exist_ok=True)
    os.makedirs("train_plots", exist_ok=True)
    plt.ion()
    main()
