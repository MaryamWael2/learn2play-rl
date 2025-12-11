import matplotlib.pyplot as plt
import os
from src.agent.cnn_agent import CNNAgent
from src.agent.cnn_model import CNN
from src.env.car_env_ai import CarGameAI
from CarGame.src.scripts.utils import save_plot, setup_logger

def main():
    model = CNN(4, 3).load()
    agent = CNNAgent(model)
    game = CarGameAI(obstacle_speed=20, max_obstacles=7)

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    logger = setup_logger("logs", "testing.log")
    logger.info("Testing started.")

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

            state_old = state_new

        agent.n_games += 1

        logger.info(f"Game {agent.n_games} | Score: {score} | Record: {record}")

        plot_scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        plot_mean_scores.append(mean_score)

        logger.info(f"Mean Score: {mean_score}")
        save_plot(plot_scores, plot_mean_scores, "plots", "testing_plot.png")

        if agent.n_games % 50 == 0:
            last_50 = plot_scores[-50:]
            logger.info("----- Last 50 Games Summary -----")
            logger.info(f"Mean score: {sum(last_50)/50}")
            logger.info(f"Zero-score games: {last_50.count(0)}")
            logger.info("---------------------------------")

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    plt.ion()
    main()
