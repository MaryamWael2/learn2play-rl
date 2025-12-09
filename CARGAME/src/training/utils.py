import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import logging
from logging.handlers import RotatingFileHandler
import os

def resize_image_nn(img, new_height, new_width):
    old_height, old_width = img.shape[:2]
    row_ratio = old_height / new_height
    col_ratio = old_width / new_width

    # Create an empty array for the new image
    if img.ndim == 3:
        resized_img = np.zeros((new_height, new_width, img.shape[2]), dtype=img.dtype)
    else:
        resized_img = np.zeros((new_height, new_width), dtype=img.dtype)

    for i in range(new_height):
        for j in range(new_width):
            orig_i = int(i * row_ratio)
            orig_j = int(j * col_ratio)
            resized_img[i, j] = img[orig_i, orig_j]

    return resized_img

def plot_image(img):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.imshow(img)
    plt.show(block=False)
    plt.pause(.1)
    
def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

# save_side_by_side(state_old, f"game_frames/game_{agent.n_games}_step_{step}_move_{final_move}_reward_{reward}.png")
def save_side_by_side(imgs, save_path):
    n = imgs.shape[0]

    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))

    # If only one image (n=1), force axes into a list
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.imshow(imgs[i], cmap="gray", vmin=imgs[i].min(), vmax=imgs[i].max())
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    
def setup_logger(log_dir = "train_logs"):
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "training.log")

    logger = logging.getLogger("trainer_logger")
    logger.setLevel(logging.INFO)

    # Rotating log file: 5 MB max, keep 3 backups
    handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)

    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(handler)

    return logger

def save_plot(scores, mean_scores, plot_dir = "train_plots"):
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(scores, label="Score per Game")
    plt.plot(mean_scores, label="Mean Score")
    plt.legend()
    plt.xlabel("Game")
    plt.ylabel("Score")
    plt.title("Training Progress")

    path = os.path.join(plot_dir, "training_plot.png")
    plt.savefig(path)
    plt.close()