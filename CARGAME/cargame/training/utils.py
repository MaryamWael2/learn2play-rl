import numpy as np
import matplotlib.pyplot as plt
from IPython import display

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
    
def log():
    pass