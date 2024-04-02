import matplotlib.pyplot as plt
from IPython import display
import os
plt.ion()
# Define the directory to save the plot
SAVE_DIR = "C:/Users/user/OneDrive/Documents/Projects/snake-pygame/graph"
SAVE_FILE_PATH = os.path.join(SAVE_DIR, "mean_scores_plot.png")
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

def save_mean_score_plot(mean_scores):
    plt.figure(figsize=(8, 6))
    plt.title('Mean Score')
    plt.xlabel('Number of Games')
    plt.ylabel('Mean Score')
    plt.plot(mean_scores, label='Mean Scores', color='red')
    plt.ylim(ymin=0)
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    
    # Save the plot to the file
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    plt.savefig(SAVE_FILE_PATH)
    plt.close()
