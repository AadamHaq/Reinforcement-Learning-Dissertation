import matplotlib.pyplot as plt
import os

# Define the directory to save the plots
SAVE_DIR = "C:/Users/user/OneDrive/Documents/Projects/snake-pygame/graph"
SAVE_FILE_PATH_SCORES = os.path.join(SAVE_DIR, "scores_plot.png")
SAVE_FILE_PATH_MEAN_SCORES = os.path.join(SAVE_DIR, "mean_scores_plot.png")

def plot(scores, mean_scores):
    plt.figure(figsize=(8, 6))
    plt.title('Scores and Mean Score')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Scores')
    plt.plot(mean_scores, label='Mean Scores', color='red')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()

    # Save the plot of scores to the file
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    plt.savefig(SAVE_FILE_PATH_SCORES)
    plt.close()

def save_mean_score_plot(mean_scores):
    plt.figure(figsize=(8, 6))
    plt.title('Mean Score')
    plt.xlabel('Number of Games')
    plt.ylabel('Mean Score')
    plt.plot(mean_scores, label='Mean Scores', color='red')
    plt.ylim(ymin=0)
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()

    # Save the plot of mean scores to the file
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    plt.savefig(SAVE_FILE_PATH_MEAN_SCORES)
    plt.close()
