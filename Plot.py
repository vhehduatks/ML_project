import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores,marker='o',linestyle='-',alpha=0.5)
    plt.plot(mean_scores,marker='o',linestyle='-',alpha=0.5)
    plt.ylim(ymin=0)
    plt.show(block=False)
    plt.pause(.1)
