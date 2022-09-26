import matplotlib.pyplot as plt


class Visualizations:
    def __init__(self):
        print(">>> Visualization instance initiated...")


def epoch_vs_loss_plot(loss_plot):
    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()
