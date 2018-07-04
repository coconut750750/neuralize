import matplotlib.pyplot as plt

def plot_2lines(xs, y1s, y2s, x_label=""):
    fig, ax1 = plt.subplots()
    ax1.plot(xs, y1s, 'b-', label="Loss")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('loss', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(xs, y2s, 'r-', label="Performance")
    ax2.set_ylabel('performance', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    ax1.legend(loc='lower right')
    ax2.legend(loc='upper right')
    plt.show()
