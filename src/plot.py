import matplotlib.pyplot as plt
import numpy as np

def plot_moving_avg(window_size: int, filename: str):
    with open(filename + '.txt', 'r') as f:
        data = f.readlines()
    data = [float(x.strip()) for x in data]
    Ep_arr = np.array(data)
    plt.plot(Ep_arr)
    plt.xlabel('Episodes')
    plt.ylabel('Moving average')
    plt.title('Moving average of the' + filename + 'over 100 episodes')
    plt.savefig(filename + '.png')
    plt.show()
    return

if __name__ == '__main__':
    plot_moving_avg(100, filename='number_of_steps100')