import numpy as np
import matplotlib.pyplot as plt
import os

def plot(x1, y1, y2, labels, title, xlabel, ylabel, filename):
    plt.plot(x1, y2, 'r--', label=labels[1])
    plt.plot(x1, y1, 'g--', label=labels[0])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper right')
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    ####Q1-Part1
    directory = os.getcwd()
    files = os.listdir(directory)
    data = np.empty((0,0))
    names = []
    for item in files:
        with open(item, "r") as f:
            if item.endswith(".txt"):
                lines = f.read().splitlines()
                lines = [float(i) for i in lines]
                lines = np.array(lines)
                lines = np.reshape(lines, (1, len(lines)))
                if data.size == 0:
                    data = lines
                else:
                    data = np.append(data, lines, axis=0)

    names = ['constant step size', 'adaptive step size']      
    x1 = [i+1 for i in range(data.shape[1])]
    y1 = data[0, :]
    y2 = data[1, :]
    plot(x1, y1, y2, names, 'Running average for the number of steps per episode', 'Episode', "Number of steps", 'qlearningVSsarsa.png')