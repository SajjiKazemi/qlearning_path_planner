import numpy as np

def get_moving_avg(data_list: list, window_size: int, filename: str):

    moving_averages= []
    i = 0
    while i < len(data_list) - window_size + 1:
        this_window = data_list[i : i + window_size]
        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1

    Ep_arr = np.array(moving_averages)
    with open(filename + f"{window_size}" + '.txt', 'w') as f:
        for item in Ep_arr:
            f.write("%s\n" % item)
    print("Moving average is available!")
    return

def write_data(data_list: list, filename: str):
    data_arr = np.array(data_list)
    with open(filename, 'w') as f:
        for item in data_arr:
            f.write("%s\n" % item)
    print("Data is available!")
    return