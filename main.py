import numpy as np


def read_data(file_name):
    return np.loadtxt(file_name)


if __name__ == '__main__':
    data = read_data('honda_civic.txt')
    print(data)
