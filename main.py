import numpy as np
# from matplotlib import pyplot as plt
from sklearn import linear_model


class SahibindenPriceEstimator:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.model = None

    def read_data(self, file_name, test_size=5):
        np.set_printoptions(suppress=True)
        data = np.loadtxt(file_name)
        self.train_data = data[:-test_size]
        self.test_data = data[-test_size:]
        print('Data is read')

    def train(self):
        self.model = linear_model.LinearRegression()
        self.model.fit(self.train_data[:, :2], self.train_data[:, 2])
        print('Model is trained')

    def predict(self):
        price_predictions = self.model.predict(self.test_data[:, :2]).astype('int')
        print("Test Results:")
        print("YEAR\tKM\tPRICE_GT\tPRICE_PRED")
        for input, price in zip(self.test_data, price_predictions):
            input = input.astype('int')
            print(f"{input[0]}\t{input[1]}\t{input[2]}\t{price}")


if __name__ == '__main__':
    estimator = SahibindenPriceEstimator()
    estimator.read_data('honda_civic.txt', test_size=5)
    estimator.train()
    estimator.predict()
