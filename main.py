import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model


class SahibindenPriceEstimator:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.model = None

    def read_data(self, file_name, train_size=50, test_size=5):
        np.set_printoptions(suppress=True)
        data = np.loadtxt(file_name)
        self.train_data = data[:train_size]
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
        print(f"Test Data Mean Abs Error: {np.abs(self.test_data[:, 2] - price_predictions).mean()}")

    def plot(self):

        plot_n = 100
        plot_data = np.zeros((plot_n, 2))

        plot_data[:, 0] = np.linspace(min(self.train_data[:, 0]), max(self.train_data[:, 0]), plot_n)
        plot_data[:, 1] = np.linspace(max(self.train_data[:, 1]), min(self.train_data[:, 1]), plot_n)

        fig = plt.figure(figsize=(9, 5))
        plt.scatter(self.train_data[:, 0], self.train_data[:, 2])
        plt.plot(plot_data[:, 0], self.model.predict(plot_data), 'r-')
        plt.xlabel('Year')
        plt.ylabel('Price')
        plt.title('Honda Civic Year vs Price Data')
        plt.legend(['Training Data', 'Regression Model'])
        plt.savefig('year_price.png', bbox_inches='tight')
        plt.show()

        fig = plt.figure(figsize=(9, 5))
        plt.scatter(self.train_data[:, 1], self.train_data[:, 2])
        plt.plot(plot_data[:, 1], self.model.predict(plot_data), 'r-')
        plt.xlabel('KM')
        plt.ylabel('Price')
        plt.title('Honda Civic Km vs Price Data')
        plt.legend(['Training Data', 'Regression Model'])
        plt.savefig('km_price.png', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    estimator = SahibindenPriceEstimator()
    estimator.read_data('honda_civic.txt', train_size=450, test_size=50)
    estimator.train()
    estimator.predict()
    estimator.plot()
