# get data here: https://www.kaggle.com/dalpozz/creditcardfraud
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import sklearn.preprocessing
import matplotlib.pyplot as plt


class LogisticModel(object):

    # helper functions
    def sigmoid(self, A):
        return 1 / (1 + np.exp(-A))

    def forward(self, X):
        return self.sigmoid(X.dot(self.w + self.b))

    def predict(self, X, Y):
        pY = self.forward(X)
        return np.round(pY)

    def error_rate(self, targets, predictions):
        return np.mean(targets != predictions)

    def sigmoid_cost(self, T, Y):
        return -(T * np.log(Y) + (1 - T) * np.log(1 - Y)).sum()

    def score(self, X, Y):
        predictions = self.predict(X)
        return 1 - self.error_rate(Y, predictions)
        
    '''
    hyper-parameter examples and their outputs
    
    learning_rate=10e-13, reg=10e-20, epochs=100000
    i: 99980 cost: 677.0974736 error: 0.381

    learning_rate=10e-13, reg=10e-20, epochs=200000
    i: 199960 cost: 548.362226598 error: 0.299

    learning_rate=10e-13, reg=10e-15, epochs=100000
    i: 99980 cost: 444.223794383 error: 0.279

    learning_rate=10e-13, reg=10e-15, epochs=200000
    i: 199980 cost: 537.157786775 error: 0.305
    '''


    def fit(self, X, Y, learning_rate=10e-13, reg=10e-15,
            epochs=50000, show_fig=False):
        X, Y = shuffle(X, Y)
        X_valid, Y_valid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]
        N, D = X.shape

        self.w = np.random.randn(D) / np.sqrt(D)
        self.b = 0

        costs = []
        for i in range(epochs):
            pY = self.forward(X)
            self.w -= learning_rate * (X.T.dot(pY - Y) + reg * self.w)
            self.b -= learning_rate * ((pY - Y).sum() + reg * self.b)

            if i % 20 == 0:
                pY_valid = self.forward(X_valid)
                cost = self.sigmoid_cost(Y_valid, pY_valid)
                costs.append(cost)
                error_rate = self.error_rate(Y_valid, pY_valid.round())
                print("i:", i, "cost:", cost, "error:", error_rate)

        if show_fig:
            plt.plot(costs)
            plt.show()


if __name__ == '__main__':
    # load data
    df = pd.read_csv('../input/creditcard.csv')
    X = np.array(df.iloc[:, 0:30])
    X = sklearn.preprocessing.StandardScaler().fit_transform(X)
    Y = np.array(df.iloc[:, 30])

    # the dataset is highly unbalanced,
    # the positive class 1 (frauds) make up ~0.172% of all transactions
    X0 = X[Y == 0, :]
    X1 = X[Y == 1, :]
    balance_factor = int(1 / (len(X1) / len(X)))
    X1 = np.repeat(X1, balance_factor, axis=0)
    X_bal = np.vstack([X0, X1])
    Y_bal = np.array([0] * len(X0) + [1] * len(X1))
    model = LogisticModel()
    model.fit(X_bal, Y_bal, show_fig=True)
