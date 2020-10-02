import numpy as np


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, layers, alpha=0.01):
        # Example layer model[2, 2, 1]
        self.layers = layers
        # learning rate
        self.alpha = alpha
        # W, b parameters
        self.W = []
        self.b = []

        # Initialize parameters in each layer
        for i in range(0, len(layers) - 1):
            w_ = np.random.randn(layers[i], layers[i + 1])
            b_ = np.zeros((layers[i + 1], 1))
            self.W.append(w_ / layers[i])
            self.b.append(b_)

    def __repr__(self):
        return "Neural network [{}]".format("-".join(str(l) for l in self.layers))

    # train model with data
    def fit_partial(self, X, y):
        A = [X]
        out = A[-1]

        # process feedforward
        for i in range(0, len(self.layers) - 1):
            out = sigmoid(np.dot(out, self.W[i]) + self.b[i].T)
            A.append(out)
        # process backpropagation
        y = y.reshape(-1, 1)
        dA = [-(y / A[-1] - (1 - y) / (1 - A[-1]))]
        dW = []
        db = []
        for i in reversed(range(0, len(self.layers) - 1)):
            dw_ = np.dot((A[i]).T, dA[-1] * sigmoid_derivative(A[i + 1]))
            db_ = (np.sum(dA[-1] * sigmoid_derivative(A[i + 1]), 0)).reshape(-1, 1)
            da_ = np.dot(dA[-1] * sigmoid_derivative(A[i + 1]), self.W[i].T)
            dW.append(dw_)
            db.append(db_)
            dA.append(da_)
        # reversed dW, db
        dW = dW[::-1]
        db = db[::-1]
        
        # Gradient descent
        for i in range(0, len(self.layers) - 1):
            self.W[i] = self.W[i] - self.alpha * dW[i]
            self.b[i] = self.b[i] - self.alpha * db[i]

    def fit(self, X, y, epochs=10):
        for epoch in range(0, epochs):
            self.fit_partial(X, y)
            loss = self.calculate_loss(X, y)
            print("Epoch {}, loss {}".format(epoch, loss))
    
    # predict
    def predict(self, X):
        for i in range(0, len(self.layers) - 1):
            X = sigmoid(np.dot(X, self.W[i]) + self.b[i].T)
        return X

    # calculate loss function
    def calculate_loss(self, X, y):
        y_predict = self.predict(X)

        # return np.sum((y_predict-y)**2)/2
        return -(np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict)))


if __name__ == "__main__":
    X = np.array(([2, 9], [1, 5], [3, 6], [5, 4], [3, 2], [4, 3]))
    y = np.array(([11], [6], [9], [9], [5], [7]))
    model = NeuralNetwork(layers=[2, 2, 1])
    model.fit(X, y)