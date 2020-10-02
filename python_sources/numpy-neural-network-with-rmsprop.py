import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

def get_normalized_data(df):
    dat = df.as_matrix().astype(np.float32)
    X = dat[:, 1:]
    mu = X.mean(axis = 0)
    std = X.std(axis = 0)
    np.place(std, std == 0, 1)
    X = (X - mu) / std
    # X /= 255.0
    Y = dat[:, 0]
    return X, Y

class NeuralNet(object):
    def __init__(self, hidden_layer_size):
        self.hidden_layer_size = hidden_layer_size
        self.costs = []
        self.test_acc = []

    def fit(self, X, y, learning_rate=0.001, decay_rate=0.999, eps = 0.0000000001, reg = 0.01, show_cost=False, 
                num_iterations=20, num_classes=10, batch_size=500, img_dim=28, print_iter=20, random_state=None):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1000, random_state = random_state)
        y_train_one_hot = self.one_hot(y_train)
        y_test_one_hot = self.one_hot(y_test)

        num_samples, num_features = X_train.shape
        num_batches = num_samples / batch_size

        img_size = img_dim * img_dim

        W1 = np.random.randn(num_features, self.hidden_layer_size) / img_dim
        b1 = np.zeros(self.hidden_layer_size)
        W2 = np.random.randn(self.hidden_layer_size, num_classes) / np.sqrt(self.hidden_layer_size)
        b2 = np.zeros(num_classes)

        cache_W2 = 0
        cache_b2 = 0
        cache_W1 = 0
        cache_b1 = 0

        for epoch in range(num_iterations):
            for i in range(int(num_batches)):
                # get batch 
                batch_x = X_train[ i * batch_size : (i * batch_size + batch_size) ,]
                batch_y = y_train_one_hot[ i * batch_size : (i * batch_size + batch_size) ,]
                # forward pass
                y_pred_batch, Z = self.forward(batch_x, W1, b1, W2, b2)
                # W2 update
                gW2 = self.derivative_w2(Z, batch_y, y_pred_batch) + reg * W2
                cache_W2 = decay_rate * cache_W2 + (1 - decay_rate) * gW2 * gW2
                W2 -= learning_rate * gW2 / (np.sqrt(cache_W2) + eps)
                # b2 update
                gb2 = self.derivative_b2(batch_y, y_pred_batch) + reg * b2
                cache_b2 = decay_rate * cache_b2 + (1 - decay_rate) * gb2 * gb2
                b2 -= learning_rate * gb2 / (np.sqrt(cache_b2) + eps)
                # W1 update
                gW1 = self.derivative_w1(batch_x, Z, batch_y, y_pred_batch, W2) + reg * W1
                cache_W1 = decay_rate * cache_W1 + (1 - decay_rate) * gW1 * gW1
                W1 -= learning_rate * gW1 / (np.sqrt(cache_W1) + eps)
                # b1 update
                gb1 = self.derivative_b1(Z, batch_y, y_pred_batch, W2) + reg * b1
                cache_b1 = decay_rate * cache_b1 + (1 - decay_rate) * gb1 * gb1
                b1 -= learning_rate * gb1 / (np.sqrt(cache_b1) + eps)

                if i % 20 == 0:
                    y_pred_test, _ = self.forward(X_test, W1, b1, W2, b2)
                    loss = self.cost(y_pred_test, y_test_one_hot)
                    self.costs.append(loss)
                    acc = self.accuracy(y_pred_test, y_test)
                    self.test_acc.append(acc)
                    print('Epoch: {}, Iteration: {}, Cost: {}, Test Accuracy Rate: {}'.format(
                                                                epoch, i, loss, acc))

        final, _ = self.forward(X_test, W1, b1, W2, b2)
        print('Final Test Accuracy: {}'.format(self.accuracy(final, y_test)))

        if show_cost:
            plt.plot(self.costs)
            plt.show()


    def predict(self, pred):
        return np.argmax(pred, axis = 1)

    def error_rate(self, pred, y_true):
        prediction = self.predict(pred)
        return np.mean(prediction != y_true)

    def accuracy(self, pred, y_true):
        prediction = self.predict(pred)
        return np.mean(prediction == y_true)

    def cost(self, p_y, t):
        tot = t * np.log(p_y)
        return -tot.sum()

    def one_hot(self, y):
        N = len(y)
        num_classes = len(set(y))
        ind = np.zeros((N, num_classes))
        for i in range(N):
            ind[i, y[i]] = 1
        return ind

    def forward(self, X, W1, b1, W2, b2):
        # relu
        Z = X.dot(W1) + b1 
        Z[Z < 0] = 0
        A = Z.dot(W2) + b2
        # softmax
        expA = np.exp(A)
        y = expA / expA.sum(axis = 1, keepdims = True)
        return y, Z

    def derivative_w2(self, Z, T, Y):
        return Z.T.dot(Y - T)

    def derivative_b2(self, T, Y):
        return (Y - T).sum(axis = 0)

    def derivative_w1(self, X, Z, T, Y, W2):
        return X.T.dot((( Y - T ).dot(W2.T) * np.sign(Z))) 

    def derivative_b1(self, Z, T, Y, W2):
        return (( Y - T ).dot(W2.T) * np.sign(Z)).sum(axis = 0)


if __name__ == '__main__':

    X, y = get_normalized_data(train)
    model = NeuralNet(hidden_layer_size=300)
    model.fit(X, y, show_cost=True, num_iterations=30)

