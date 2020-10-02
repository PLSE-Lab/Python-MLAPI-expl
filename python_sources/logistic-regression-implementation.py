import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/PimaIndians.csv')
data.test = [1 if diagnosis == 'positif' else 0 for diagnosis in data.test]

Y = np.transpose([data.test])
data.drop('test', axis=1, inplace=True)
X = data.values
X = (X - np.min(X)) / np.max(X) - np.min(X)

X, X_evaluate, Y, Y_evaluate = train_test_split(X, Y, test_size=0.15, random_state=42, shuffle=True)

m = X.shape[0]
n = X.shape[1]

w = np.random.randn(n, 1)
b = 0.0


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_propagate(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)


def predict(X, activations):
    predictions = np.zeros((X.shape[0], 1))

    for i in range(X.shape[0]):
        if activations[i, 0] >= 0.5:
            predictions[i, 0] = 1
        else:
            predictions[i, 0] = 0

    return predictions


def compute_efficiency(X, Y, parameters):
    activations = forward_propagate(X, parameters['weights'], parameters['bias'])
    predictions = predict(X, activations)

    successes = 0
    training_examples = X.shape[0]

    for i in range(training_examples):
        if predictions[i, 0] == Y[i, 0]:
            successes = successes + 1

    return (successes / training_examples) * 100


def compute_cost(activations):
    return np.sum(-Y * np.log(activations) - ((1 - Y) * np.log(1 - activations))) / m


def plot_cost(costs, indexes):
    plt.plot(indexes, costs)
    plt.xticks(indexes, rotation='vertical')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()


def compute_gradients(activations):
    gradient_weights = np.dot(X.T, (activations - Y)) / m
    gradient_bias = np.sum(activations - Y) / m
    return {'weights': gradient_weights, 'bias': gradient_bias}


def gradient_descent(w, b, alpha, iterations):
    cost_list = []
    index_list = []

    for i in range(iterations + 1):
        activations = forward_propagate(X, w, b)
        cost = compute_cost(activations)
        gradients = compute_gradients(activations)

        w = w - alpha * gradients['weights']
        b = b - alpha * gradients['bias']

        if i % 10 == 0:
            cost_list.append(cost)
            index_list.append(i)

    parameters = {'weights': w, 'bias': b}
    return parameters, cost_list, index_list


parameters, cost_list, index_list = gradient_descent(w, b, alpha=5, iterations=1000)
plot_cost(cost_list, index_list)

training_efficiency = compute_efficiency(X, Y, parameters)
evaluation_efficiency = compute_efficiency(X_evaluate, Y_evaluate, parameters)

print('EFFICIENCY (Training): \t{}% '.format(training_efficiency))
print('EFFICIENCY (Testing): \t{}% '.format(evaluation_efficiency))