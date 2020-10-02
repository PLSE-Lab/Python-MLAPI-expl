#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import gzip
import numpy as np
import random
import matplotlib.pyplot as plt

import sys
sys.executable


# In[ ]:


def load_data():
    f = open('../input/mnistpkl/mnist.pkl', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)


# In[ ]:


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    
    # see 'vectorized_result' function below
    training_results = [vectorized_result(y) for y in tr_d[1]]
    
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)


# In[ ]:


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


# In[ ]:


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data):
        # data to be used to train the network
        training_data = list(training_data)
        n = len(training_data)
        
        # data to check performance at the end of each epoch
        test_data = list(test_data)
        n_test = len(test_data)
        
        
        evaluation_accuracy = []
        training_accuracy = []
        for j in range(epochs):
            # shuffle to avoid effects of undesired correlations
            random.shuffle(training_data)
            
            # split data into ``mini_batches``
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            # each of them is used to estimate the gradient of the 
            # cost function in order to perform stochastic gradient
            # descent (SGD)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            correct = self.accuracy(test_data)
            print("Epoch {} : {} / {}".format(j,correct,n_test));
            evaluation_accuracy.append(correct/n_test)
            
            correct = self.accuracy(training_data, convert=True)
            training_accuracy.append(correct/n)
            
        return training_accuracy, evaluation_accuracy
    


# In[ ]:


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data):
        # data to be used to train the network
        training_data = list(training_data)
        n = len(training_data)
        
        # data to check performance at the end of each epoch
        test_data = list(test_data)
        n_test = len(test_data)
        
        
        evaluation_accuracy = []
        training_accuracy = []
        for j in range(epochs):
            # shuffle to avoid effects of undesired correlations
            random.shuffle(training_data)
            
            # split data into ``mini_batches``
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            # each of them is used to estimate the gradient of the 
            # cost function in order to perform stochastic gradient
            # descent (SGD)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            correct = self.accuracy(test_data)
            print("Epoch {} : {} / {}".format(j,correct,n_test));
            evaluation_accuracy.append(correct/n_test)
            
            correct = self.accuracy(training_data, convert=True)
            training_accuracy.append(correct/n)
            
        return training_accuracy, evaluation_accuracy

    def update_mini_batch(self, mini_batch, eta):    
        # define the gradient vectors (with respect to the biases and
        # the weights)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # use the minibatch (passed as argument) to evaluate the 
        # gradients through backpropagation (see function `backprop`
        # below)
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        # change the weights of the `Network` object
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        #
        # 1. feedforward: calculate inputs and activities at each layer
        #
        activation = x    # this is the input layer (passed as argument)
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        #
        # 2. backward pass: calculate the ``error`` at each layer
        #      by propagating errors back
        #
        delta = self.cost_derivative(activations[-1], y) *             sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # Note that the numeration of layers, here grows from the output
        # to the input layer (Python convention for negative indices in
        # lists)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def cost_derivative(self, output_activations, y):
        return (output_activations-y)


    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]

        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy
    
    
    def show_test(self, data):
        labels = np.array([np.argmax(self.feedforward(x)) for (x,y) in data])
        ground_truth = np.array([y for (x,y) in data])
        accuracy = np.sum(labels == ground_truth)/labels.size
        num_errors = min(np.sum(labels != ground_truth), 20)
        errors = np.random.choice(np.nonzero(labels != ground_truth)[0], num_errors, replace=False)
        correct = np.random.choice(
            np.nonzero(labels == ground_truth)[0], 60 - num_errors, replace=False)
        stimuli = np.hstack((errors, correct))
        np.random.shuffle(stimuli)
        plt.style.use('grayscale')
        num_columns = 10
        num_rows = 6
        fig, axes = plt.subplots(num_rows, 10, figsize=(20, 3 * num_rows))
        for idx, ax in zip(stimuli, axes.ravel()):
            if idx in errors:
                c = 'r'
            else:
                c = 'k'
            ax.matshow(np.reshape(1-data[idx][0], (28,28)))
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title('Prediction: '+str(labels[idx])+
                     ' \n Truth: '+str(ground_truth[idx]), color=c)
        plt.suptitle("Model accuracy: "+str(accuracy*100)+" %")
        plt.show()


# In[ ]:


#### Activation function and its derivative
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


# In[ ]:


# load the data
training_data, validation_data, test_data = load_data_wrapper()

# set the network up
net = Network([784, 100, 60, 10])

# set the parameters
#c1c2:epochs = 30
epochs = 30
#c1:mini_batch_size = 10
mini_batch_size = 1
#c1c2c4:eta = 0.1
#c3eta = 0.05
eta = .1

### convert the zip iterators into lists (so that we can use them multiple times)
n_training = 50000   # <= 50'000
training_data = list(training_data)[:n_training]

n_test = 10000       # <= 10'000
test_data = list(test_data)[:n_test]
validation_data = list(validation_data)[:n_test]

# see what's the ratio of correct classifications vs total test data
training_accuracy, evaluation_accuracy = net.SGD(training_data, epochs, mini_batch_size, eta, test_data)

# show sample of correct/wrong classifications
net.show_test(test_data)


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(0, 30, 1), training_accuracy, color='#FFA933',
            label="Accuracy on the training data")
ax.plot(np.arange(0, 30, 1), evaluation_accuracy, color='#2A6EA6', 
            label="Accuracy on the test data")
ax.set_xlim(0, epochs)
ax.set_xlabel('Epoch')
ax.set_ylim(0.5, 1)
ax.set_title('Classification accuracy')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


print(training_accuracy)


# In[ ]:


print(evaluation_accuracy)

