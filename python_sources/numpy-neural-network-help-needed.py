# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

class Layer:
    def __init__(self):
        pass
    
    def forward(self, input):
        return input
    
    def backward(self, input, grad):
        m = input.shape[0]
        d = np.eye(m)
        
        return np.dot(grad, d)

class ReLU(Layer):
    def __init__(self):
        pass
    
    def forward(self, input):
        return np.maximum(input, np.zeros(input.shape))
    
    def backward(self, input, grad):
        r_grad = input > 0
        return grad * r_grad

class Dense(Layer):
    def __init__(self, in_size, out_size, eta=0.1):
        self.w = np.random.randn(in_size, out_size) * 0.01
        self.b = np.zeros(out_size)
        self.eta = eta

    def forward(self, input):
        return np.dot(input, self.w) + self.b
    
    def backward(self, input, grad):
        grad_input = -np.dot(grad, self.w.T)
        
        grad_w = -np.dot(input.T, grad)/input.shape[0]
        grad_b = np.sum(grad, axis=0)
        
        self.w = self.w - self.eta * grad_w
        self.b = self.b - self.eta * grad_b
        
        return grad_input

class Sequential(Layer):
    def __init__(self):
        self.network = []
        self.activations = []
    
    def add(self, model):
        self.network.append(model)

    def forward(self, input):
        self.activations = [input]
        
        for layer in self.network:
            input = layer.forward(input)
            self.activations.append(input)
        
        return self.activations[-1]

    def backward(self, input, grad):
        for i in reversed(range(len(self.network))):
            grad = self.network[i].backward(self.activations[i], grad)
        
        return grad

class SoftmaxCrossEntropy(Layer):
    def __init__(self):
        pass
    
    def forward(self, logits, y):
        m = logits.shape[0]
        
        l = logits[np.arange(m),y]
        e = -l + np.log(np.sum(np.exp(logits), axis=-1))
        
        return e

    def backward(self, logits, y):
        m = logits.shape[0]
        
        ones = np.zeros_like(logits)
        ones[np.arange(m),y] = 1
        
        softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
        
        return (-ones + softmax) / m

class StochasticGradientDescent:
    def __init__(self, model, loss, batch_size=64, epochs=10):
        self.model = model
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs

    def _iterate_minibatch(self, X, y):
        size = X.shape[0]
        max_batch = int(size / self.batch_size) 
        batch_num = 0
        
        while batch_num < max_batch:
            batch_pos = batch_num * self.batch_size
            
            X_batch = X[batch_pos:batch_pos+self.batch_size]
            y_batch = y[batch_pos:batch_pos+self.batch_size]
            
            yield X_batch, y_batch, batch_num
            
            batch_num += 1

    def fit(self, X_train, y_train, X_val, y_val):
        for epoch in range(self.epochs):
            losses = []
            acc = []
            
            for X_batch, y_batch, batch_num in self._iterate_minibatch(X_train, y_train):
                logits = self.model.forward(X_batch)
                            
                loss = self.loss.forward(logits, y_batch)
                loss_grad = self.loss.backward(logits, y_batch)
                
                self.model.backward(X_batch, loss_grad)
        
                mean_loss = np.mean(loss)
                losses.append(mean_loss)
                
                error = mean_squared_error(y_batch, np.argmax(logits, axis=1))
                acc.append(error)
                
                print("Epoch: {}, Batch: {}, Avg Loss: {}, Error: {}".format(epoch, batch_num, mean_loss, error))
        
            acc = []
            
            for X_batch, y_batch, batch_num in self._iterate_minibatch(X_val, y_val):                
                logits = self.model.forward(X_batch)
                error = mean_squared_error(y_batch, np.argmax(logits, axis=1))
                                
                acc.append(error)
            
            print("Epoch {} finished! Avg Loss: {}, Val Error: {}".format(epoch, np.mean(losses), np.mean(acc)))

        print("Model Trained!")
        
        return self.model

def train():
    dataset = pd.read_csv('../input/train.csv')
    
    X, y = dataset.iloc[:,1:], dataset.iloc[:,:1]
    
    del dataset
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    del X, y
    
    model = Sequential()
    
    model.add(Dense(784, 100))
    model.add(ReLU())
    model.add(Dense(100, 200))
    model.add(ReLU())
    model.add(Dense(200,10))
    
    loss = SoftmaxCrossEntropy()
    
    sgd = StochasticGradientDescent(model, loss, epochs=5)
    sgd.fit(X_train, y_train, X_val, y_val)
    
    return model

if __name__ == '__main__':
    train()










