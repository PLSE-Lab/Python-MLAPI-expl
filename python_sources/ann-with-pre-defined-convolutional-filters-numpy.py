#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


IMG_SIZE = 28
VERSION = 'simple'


# In[ ]:


X_train = np.loadtxt('../input/train.csv', skiprows=1, delimiter=',')
X_train = np.delete(X_train, 0, axis=1)
labels = np.loadtxt('../input/train.csv', usecols= 0, skiprows=1, delimiter=',', dtype= np.int)
X_test = np.loadtxt('../input/test.csv', skiprows=1, delimiter=',')


# In[ ]:


def standarize_normalize_set(dataset):
    return dataset / 255.0
def reshape_x_for_cnn(dataset, dim1, dim2):
    return dataset.reshape(-1, dim1, dim2, 1)


# In[ ]:


def one_hot_encoding(dataset, dim):
    dataset_encoded = np.zeros((len(dataset), dim))
    dataset_encoded[np.arange(len(dataset)), labels] = 1
    return dataset_encoded


# In[ ]:


def softmax(matrix):
    e_x = np.exp(matrix - np.max(matrix))
    return e_x / e_x.sum()

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def compute_multiclass_loss(Y, Y_hat):

    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[0]
    L = -(1/m) * L_sum

    return L


# In[ ]:


def train_ann(X_train, y_train, learning_rate = 0.1, epochs=1000):

    n_x = X_train.shape[1]
    n_h = 45
    n_h2 = 14
    digits = 10

    W1 = np.random.randn(n_x, n_h)
    b1 = np.zeros((1, n_h))

    W2 = np.random.randn(n_h, n_h2)
    b2 = np.zeros((1, n_h2))

    W3 = np.random.randn(n_h2, digits)
    b3 = np.zeros((1, digits))

    for i in range(epochs):

        m = X_train.shape[1]

        Z1 = np.matmul(X_train, W1) + b1
        A1 = sigmoid(Z1)

        Z2 = np.matmul(A1, W2) + b2
        A2 = sigmoid(Z2)

        Z3 = np.matmul(A2, W3) + b3
        A3 = np.exp(Z3) / np.sum(np.exp(Z3), axis=1, keepdims=True)

        cost = compute_multiclass_loss(y_train, A3)

        dZ3 = A3-y_train
        dW3 = (1./m) * np.matmul(A2.T, dZ3)
        db3 = (1./m) * np.sum(dZ3, axis=0, keepdims=True)

        dA2 = np.matmul(dZ3, W3.T)
        dZ2 = dA2 * sigmoid(Z2) * (1 - sigmoid(Z2))
        dW2 = (1./m) * np.matmul(A1.T, dZ2)
        db2 = (1./m) * np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.matmul(dZ2, W2.T)
        dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
        dW1 = (1./m) * np.matmul(X_train.T, dZ1)
        db1 = (1./m) * np.sum(dZ1, axis=0, keepdims=True)

        W3 = W3 - learning_rate * dW3
        b3 = b3 - learning_rate * db3
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1

        if (i % 100 == 0):
            print("Epoch", i, "cost: ", cost)
            
    return W1, b1, W2, b2, W3, b3


# In[ ]:


def predict_new_data(X_test, W1, b1, W2, b2, W3, b3):
    Z1 = np.matmul(X_test, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.matmul(A1, W2) + b2
    A2 = sigmoid(Z2)

    Z3 = np.matmul(A2, W3) + b3
    A3 = np.exp(Z3) / np.sum(np.exp(Z3), axis=1, keepdims=True)

    y_hat= np.argmax(A3, axis=1)
    
    return y_hat


# In[ ]:


X_train = standarize_normalize_set(X_train)
y_train = one_hot_encoding(labels, 10)
W1, b1, W2, b2, W3, b3 = train_ann(X_train, y_train,learning_rate = 0.01, epochs=10000)


# In[ ]:


X_test = standarize_normalize_set(X_test)
y_pred = predict_new_data(X_test, W1, b1, W2, b2, W3, b3)


# In[ ]:


pd.DataFrame({'ImageId': np.arange(1, len(y_pred)+1), 'Label': y_pred}).to_csv("Submission_normal.csv", index=False)


# In[ ]:


del W1, b1, W2, b2, W3, b3


# # Method 2

# In[ ]:


X_train = np.loadtxt('../input/train.csv', skiprows=1, delimiter=',')
X_train = np.delete(X_train, 0, axis=1)
labels = np.loadtxt('../input/train.csv', usecols= 0, skiprows=1, delimiter=',', dtype= np.int)
X_test = np.loadtxt('../input/test.csv', skiprows=1, delimiter=',')


# In[ ]:


y_train = one_hot_encoding(labels, 10)

X_train = standarize_normalize_set(X_train)
X_test = standarize_normalize_set(X_test)
X_train = reshape_x_for_cnn(X_train, IMG_SIZE, IMG_SIZE)
X_test = reshape_x_for_cnn(X_test, IMG_SIZE, IMG_SIZE)


# In[ ]:


def convulving_matrix(input_matrix, conv_kernel, stride=(1, 1), pad_method='same', bias=1):

    input_h, input_w, input_d = input_matrix.shape[0], input_matrix.shape[1], input_matrix.shape[2]
    kernel_h, kernel_w, kernel_d = conv_kernel.shape[0], conv_kernel.shape[1], conv_kernel.shape[2]
    stride_h, stride_w = stride[0], stride[1]

    if pad_method == 'same':
        # same is the method to returns pciture of the same size
        # so we are zero-padding around it
        output_h = int(np.ceil(input_matrix.shape[0] / float(stride[0])))
        output_w = int(np.ceil(input_matrix.shape[1] / float(stride[1])))
        output_d = input_d
        output = np.zeros((output_h, output_w, output_d))

        pad_h = max((output_h - 1) * stride[0] + conv_kernel.shape[0] - input_h, 0)
        pad_h_offset = int(np.floor(pad_h/2))  
        pad_w = max((output_w - 1) * stride[1] + conv_kernel.shape[1] - input_w, 0)
        pad_w_offset = int(np.floor(pad_w/2))

        padded_matrix = np.zeros((output_h + pad_h, output_w + pad_w, input_d))

        for l in range(input_d):
            for i in range(input_h):
                for j in range(input_w):
                    padded_matrix[i + pad_h_offset, j + pad_w_offset, l] = input_matrix[i, j, l]

        for l in range(output_d):
            for i in range(output_h):
                for j in range(output_w):
                    curr_region = padded_matrix[i*stride_h : i*stride_h + kernel_h, j*stride_w : j*stride_w + kernel_w, l]
                    output[i, j, l] = (conv_kernel[..., l] * curr_region).sum()

    elif pad_method == 'valid':

        output_h = int(np.ceil((input_matrix.shape[0] - kernel_h + 1) / float(stride[0])))
        output_w = int(np.ceil((input_matrix.shape[1] - kernel_w + 1) / float(stride[1])))
        output = np.zeros((output_h, output_w, layer+1))

        for l in range(layer + 1):
            for i in range(output_h):
                for j in range(output_w): 
                    curr_region = input_matrix[i*stride_h:i*stride_h+kernel_h, j*stride_w:j*stride_w+kernel_w, l]
                    output[i, j, l] = (conv_kernel[..., l] * curr_region).sum()

    output = np.sum(output, axis=2) + bias

    return output


# In[ ]:


def convulve_whole_dataset(matrix, method='simple'):
      
    range_ = matrix.shape[0]
    if method == 'simple':
        K  = np.array([0, -1, 0, -1, 5, -1, 0, -1, 0]).reshape((3, 3, 1))
        I = np.zeros((range_, matrix.shape[1], matrix.shape[2]))

        for number in range(range_):
            I[number, :, :] = convulving_matrix(matrix[number], K)
        G = I.reshape(-1, IMG_SIZE * IMG_SIZE)
 
    else:
        Kx  = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape((3, 3, 1))
        Ky = np.array([1, 2, 1, 0, 0, 0, -1, -2, -1]).reshape((3, 3, 1))

        Ix = np.zeros((range_, matrix.shape[1], matrix.shape[2]))
        Iy = np.zeros((range_, matrix.shape[1], matrix.shape[2]))

        for number in range(range_):
            Ix[number, :, :] = convulving_matrix(matrix[number], Kx)
            Iy[number, :, :] = convulving_matrix(matrix[number], Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        G = G.reshape(-1, IMG_SIZE * IMG_SIZE)
        
        theta = np.arctan2(Iy, Ix)
    
    return G


# In[ ]:


def visualize_example(dataset_x, dataset_y, object_):
    print(dataset_y[object_])
    plt.imshow(dataset_x[object_, :].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')


# In[ ]:


visualize_example(X_train, y_train, 10)


# In[ ]:


X_train_ = convulve_whole_dataset(X_train, VERSION)
X_test_ = convulve_whole_dataset(X_test, VERSION)
y_train_ = y_train.copy()


# In[ ]:


visualize_example(X_train_, y_train_, 10)


# In[ ]:


W1, b1, W2, b2, W3, b3 = train_ann(X_train_, y_train_,learning_rate = 0.01, epochs=10000)


# In[ ]:


y_pred = predict_new_data(X_test_, W1, b1, W2, b2, W3, b3)


# In[ ]:


pd.DataFrame({'ImageId': np.arange(1, len(y_pred)+1), 'Label': y_pred}).to_csv("Submission_{}.csv".format(VERSION), index=False)

