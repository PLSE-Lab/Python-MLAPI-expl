#!/usr/bin/env python
# coding: utf-8

# # NNNN - a Naive Neutral Network iNtroduction
# 
# I've just completed Andrew Ng's Deep Learning specialization on Coursera, and I want to cement my knowledge via reimplementing main concepts using the dataset - the "Hello World1!" of image recognition - and the grader of this competition. 
# 
# I used to be a software developer in a previous life, but I haven't seen a blank code page before me for 15+ years :) Completing this notebook took 3-4 longer than I first estimated. Honestly speaking, I wanted to give up two or three times in the process when problems seemed insurmountable and bugs unsquashable.
# 
# The basic idea was to go from the simplest possible model - one layer of softmax activations - to some degree of model complexity, that would give a reasonable accuracy, and to try out different optimization techniques. I tried to create a mini framework that can be easily configured by tweaking a few parameters - it is a fantastic power of the software systems. However, it gets painful to train the model beyond three layers - apparently, numpy isn't accelerated by GPU. 
# 
# I've hidden most of supporting code from the published version. However, I hope it is still can be accessed if you fork the kernel (it is my first kernel and I'm only learning Kaggle). 
# 
# If you see a bug, please let me know at alukashenkov@gmail.com. If you find this piece of use and interest, please upvote.

# ## Acknowledgments
# I took much inspiration and borrowed heavily from programming assignments of the first three courses in the [Deep Learning Specialization](https://www.deeplearning.ai/courses/).
# 
# I also borrowed many ideas from [Introduction to CNN Keras - 0.997 (top 6%)](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6) kernel. It also was beneficial as 101 on Kaggle submissions.
# 
# [Softmax Regression](http://saitcelebi.com/tut/output/part2.html) by Sait Celebi was very helpful to make sense of softmax function derivatives.

# ## Getting Data and Preparing Train/CV Sets
# Importing libraries.

# In[ ]:


import numpy as np
import pandas as pd

import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# In[ ]:


import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(Z):
    
    A = 1/(1 + np.exp(-Z))
    
    cache = Z
    return A, cache

def relu(Z):
    
    A = np.maximum(0,Z)
    
    cache = Z 
    return A, cache

def softmax(Z):
    
    e_z = np.exp(Z - np.max(Z))
    A = e_z / e_z.sum(axis = 0)
    
    cache = Z
    
    return A, cache

def softmax_backward(A, dA, cache):
    
    Z = cache
    c, m = A.shape
    
    dZ = np.zeros(A.shape)

    for i in range(m):
        matrix = np.matmul(A[:, i].reshape(c,1), np.ones((1, c))) * (np.identity(c) - np.matmul(np.ones((c, 1)), A[:, i].reshape(c,1).T))    
        dZ[:, i] = np.matmul(matrix, dA[:, i])
    
    assert(dZ.shape == dA.shape)

    return dZ

def relu_backward(dA, cache):
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    return dZ

def sigmoid_backward(dA, cache):
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    return dZ

def initialize_parameters_deep(layer_dims):
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters

def initialize_adam(parameters) :
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
    
    return v, s

def linear_forward(A, W, b):
    
    Z = np.dot(W, A) + b     # Compute linear part of the forfard propagation
    assert(Z.shape == (W.shape[0], A.shape[1]))
    
    cache = (A, W, b)        # Return the product and the params
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation, keep_prob = 1):
    
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)   
    elif activation == "relu":
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        A, activation_cache = softmax(Z)
    
    D = np.random.rand(np.shape(A)[0], np.shape(A)[1])   # Initialize matrix D = np.random.rand(..., ...)
    if keep_prob != 1:
        D = D < keep_prob                                # Convert entries of D to 0 or 1 (using keep_prob as the threshold)
        A = A * D                                        # Shut down some neurons of A1
        A = A / keep_prob                                # Scale the value of neurons that haven't been shut down
        
    cache = (linear_cache, activation_cache, D)

    return A, cache

def L_model_forward(X, parameters, keep_prob = 1):
    
    caches = []
    A = X
    L = len(parameters) // 2            # number of layers in the neural network
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], "relu", 
                                             keep_prob)
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], "softmax")
    caches.append(cache)
            
    return AL, caches

def compute_cost(AL, Y):
    
    m = Y.shape[1]

    # Compute loss from aL and y. Hardcoding for softmax
    cost = (-1./m) * np.sum(Y * np.log(AL))

    cost = np.squeeze(cost)      # To make sure the cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

def compute_cost_with_regularization(AL, Y, parameters, lambd):

    m = Y.shape[1]
    L = len(parameters) // 2   # number of layers in the neural network

    cross_entropy_cost = compute_cost(AL, Y) # This gives you the cross-entropy part of the cost
    L2_regularization_cost = 0

    # Update rule for each parameter.
    for l in range(L):
        W = parameters["W" + str(l + 1)]
        L2_regularization_cost = L2_regularization_cost + np.sum(np.square(W))
    
    cost = cross_entropy_cost + lambd / (2 * m) * L2_regularization_cost
    
    return cost

def linear_backward(dZ, cache, lambd):

    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ, A_prev.T) / m + lambd / m * W
    db = np.sum(dZ, axis = 1, keepdims = True) / m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(A, dA, cache, lambd, activation, keep_prob = 1):

    linear_cache, activation_cache, D = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == "softmax":
        dZ = softmax_backward(A, dA, activation_cache)

    if keep_prob != 1:
        dA = dA * D           # Apply mask D to shut down the same neurons as during the forward propagation
        dA = dA / keep_prob   # Scale the value of neurons that haven't been shut down
        
    dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
        
    return dA_prev, dW, db

def L_model_backward_with_regularization(AL, Y, caches, lambd, keep_prob):
    
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]

    # Initializing the backpropagation    
    dAL = np.divide(- Y, AL)            # Hardcoding for softmax
    assert(dAL.shape == AL.shape)
    
    current_cache = caches[L - 1]
    grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = linear_activation_backward(AL, dAL, current_cache, lambd,
                                                                                              "softmax")
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(AL, grads["dA"+str(l+1)], current_cache, lambd,
                                                                    "relu", keep_prob)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - math.pow(beta1, t))
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - math.pow(beta1, t))

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * grads["dW" + str(l+1)] * grads["dW" + str(l+1)]
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * grads["db" + str(l+1)] * grads["db" + str(l+1)]

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - math.pow(beta2, t))
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - math.pow(beta2, t))
        
        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / (epsilon + np.sqrt(s_corrected["dW" + str(l+1)]))
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)] / (epsilon + np.sqrt(s_corrected["db" + str(l+1)]))

    return parameters, v, s

def predict(X, y, parameters):
    
    m = X.shape[1]
    n = len(parameters) # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # Convert probabilities into labels
    p = probas.argmax(axis = 0).reshape(1,-1)
    
    assert(p.shape == y.shape)

    print("Accuracy: " + str(np.sum((p == y)/m)))
        
    return p

def print_10_mislabeled_images(X, y, p):
    
    img_size = 28 
    num_images_to_show = 10
    
    a = p - y
    mislabeled_indices = np.asarray(np.where(a != 0))

    for i in range(num_images_to_show):
        sample = np.random.randint(0, mislabeled_indices.shape[1])
        index = mislabeled_indices[1][sample]
        
        plt.subplot(2, num_images_to_show, i + 1)
        plt.imshow(X[:, index].reshape(img_size, img_size), interpolation = 'nearest')
        plt.rcParams['figure.figsize'] = (50.0, 50.0) # set default size of plots
        plt.axis('off')
        plt.title("Prediction: " + str(p[0, index]) + " \n Label: " + str(y[0, index]))

def print_10_images(X, y):
    
    img_size = 28 
    num_images_to_show = 10
    m = X.shape[1]
    
    for i in range(num_images_to_show):
        sample = np.random.randint(0, m - 1)
        
        plt.subplot(2, num_images_to_show, i + 1)
        plt.imshow(X[:, sample].reshape(img_size, img_size), interpolation = 'nearest')
        plt.rcParams['figure.figsize'] = (50.0, 50.0) # set default size of plots
        plt.axis('off')
        plt.title("Index: " + str(sample) + " \n Label: " + str(y[0, sample]))

def random_mini_batches(X, Y, mini_batch_size = 64):
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((-1,m))

    # Partition (shuffled_X, shuffled_Y). Minus the end case.
    
    # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = math.floor(m/mini_batch_size) 
    
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


# Definig some useful constants.

# In[ ]:


img_size = 28               # Known from dataset description
n_classes = 10              # We have 10 digits represented in dataset


# Load train data.

# In[ ]:


# Getting data as Numpy object and transpose it to have training examples as columns of the matrix
train_data = pd.read_csv('../input/train.csv').values.T

# Train data shape
m = train_data.shape[1]

# Labels
Y_train_data = train_data[0, :]
Y_train_data = Y_train_data.reshape(1, m)

#Pixels
X_train_data = train_data[1:, :]
X_train_data = X_train_data/255.0     #Normalising values

# Little clean up
del train_data


# Let's look at what we've got to work with.

# In[ ]:


plt.rcParams['figure.figsize'] = (50.0, 50.0) # set default size of plots
    
print("Number of samples in training data is:", m)

print("Ten random images with their labels:")
print_10_images(X_train_data, Y_train_data)


# Les's properly split train data into training ant cross-validation sets.

# In[ ]:


train_proportion = 0.05      # Proportion of CV set to be extracted from training data

# Initializing arrays
train_inds = np.zeros((1, m), dtype=bool)

#Getting unique values of the labels
values = np.unique(Y_train_data)

# Creating indices for trainig and CV samples
for value in values:
    value_inds = np.nonzero(Y_train_data == value)[1]
    np.random.shuffle(value_inds)
    n = int(train_proportion * len(value_inds))

    train_inds[:, value_inds[n:]] = True

# Now split provided dataset
X_train = X_train_data[:, train_inds[0,:]]
X_cv = X_train_data[:, ~train_inds[0,:]]
Y_train = Y_train_data[:, train_inds[0,:]]
Y_cv = Y_train_data[:, ~train_inds[0,:]]

# And create one-hot representations for labels
Y_train_oh = np.eye(n_classes)[Y_train.reshape(-1)].T
Y_cv_oh = np.eye(n_classes)[Y_cv.reshape(-1)].T


# Now visually checking training and CV sets labels distribution. They are expected to be fairly equal.

# In[ ]:


plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
sns.countplot(Y_train[0])


# In[ ]:


sns.countplot(Y_cv[0])


# ## The Model
# Here comes the core `L_layer_model()` function that iterates through epoches and minibatches.

# In[ ]:


def L_layer_model(X, Y, layers_dims, learning_rate, lr_decay_base, batch_size, num_epoch, lambd = 0, keep_prob = 1, print_cost = False):
    
    np.random.seed(1)
    costs = []                      # keep track of cost
    t = 0                           # initializing the counter required for Adam update
    lr = learning_rate
    batch_num = 0
    
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    v, s = initialize_adam(parameters)
    
    # Loop (gradient descent)
    for i in range(0, num_epoch):
        
        minibatches = random_mini_batches(X, Y, batch_size)

        for minibatch in minibatches:

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SOFTMAX.
            AL, caches = L_model_forward(X, parameters, keep_prob)
        
            # Compute cost. As the last activation is SOFTMAX use appropriate formula
            cost = compute_cost_with_regularization(AL, Y, parameters, lambd)
            costs.append(cost)
    
            # Backward propagation.
            grads = L_model_backward_with_regularization(AL, Y, caches, lambd, keep_prob)
            
            # Update parameters.
            t = t + 1 # Adam counter
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, lr)
            
            if print_cost:
                print ("Cost in epoch %i after batch %i: %f" %(i, batch_num, cost))
                batch_num = batch_num + 1
        
        # Update learning rate
        lr = lr * (1 / (1 + lr_decay_base / num_epoch))
        
        # Reset batch count
        batch_num = 0

    return parameters, costs


# Specifying model parameters.

# In[ ]:


# Define a model
n_x = img_size * img_size          # Square image with one channel 
n_y = n_classes                    # Ten output classes

# Specify model configuration - add number of hidden layers' activations beween n_x and n_y
# The model would use relu activation for hidden layers and softmax for the output one
layers_dims = [n_x, 1024, 512, 256, n_y]
    
# Define learning parameters
learning_rate = 0.003               
lr_decay_base = 0          
batch_size = m          
num_epoch = 400
lambd = 0.7                # Weight decay
keep_prob = 0.8            # Dropout ratio


# Running the model.

# In[ ]:


get_ipython().run_line_magic('time', 'parameters, costs = L_layer_model(X_train, Y_train_oh, layers_dims, learning_rate, lr_decay_base, batch_size, num_epoch, lambd, keep_prob, True)')


# Let's look at the cost graph.

# In[ ]:


# plot the cost
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate = " + str(learning_rate))
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.show()


# Predict on the training set.

# In[ ]:


pred_train = predict(X_train, Y_train, parameters)


# Sanity check - what images the model gets wrong? In my case, most of the images I see show weird digits representations.

# In[ ]:


plt.rcParams['figure.figsize'] = (50.0, 50.0) # set default size of plots

print_10_mislabeled_images(X_train, Y_train, pred_train)


# Predict on the cross-validation set.

# In[ ]:


pred_cv = predict(X_cv, Y_cv, parameters)


# In[ ]:


print_10_mislabeled_images(X_cv, Y_cv, pred_cv)


# ## Results Overview
# My quick and rather impatient testing shows the following results:
# 
# * A model with one output layer (inputs pixels are directly mapped to softmax activations) would give accuracy at around 90%. 
# * Adding one hidden layer (1024 neurons) with dropout regularisation moves accuracy up to 95%.
# * With two hidden layers (1024 and 512 neurons) trained for 5 epochs the model gives 97,5% accuracy. Further training only makes the model to overfit training data - I've seen accuracy around 99.93% on the training set, but still only 97,5% on the CV set.
# * The third hidden layer (1024, 512, and 256 neurons) improves accuracy just a little bit up to 98,1%. Training longer wouldn't help much to improve results and the model starts suffering from gradient explosion (or there is a bug in my code).

# ## Testing and Submitting

# In[ ]:


# Getting data as Numpy object and transpose it to have training examples as columns of the matrix
test_data = pd.read_csv('../input/test.csv').values.T

# Train data shape
m_test = test_data.shape[1]

#Pixels
X_test_data = test_data[:, :]
X_test_data = X_test_data/255.0     #Normalising values

# Little clean up
del test_data


# In[ ]:


# Forward propagation
probs_test, _ = L_model_forward(X_test_data, parameters)

# Convert probabilities into labels
p = probs_test.argmax(axis = 0)

results = pd.Series(p, name = "Label")

submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), results], axis = 1)

submission.to_csv("submission.csv", index = False)

