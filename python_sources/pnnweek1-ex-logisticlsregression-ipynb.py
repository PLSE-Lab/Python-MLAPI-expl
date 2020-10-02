#!/usr/bin/env python
# coding: utf-8

# # <center> KIT Praktikum NN: L2-regularized Logistic Least Squares Regression </center>
# 
# </br>
# On this exercise, you are going to apply what you learn from the `numpy` tutorial in the implementation of L2-regularized Logistic Least Squares Regression (LLSR). I will provide you the formula by now (you can do it yourself after the next lecture!!!), first you should use pens and papers to vectorize them. Then you implement the full of the classifier based on your vectorized version.
# 
# </br>
# <center><img src="https://github.com/thanhleha-kit/PraktikumNeuronaleNetze/blob/master/Images/LogisticRegression.png?raw=true" style="width:298px;height:275px"></center>
# 
# </br>
# L2-regularized Logistic Least Squares Regression is similar to the standard Logistic Regression: It is a binary classifier containing only one layer, mapping the input features to only one output using sigmoid function. The differents here are two things: 
# * Instead of the _binary crossentropy error_ for the loss, it uses the _squared error_.
# * It is applied the L2-regularization.
# 
# Note that we will do an SGD training for this exercise. More specifically:
# * There are $m$ data instance on the training set, each has $n$ input features. 
# * $x_{i}^{(j)}$ denotes the $i^{th}$ input feature of the $j^{th}$ data instance.
# * $y^{(j)}$ denotes the binary label ($0$ or $1$) of the $j^{th}$ data instance.
# * $w_{i}$ denotes the weight connecting the $i^{th}$ input feature to the output.
# * $b$ is the bias of the Logistic Least Squares Regression.
# 
# So the steps of an unvectorized version are:
# * The weights are initialized using Xavier Initialization, the bias can be initialized as 0.
# * Train over 5 epochs, each epoch we do those steps:
#   *  Loop over every data instance $x^{(j)}$:
#      * Calculate the output of the LLSR: $o^{(j)} = \sigma(\sum_{i=1}^{n} w_ix_i^{(j)} + b)$
#      * Calculate the cost: squared error $c^{(j)} = (y^{(j)} - o^{(j)})^2$
#      * The final loss function is L2-regularized: $l^{(j)} = \frac{1}{2}c^{(j)} + \frac{\lambda}{2}\sum_{i=1}^{n}w_i^2$. 
#      * Update the weights: 
#          * Loop over every weight $w_i$ and update once at a time: $w_i = w_i - \eta((o^{(j)}-y^{(j)})o^{(j)}(1-o^{(j)})x_i^{(j)} + \lambda w_i)$
#      * Update the bias: $b = b - \eta (o^{(j)}-y^{(j)})o^{(j)}(1-o^{(j)})$
#   *  Calculate the total loss (of the epoch): $L = \frac{1}{m}\sum_{j=1}^{m}l^{(j)}$. Print it out. 
# 

# The guideline is to avoid explicit for-loops. _Hint_: We cannot make the epoch loop disappears, but all other loops can be replaced by vectorization.

# First, import numpy and math:

# In[ ]:


import numpy as np
import math


# 
# We will use LLSR for the MNIST_SEVEN task: predict a $128\times 128$-pixel image of a handwritten digit whether it is a "7" or not. This is a binary classification task. I did the data reading for you. There is 5000 images, I split the first 4000 images for training, 500 images for tuning, 500 images for test. On this exercise we do not need to tune anything, so we'd leave the tuning (called the _dev set_) untouch. The first field is the label ("0"-"9") of the image, the rest are the grayscale value of each pixel. 
# 
# 
# Before running the below cell, you should change the `data_path` pointing to the correct location of your dataset csv file.

# In[ ]:


data_path = "../input/mnist_seven.csv"
data = np.genfromtxt(data_path, delimiter=",", dtype="uint8")
train, dev, test = data[:4000], data[4000:4500], data[4500:]


# In[ ]:


def normalize(dataset):
    X = dataset[:, 1:] / 255.     # Normalize input features
    Y = (dataset[:, 0] == 7) * 1  # Convert labels from 0-9 to Is7 (1) or IsNot7(0)
    return X.T,Y.reshape(1, -1)


# In[ ]:


X_train, Y_train = normalize(train)
print(X_train.shape)
print(Y_train.shape)

X_test, Y_test = normalize(test)
print(X_test.shape)
print(Y_test.shape)

# shuffle the training data since we do SGD
# we shuffle outside the training 
# since we want to compare unvectorized and vectorized versions
# It doesn't affect to batch training later
np.random.seed(8888)     # Do not change those seedings to make our results comparable
np.random.shuffle(train) 


# # Unvectorized Version of Stochastic Gradient Descent
# 
# First the unvectorized version of training:

# In[ ]:


def train_unvectorized(X_train, Y_train, lr=0.2, lambdar=0.0001, epochs=5):
    
    n = X_train.shape[0]
    m = X_train.shape[1]
    
    # Xavier Initialization
    np.random.seed(1234)
    w = np.random.randn(n) * (np.sqrt(2. / (n + 1)))
    b = 0

    for epoch in range(epochs):
        L = 0
        for j in range(m):   # Loop over every training instance
            # Forward pass
            # CODE HERE
            

            # Calculate the loss
            # CODE HERE
            
            
            # Backward pass and update the weights/bias
            # CODE HERE
            pass
        # Accumulate the total loss and print it
        L /= m
        print("Error of the epoch {0}: {1}".format(epoch + 1, L))
    
    return w, b
        


# And the (unvectorized) inference:

# In[ ]:


def test_unvectorized(X_test, Y_test, w, b):
    
    n_test = X_test.shape[0]
    m_test = X_test.shape[1]
    corrects = 0
    
    for j in range(m_test):
        
        # Forward pass
        # CODE HERE
        
        # Evaluate the outputs
        # CODE HERE
        
        pass
    print("Accuracy of our LLSR:" + str((corrects * 100.) / m_test) + "%")
    
    return corrects


# Test on our test data. The accuracy should be better than 89.2%. This high score 89.2% is the baseline, achieved by do nothing rather than predicting all images are not a "seven" :p.

# In[ ]:


w, b = train_unvectorized(X_train, Y_train)
_ = test_unvectorized(X_test, Y_test, w, b)


# # Vectorized Version of Stochastic Gradient Descent
# 
# Now we move to the vectorized version of training and inference, just replace for-loops and total-sums by $np.dot()$,  $np.sum()$ and the numpy pair-wise operations (you should do the vectorization using pens and papers first).

# In[ ]:


def train_vectorized(X_train, Y_train, lr=0.2, lambdar=0.0001, epochs=5):
    
    n = X_train.shape[0]
    m = X_train.shape[1]
    
    # Xavier Initialization
    np.random.seed(1234)
    w = np.random.randn(n) * (np.sqrt(2. / (n + 1)))
    b = 0

    for epoch in range(epochs):
        L = 0
        for j in range(m):

            # Forward pass
            # CODE HERE
            
            # Calculate the loss (for each instance - SGD) 
            # CODE HERE
            
            # Backward pass and update the weights/bias (for each instance - SGD) 
            # CODE HERE
            pass    
        L /= m
        # print("Error of the epoch {0}: {1}".format(epoch + 1, L))
    return w, b


# And the vectorized inference (short, clear and fast):

# In[ ]:


def test_vectorized(X_test, Y_test, w, b):
    
    m_test = X_test.shape[1]
    corrects = 0
    
    # CODE HERE
    
    print("Accuracy of our LLSR:" + str((corrects * 100.) / m_test) + "%")
    
    return corrects


# Those following runs should return exact the same outputs like the (unvectorized) training and inference before but in less than a second. The vectorization should be more effective (much faster) if this is not an one-layer logistic regression but a deep network.

# In[ ]:


w, b = train_vectorized(X_train, Y_train)
_ = test_vectorized(X_test, Y_test, w, b)


# # Vectorized Version of Batch Gradient Descent 
# 
# Here is the fully vectorized version, batch training (vectorizing over training instances). The formula (you might be able to derive them after the next lecture):
# 
# $$ z = w \cdot X + b $$
# 
# $$ o = \sigma(z) $$
# 
# $$ C = \frac{1}{2m}\sum_{j=1}^{m}(y^{(j)}-o^{(j)})^2 $$
# 
# $$ R = \frac{1}{2m}\sum_{i=1}^{n}w_i^2 $$
# 
# $$ L = C + \lambda R $$
# 
# $$ \frac{\partial C}{\partial z^{(j)}} = \frac{1}{m}(o^{(j)} - Y^{(j)}) * o^{(j)} * (1 - o^{(j)}) $$
# 
# $$ \frac{\partial z^{(j)}}{\partial w_i} = x_i $$
# 
# $$ \Rightarrow \frac{\partial C}{\partial w} = \frac{\partial C}{\partial z} \cdot X^T $$
# 
# $$ \frac{\partial R}{\partial w} = \frac{1}{m}w $$ 
# 
# $$ \Rightarrow \frac{\partial L}{\partial w} = \frac{\partial C}{\partial w} + \lambda\frac{\partial R}{\partial w} $$
# 
# $$ \frac{\partial z}{\partial b} = 1 $$
# 
# $$ \Rightarrow \frac{\partial L}{\partial b} = \frac{\partial C}{\partial b} = \sum_{j=1}^{m}(o^{(j)} - Y^{(j)}) * o^{(j)} * (1 - o^{(j)}) $$
# 
# $$ w = w - \eta * \frac{\partial L}{\partial w} $$
# 
# $$ b = b - \eta *  \frac{\partial L}{\partial b} $$

# In[ ]:


def train_batch(X_train, Y_train, lr=0.1, lambdar=0.0001, epochs=50):
    
    n = X_train.shape[0]
    m = X_train.shape[1]

    # Xavier Initialization
    np.random.seed(1234)
    w = np.random.randn(1, n) * (np.sqrt(2. / (n + 1)))
    b = 0
    
    L = 0

    for epoch in range(epochs):

        # Forward pass
        # CODE HERE

        # Calculate the loss 
        # CODE HERE
        
        # Backward pass and update the weights/bias
        # CODE HERE
        pass
        # print("Error of the epoch {0}: {1}".format(epoch + 1, L))
        
    return w, b
        


# Since it is a batch training and requires different hyperparameters, the result might not be comparable to the SGD trainings above. 

# In[ ]:


w_batch, b_batch = train_batch(X_train, Y_train, lr=2, lambdar=0.5, epochs=1001)
_ = test_vectorized(X_test, Y_test, w_batch, b_batch)


# One thing to compare: the speed. Try to run the same number of epochs (1000) with SGD, vectorized training, you can see it still takes a long time to run compared to the fully batch training.

# In[ ]:


w, b = train_vectorized(X_train, Y_train, epochs=1001)
_ = test_vectorized(X_test, Y_test, w, b)


# # Vectorized Version of Minibatch Gradient Descent
# 
# Finally, we can do minibatch training, it is the same as batch training (the formula) but one iteration runs over a subset of the whole dataset at a time, and those subsets (minibatches) are shuffled before training:

# In[ ]:


def train_minibatch(X_train, Y_train, batch_size=256, lr=0.1, lambdar=0.0001, epochs=50):
    
    n = X_train.shape[0]
    
    # Xavier Initialization
    np.random.seed(1234)
    w = np.random.randn(1, n) * (np.sqrt(2. / (n + 1)))
    b = 0

    L = 0
    for epoch in range(epochs):
        
        # Split into minibatches 
        # CODE HERE
        
        # We shuffle the minibatches of X and Y in the same way
        # CODE HERE
        
        # Now we can do the training, we cannot vectorize over different minibatches
        # They are like our "epochs"
        for i in range(None): # CODE HERE
            
            # Extract a minibatch to do training
            X_current = None # CODE HERE
            Y_current = None # CODE HERE
            m = X_current.shape[1]

            # Forward pass
            # CODE HERE  

            # Calculate the loss 
            # CODE HERE

            # Backward pass and update the weights/bias
            # CODE HERE

            # print("Error of the iteration {0}: {1}".format(None, L)) # CODE HERE

    return w, b


# Minibatch Training for this LLSR is very sensitive to hyperparameter choosing. Should use with early stopping. Do not supprise if the accurary is bad. Shuffling the minibatch also takes time, so do not run this with large number of epochs.

# In[ ]:


# Do not run this for more than 100 epochs!!!!!!!!!
w_minibatch, b_minibatch = train_minibatch(X_train, Y_train, batch_size=512, lr=0.001, lambdar=0.0001, epochs=30)
_ = test_vectorized(X_test, Y_test, w_minibatch, b_minibatch)


# In[ ]:




