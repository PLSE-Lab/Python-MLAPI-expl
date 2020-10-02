#!/usr/bin/env python
# coding: utf-8

# Table of Contents
# ---
# 
# 
# 
# > Introduction
# 
# 
# > Reading the MNIST data set
# 
# 
# > Extreme Learning Machine implementation
# 
# 
# > References
# 
# 
# 
# 
# 
# 
# 
# 

# # Introduction
# 
# This is a quick example of an **Extreme Learning Machine** implementation/solution to the MNIST handwritten digit digital recognizer problem.
# 
# I chose this dataset since a high accuracy on MNIST is regarded as a basic requirement of credibility in a classification algorithm.
# 
# Extreme learning machines are feedforward neural networks, that can be extremely easy to implement and offer decent results, considering the speed and simplicity of this algorithm compared to more complex solutions.
# 
# I think this example could be especially interesting to novice machine learning students mainly because of its simplicity.

# # Reading the MNIST dataset
# 
# The MNIST dataset contains a series of monochrome images **28x28** of handwritten digits, on each row of the dataset stored as a vector with 784 values, each representing a pixel value, the training data has an additional column containing the label associated with each image.

# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("../input/train.csv")
train.head()


# As we can see, each row has 785 columns, with the first being the label and the rest of them representing the  pixel values (28x28) of the image.
# 
# Next, we will need to separate the labels from the pixel values. 

# In[ ]:


x_train = train.iloc[:, 1:].values.astype('float32')
labels = train.iloc[:, 0].values.astype('int32')


# Let's plot the first 5 images from the dataset to better visualize the data.

# In[ ]:


fig = plt.figure(figsize=(12, 12))
for i in range(5):
    fig.add_subplot(1, 5, i+1)
    plt.title('Label: {label}'.format(label=labels[i]))
    plt.imshow(x_train[i].reshape(28, 28), cmap='Greys')


# Since this is a multiclass classification problem, we will **One Hot Encode** the labels.
# This simply means that we will use vectors to represent each class, instead of the label value.
# Each vector contains the value 1 at the index corresponding to the class it represents, with the rest of the values set to 0.

# In[ ]:


CLASSES = 10
y_train = np.zeros([labels.shape[0], CLASSES])
for i in range(labels.shape[0]):
        y_train[i][labels[i]] = 1
y_train.view(type=np.matrix)


# The next step is to split the data into training and testing parts, since we would like to test our accuracy of our model at the end. We will use around 10% of our training data for testing.

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)
print('Train size: {train}, Test size: {test}'.format(train=x_train.shape[0], test=x_test.shape[0]))


# Now, our data is ready for both training and testing our neural network. Next, we will take a look at the implementation of the Extreme Learning Machine.

# # Extreme Learning Machine implementation
# 
# The ELM algorithm is similar to other neural networks with 3 key differences:
# 1. The number of hidden units is usually larger than in other neural networks that are trained using backpropagation.
# 2. The weights from input to hidden layer are randomly generated, usually using values from a continuous uniform distribution.
# 3. The output neurons are linear rather than sigmoidal, this means we can use least square errors regression to solve the output weights.

# Let's start by defining some constants and generate the input to hidden layer weights:

# In[ ]:


INPUT_LENGHT = x_train.shape[1] # 784 
HIDDEN_UNITS = 1000

Win = np.random.normal(size=[INPUT_LENGHT, HIDDEN_UNITS])
print('Input Weight shape: {shape}'.format(shape=Win.shape))


# The next step is to compute our hidden layer to output weights. This is done in the following way:
# * Compute the dot product between the input and  input-to-hidden layer weights, and apply some activation function. Here we will use ReLU, since it is simple and in this case it gives us a good result:

# In[ ]:


def input_to_hidden(x):
    a = np.dot(x, Win)
    a = np.maximum(a, 0, a) # ReLU
    return a


# * Compute output weights, this is a standard least square error regression problem, since we try to minimize the least square error between the predicted labels and the training labels.
# The solution to this is:
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/3c23b9696f0b9c900b68e71fc63b1b5a0e0cc1e1)
# > Where **X** is our input to hidden layer matrix computed using the function from the previous step, and **y** is our training labels.

# In[ ]:


X = input_to_hidden(x_train)
Xt = np.transpose(X)
Wout = np.dot(np.linalg.inv(np.dot(Xt, X)), np.dot(Xt, y_train))
print('Output weights shape: {shape}'.format(shape=Wout.shape))


# Now that we have our trained model,  let's create a function that predicts the output, this is done simply by computing the dot product between the result from the *input_to_hidden* function we defined earlier, with the output weights:

# In[ ]:


def predict(x):
    x = input_to_hidden(x)
    y = np.dot(x, Wout)
    return y


# Next, we can test our model:

# In[ ]:


y = predict(x_test)
correct = 0
total = y.shape[0]
for i in range(total):
    predicted = np.argmax(y[i])
    test = np.argmax(y_test[i])
    correct = correct + (1 if predicted == test else 0)
print('Accuracy: {:f}'.format(correct/total))


# As we can see, our accuracy is far from the state of the art results on the MNIST dataset, but can easily be improved.
# 
# The easiest improvement would be to increase the number of hidden units in our model, as the universal approximation theorem states that: 
# > A feedforward network with a single layer can represent any function, given it has sufficient hidden units.
# 
# This is theoretically true, but in practice it will take a very long time and we will quickly run out of RAM.
# 
# But we can still increase the number of hidden units to 1e+4 instead of 1e+3 and we can observe that our accuracy goes up to 0.97.
# 
# We reached the end of this notebook, hopefully you found this interesting and fun.
# 
# Good luck and continue exploring and learning new and exciting stuff !

# # References
# * https://en.wikipedia.org/wiki/Extreme_learning_machine
# * https://arxiv.org/pdf/1412.8307.pdf
# * https://en.wikipedia.org/wiki/Universal_approximation_theorem
# * https://en.wikipedia.org/wiki/Least_squares
