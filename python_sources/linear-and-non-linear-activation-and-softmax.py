#!/usr/bin/env python
# coding: utf-8

# # Linear and non-linear activation, and softmax
# 
# ## Introduction
# 
# This notebook is continuation of "[Implementing a linear neural network](https://www.kaggle.com/residentmario/implementing-a-feedforward-neural-network)".
# 
# Every node in a neural network (save the output layer and input layers) both emit and recieve **tensors**. The inputs given to the node by the input tensors must be reshaped so that may be shipped off to the output tensor. The reshaping operation that does this is known as the **activation function**.
# 
# The simplest activation function is `linear`. A linear activation is simply a linear transform on the data. The input signals from the tensors that sink into the node are simply added up, and the result of that addition is packaged and sent downstream through the output tensor. A linear activator allows a node to learn linear relationships within the data. However, linear relationships are limited in power, as they can only learn on linear transformations of the dataset. A neural network consisting of linear layers is comparable to a simple linear regression model in power, as it can only discover and interpret signals in the data which are linearly shaped.  But, they are brutally simple, and trivially easy to train, as their derivative when performing backpropagation is a constant value of 1. In other words:
# 
# $$f(x) = x \implies f'(x) = 1$$
# 
# Technically speaking, there *is* one other activation function that is even simpler than linear activation: **binary activation**. We saw this kind of activation function in "[Implementing a perceptron](https://www.kaggle.com/residentmario/implementing-a-perceptron/)". Binary activation is so simple, however, that it isn't useful for anything except for trivial examples, and Keras doesn't even implement it.
# 
# In this notebook I'm going to look at an example application of both linear activation and a non-linear competitor: softmax activation. The example will be based on an artificial dataset.

# ## Single-class model
# 
# We'll start with a demonstration of basic `keras` in action. For more on Keras, see the ["Keras sequential and functional modes"](https://www.kaggle.com/residentmario/keras-sequential-and-functional-modes) notebook.
# 
# In this section we'll build a simpe single-layer feedforward linear neural network.

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD


# In[ ]:


# Generate dummy data
import numpy as np
y_train = np.random.randint(2, size=(1000, 1))
X_train = y_train + (np.random.normal(size=(1000, 1)) / 5)
y_test = np.random.randint(2, size=(100, 1))
X_test = y_test + np.random.random(size=(100, 1))


# Here is the data we are working with. `y_test` and `y_train` are classes of two clusters of points, one meaned near 1 and the other near 0. `X_test` and `X_train` are the same 0 and 1 values, plus some Gaussian noise for displacement. There is a very small amount of overlap in this artificial dataset.

# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(X_train[:, 0], y_train, marker='x', c=y_train)
plt.axis('off')


# We build a simple `keras` model. Notice the use of `binary_crossentropy` for the loss. Cross-entropy is a recommended loss metric for categorical classification tasks, and the subject of the next notebook.

# In[ ]:


clf = Sequential()
clf.add(Dense(2, activation='linear', input_shape=(1,), name='hidden'))
clf.add(Dense(1, activation='linear', name='out'))
clf.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=['accuracy'])

clf.fit(X_train, y_train, epochs=10, batch_size=128)


# We can use `model_to_dot` to generate and view a `pydot` visualization of our network:

# In[ ]:


from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(clf).create(prog='dot', format='svg'))


# The hidden layer consists of two nodes using linear activation. The output consists of a single node combining output from the previous two nodes using yet another linear activation function. The end predictions looks something like this:

# In[ ]:


y_pred = clf.predict(X_test)
y_pred[:5]


# Notice that the resulting values are not `0` and `1` class labels, but numbers somewhere in between. They can even be negative values!
# 
# Linear activation doesn't map values directly to classes. In fact, very few activation functions ever do. Most activation functions are functions with long tails that are asymptotically bounded by 0 or 1. So high-confidence predictions will be *close* to those values, but they will never be 0 or 1 exactly. Binary activation is an exception to this rule, because it explicitly codifies `0` and `1` as the only legal target values. And linear activation is an exception to this rule, as demonstrated here, because `f(x) = x` is an unbounded function!
# 
# To generate true class labels using linear activation, we can assign a `0` when `y_pred <= 0.5` and a `1` otherwise.

# In[ ]:


y_pred = clf.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)


# Here's a classification report telling us how well we did:

# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# It looks like this model has no false positives for the 0 class, and no false negatives for the 1 class. An interesting result!

# ## A weakness of  linear activation
# 
# Linear activation functions have many weaknesses.
# 
# One is that they are not a great fit for categorical tasks like this one. In the same vein as linear regression models, they do not map very naturally onto the data.
# 
# In this following few cells I build and run a linear model on a slightly more complicated multiclassification problem. In this case `y` consists of three mutually exclusive classes.

# In[ ]:


def sample_threeclass(n, ratio=0.8):
    np.random.seed(42)
    y_0 = np.random.randint(2, size=(n, 1))
    switch = (np.random.random(size=(n, 1)) <= ratio)
    y_1 = ~y_0 & switch
    y_2 = ~y_0 & ~switch
    y = np.concatenate([y_0, y_1, y_2], axis=1)
    
    X = y_0 + (np.random.normal(size=n) / 5)[np.newaxis].T
    return (X, y)


X_train, y_train = sample_threeclass(1000)
X_test, y_test = sample_threeclass(100)


# In[ ]:


clf = Sequential()
clf.add(Dense(3, activation='linear', input_shape=(1,), name='hidden'))
clf.add(Dense(3, activation='linear', name='out'))
clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

clf.fit(X_train, y_train, epochs=10, batch_size=128)


# Here are the results we get:

# In[ ]:


clf.predict(X_test)[:5]


# As you can see, the values that we get are not particularly meaningful, and they range wildly, roughly through $x \in [-1, 1]$.
# 
# Nevertheless, we still get good accuracy on this artificial dataset if we take the class score with the highest value for each observation as the class.

# In[ ]:


y_test_pred = np.zeros(shape=y_test.shape)

for x, y in enumerate(clf.predict(X_test).argmax(axis=1)):
    y_test_pred[x][y] = 1


# In[ ]:


f'Achieved accuracy score of {(y_test_pred == y_test).sum().sum() / (y_test.shape[0] * y_test.shape[1])}'


# ## Introducing softmax
# 
# Softmax is a non-linear activation function, and is arguably the simplest of the set.
# 
# Recall that the linear activation function is:
# 
# $$f(x) = x$$
# 
# Ok, here's the softmax activation function:
# 
# $$y_i = f(x) = \frac{\exp{z_i}}{\sum_{j=0}^n \exp{z_j}}$$
# 
# In this expression, $z_i$ is the current value. The denominator in the expression is the sum across every value passed to a node in the layer. In other words, the softmax function is dependent not just on the contribution of the current value, but on that of every other value passed to this layer in the network. This is nice because it normalizes the values. The values outputted by nodes in a softmax layer will always sum to 1. When we are performing classification, these values are directly interpretable as probabilities!
# 
# The `exp` terms ($e$ to an exponent) give the resulting values an exponential character. This formula has a few interesting characteristics. Maximum values and values very near maximum are distorted to be even larger after the transformation, at the cost of smaller values, which are made even smaller.
# 
# This effect is further enhanced if you start with large values, which will increase the skew even further. In other words, unlike the linear function the softmax function is *not* scale-invariant. It can help (a lot) to normalize the values in the dataset prior to training.
# 
# Let's train a neural network which differs from the previous neural network solely in its use of the softmax function for the output layer.

# In[ ]:


clf = Sequential()
clf.add(Dense(3, activation='linear', input_shape=(1,), name='hidden'))
clf.add(Dense(3, activation='softmax', name='out'))
clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

clf.fit(X_train, y_train, epochs=20, batch_size=128)


# In[ ]:


y_test_pred = clf.predict(X_test)


# In[ ]:


y_test_pred[:5]


# As you can see, the values we get now are very easily interpretable. We can see in the second case for example that the network is highly confident that the given value is a class 0 observation. In the first case it is less confident that that the second.
# 
# Note that the output of a softmax layer will always sum to 1, and can thus be *interpreted* as a probability. However, doing so is highly suspect when the loss metric we are using doesn't use these probabilities. For example, loss metric I used for this example, cross-entropy, only cares about getting things right or wrong, and not the probabilities assigned to those values by the model. Thus the model has no incentive to gauge realistic probabilities; it only needs to get close enough approximations to pick a winner. We can reliably order these values: e.g. look at `[0.29880288, 0.41953945, 0.2816577 ]` and say "Class 2 is more likely than Class 1, which is more likely than Class 3". But we would get into hot water if we tried to say "Class 2 and Class 3 are almost equally likely".
# 
# The Naive Bayes algorithm is a non-neural machine learning technique has a similar asterisk in this regard. See my notebook on "[Probability calibration](https://www.kaggle.com/residentmario/notes-on-classification-probability-calibration/)" for some thoughts on techniques you can use to transform an ordered result into a probability that you can rely on.

# In[ ]:


y_test_pred.argmax(axis=1)


# In[ ]:


y_test_pred = np.zeros(shape=y_test.shape)

for x, y in enumerate(clf.predict(X_test).argmax(axis=1)):
    y_test_pred[x][y] = 1


# In[ ]:


f'Achieved accuracy score of {(y_test_pred == y_test).sum().sum() / (y_test.shape[0] * y_test.shape[1])}'


# ## Conclusion
# 
# To review:
# 
# **Linear activation**
# * Simple and easy to reason about.
# * Scale-invariant.
# * Generates results that are not very fitted, w.r.t classification tasks.
# * Can only discover linear relationships in the data.
# 
# **Non-linear activation**
# * Can discover non-linear relationships in the data.
# * Usually not scale-invariant.
# * More complex and harder to reason about.
# 
# **Sofmax activation** (a type of non-linear activation)
# * Not scale-invariant.
# * Provides scaled results which may be interpretable as probabilities, making it very appropriate for classification tasks.
