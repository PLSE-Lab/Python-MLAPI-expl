#!/usr/bin/env python
# coding: utf-8

# # A Perceptron Learning Algorithm from Scrach

# # Why we should know the math behind?
# Yes, we can use the scikit-learn api with out knowing an algorithm. But in that way we won't have good knowledge on what the problem should be if the model is not good or how we can tune the model etc. I am more interested in knowing what's happening behind when we are using an algorithm. 
# # Why I wrote that kernal?
# Every time I start reading a book on ML it goes throung all the math and implement directly with a practical example. But not in detail though it make sence as the books are more focus on general overview. So I decide to break down the examples in to small part so that I or anyone can review when we forget how the whole model works .Before we create our own API let's have a look on the steps it's need to go through to fit and predict from a model.
# # What will be covered?
# The percentron learning Algorithm will be explained in 3 phases through this kernel.
# * First we will start with a small sample (4 sample and 2 feature) and explain a single iteration of the algorithm
# * Second we run the same algorithm in a loop for multiple iteration
# * Third we write the second step in a object oriented way to use it like scikit-learn API

# In[ ]:


# necessary import
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# import os
# print(os.listdir("../input"))


# # A Single Sample Case Study:
# Let's generate a small sample with 2 features as $X$ and y as prediction. So we will use the relation between $X$ and $y$ and learn about the relation. Then we will try to predict from what we have learn by building an API.
# 

# # Sample Data
# Lets create a 2d numpy array with the dimention of 4 by 2 for input and 1d array for output/prediciton.

# In[ ]:


# create a data set
np.random.seed(42)
X = np.random.random((4, 2))
X = np.round_(X, 2)
y = np.array([1, -1, 1, 1])
print(X)
print(y)


# Here each row is a sample $X = [x^1, x^2, x^4 , x^3]$ . Each sample has 2 feature. For example sample $x^1 = \begin{bmatrix}x^1_0\\x^1_1\end{bmatrix} = \begin{bmatrix}0.73\\0.60\end{bmatrix}$ .  Then we have value to predict $y^t = [y^1, y^2, y^4 , y^3] = [1, 0, 1, 1]$

# # Phase 1: Working with a Single Sample
# In that phase we will select a single sample (single row) from the metrics ($x^t = [0.37, 0.95]$) and do a single iteration.

# **Generating Weight:**
# 
# To initiate the learning it's is a classical approach to start with a random weight. So we will have a weight vector $ w = [w_0, w_1, w_2]$ including the intecept/error term as $w_0$.

# In[ ]:


# generate initial value of estimator 
rgen = np.random.RandomState(42)
w = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
print(w)


# **Tne net input calculation:** 
# 
# For a single sample the multiplication is as follows: Here $x_0 = 1$
# 
# $w^{t}x = \begin{bmatrix}w_0&w_1 & w_2\end{bmatrix} \begin{bmatrix}x_0\\x_1 \\ x_2\end{bmatrix} = \begin{bmatrix}w_0 +w_1x_1 + w_2x_2\end{bmatrix} $ 
# 
# In case of `numpy` **w** do not need to be Transposed as it is a array and `numpy.dot()` will do the the vector dot product for us. In the below example we can see how it works practically in numpy. I just took first vector from the $X$ array and multiply with $w$ to get the net input for a single sample. As I do not have $x_0w_0$ in my main metrix, so it's added later with $XW$ to get $XW_{intercept}$.

# In[ ]:


# evaluate the net input 
xw = np.dot(X[0], w[1:])
xw_intercept = xw + w[0]
print(xw)
print(xw_intercept)


# **Pridicition from a net Input:**
# 
# If we consider a function as:
# 
# $\begin{aligned}
# f(z=w^{t}_{intercept}x) = 1, if  z >= 0.0 \\
# = 0, otherwise 
# \end{aligned}$
# 
# Then we can develope a `np.where()` method to calculate the predicted output $\hat{y}$

# In[ ]:


# Pridiction from a single iteration
y_hat = np.where(xw_intercept >= 0.0, 1, -1)
print(y_hat)


# **Update the Parameters**:
# 
# Each time parameter is updated as follows: $w_j = w_j + \Delta w_j$
# where $\Delta w_j$ can be expressed as: $\Delta w_j = \eta (y^i - y^i_{hat}) x^i_j$, where $\eta$ is called the learning rate one of a hyperparameter of the model. In the below section we will do the same operation for the sample $x^0$ and $y^0$. First we will update the weight for the parameter of the variables ($x_1$ and $x_2$) then we will udate the parameter for intercept/error ($x_0$). 

# In[ ]:


eta = 0.01
target = y[0]
update = eta * (y[0] -y_hat)
print(update)
w[1:] += update * X[0]
print('With out Intercept:', w)
w[0] += update
print('With Intercept:', w)


# **Classification Error Calculation:**
# 
# We can also record after each weight updated how many of the output classified correctly and how many of them classified incorrectly. As so far we just have worked with single sample and one iteration so there will be just one value. In the next section we are going to run the algorithm in a batch over all samples which will make more sence.

# In[ ]:


errors_ = [] 
error =0
error += int(update != 0.0) # give out put 1 or 0
print(error)
errors_.append(error)


# # Phase 2: Run the Algorithom in a Batch

# This time we will take the same sample and run the algorithm in a loop to iterate thorugh all sample for 10 times. In that way our weight will be adjusted everytime and will try to find the best fit based on the data.

# In[ ]:


# sample data
np.random.seed(42)
X = np.random.random((4,2))
X = np.round_(X, 2)
y = np.array([1, -1, 1, 1])
print(X)
print(y)


# In[ ]:


# enitiate the value of estimator
rgen = np.random.RandomState(42)
w = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
print(w)


# In[ ]:


# function to calculate the net input w_transopseX
def net_input(x):
    return np.dot(x, w[1:]) + w[0]


# In[ ]:


# create the threshlold function for prediction
def prediction(x):
    return np.where(net_input(x) >= 0.0, 1, -1)


# In[ ]:


# lets look over one single loop
result_list = []
for x_i, y_i in zip(X, y):
    dict_ = {'x': x_i,
             'y': y_i,
             'net_input': round(net_input(x_i), 3),
             'y_hat': prediction(x_i)
            }
    result_list.append(dict_)


# In[ ]:


result_list


# The following loop goes through each sample calculate the small update for the weights and also store the miss-classificaton.

# In[ ]:


eta = 0.01
error_list = []
errors = 0
for x_i, y_i in zip(X, y):
    update = eta * (y_i - prediction(x_i))
    w[1:] += update * x_i
    w[0] += update
    errors += int(update != 0.0)
error_list.append(errors)


# In[ ]:


error_list


# In[ ]:


# iterate the same process multiple time
eta = 0.01
error_list = []
for _ in range(10):
    errors = 0
    for x_i, y_i in zip(X, y):
        update = eta * (y_i - prediction(x_i))
        w[1:] += update * x_i
        w[0] += update
        errors += int(update != 0.0)
    error_list.append(errors)


# In[ ]:


error_list


# # Phase 3: Create the API
# Finally we can put them all together to create a package like structure using the previous knowledge. I will not go to the detail that how exactly to create a package. But I will more focus on creating a perceptron learning algorithm class, which can be used exactly as scikit-learn.  The code I am using here is taken from Python Mechine Learning (S Raschka, V Mirjalili, 2017, packt) with minor modification. Actually the previous part was the break down of that API. If some one is not familier with how to create a package/API in object oriented approach can read the following blog https://datapsycho.github.io/PsychoBlog/dataparrot-18-01

# In[ ]:


import numpy as np

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of 
          samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, 
                              size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for x_i, y_i in zip(X, y):
                update = self.eta * (y_i - self.predict(x_i))
                self.w_[1:] += update * x_i
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# # A Object Oriented Class to generate Logistic Like Data
# Here I have written a class which can generate logistic like data with 2 feature. We are able to specify the sample and random state to generate such data.

# In[ ]:


class LogisticSimulator(object):
    """Logistic data generator with 2 feature.

    Parameters
    ------------
    size : sample size (int)
      Any value between 0 and Infinity.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    X_ : nd-array
      Feature metrics
    y : 1d-arry
      Outcome variable """

    def __init__(self, size=4000, random_state=13):
        self.size = size
        self.random_state = random_state

    def sample_generator(self):
        np.random.seed(self.random_state)
        x1_ = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], self.size)
        x2_ = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], self.size)
        X_ = np.vstack((x1_, x2_)).astype(np.float32)
        y_ = np.hstack((np.ones(self.size)*-1, np.ones(self.size)))
        return X_, y_
    
    def viz_generator(self, X, y):
        plt.figure(figsize=(12,8))
        plt.scatter(X[:, 0], X[:, 1], c = y, alpha = .3)


# In[ ]:


# call the class and generate feature metrix and outcome variable
lgen = LogisticSimulator(size=1000, random_state=13)
X, y = lgen.sample_generator()
lgen.viz_generator(X, y)


# In[ ]:


ppn = Perceptron(eta=0.01, n_iter=500)
ppn.fit(X, y)


# Finally we can plot the classificaiton errors to see how the model is working. We can see the model is not working well. But that was not the main point of that karnel. The main focus was to look insede of perceptron learning Algorithm.

# In[ ]:


plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()


# # Conclusion
# We started from scratch and were able to look inside of perceptron learning algorithm. We were able build an API based on phase 1 and phase 2. Then we create another class to geherate logistic like data and fit our algorithm with the data. There is lot of things we can imporve including model it self. We can implement gradient descent to the algoritm and many other decscend funciton as fruther improvement.
