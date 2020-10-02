#!/usr/bin/env python
# coding: utf-8

# # Softmax Regression from Scratch Using Gradient Descent: A Vectorized Approach With Numpy

# In this notebook we will take another look at the MNIST data set with the goal of constructing a softmax regressor using numpy and without the aid of Scikit Learn. Scikit Learn offers some incredibly useful out of the box tools for machine learning but it can be instructive to check our own understanding of what's going on under the hood by building a model from scratch. We will also demonstrate a possible way one might fail to utilize the power of numpy when implementing algorithms and how to go about fixing it (this is a real life example of a mistake I made myself). This notebook is inspired by the work I did for a homework project in 'Machine Learning with Python: From Linear Models to Deep Learning' produced and taught by MIT for the EDX course platform. First we perform some imports and cheat slightly by using scikit learn only for importing the data set.

# In[ ]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version = 1)


# For those new to MNIST and/or those who didn't view my previous notebooks on the subject, MNIST is a collection of 70,000 hand drawn images of digits 0 through 9 with 784 pixels (28x28) and an associated label of the corresponding digit. They are represented by feature vectors with 784 features each ranging in value from 0 to 255 corresponding to the darkness of the pixel.

# To begin let's shuffle up the data. Machine learning algorithms typically perform better when the data is appropriately scaled so we will divide the pixel values by 255 to ensure that they lie between 0 and 1. Lastly, let's split the data into training and test sets and take a look at a single image and its label to get an idea of what we're dealing with.

# In[ ]:


shuffle_index = np.random.permutation(70000)
X, Y = mnist["data"][shuffle_index]/255, mnist["target"][shuffle_index]

x_train, x_test, y_train, y_test = X[:60000], X[60000:], Y[:60000].astype(int), Y[60000:].astype(int)

digit = x_train[2500]
image = digit.reshape(28, 28)
plt.imshow(image)
plt.axis("off")


# In[ ]:


y_train[2500]


# If we'd like our model to include bias we can accomodate offset parameters by simply appending the value $1$ to each feature vector. Here's a convenient function to do that:

# In[ ]:


def augment_feature_vector(X):
    return np.hstack((np.ones([len(X), 1]), X))


# # Assigning Probabilities to Labels

# Softmax regression is fundamentally a classification algorithm despite its name. It works by using exponentiation and a normalizing constant to convert the outputs of the model to a probability distribution where each value is between 0 and 1 and all values sum to 1 and then selects the highest probability as the predicted label. More specifically, for a classifier with labels $j = 0$, $1$, ...$k-1$, we assign to each feature vector $x$ a probability vector $h(x)$ as follows:

# ## $$h(x) = \frac{1}{\sum_{j=0}^{k-1}e^{\theta_j \cdot x-c}}\begin{bmatrix} e^{\theta_0 \cdot x-c} \\ e^{\theta_1 \cdot x-c} \\ \vdots \\ e^{\theta_{k-1} \cdot x-c} \end{bmatrix}$$

# where we've included the constant $c = \max_j\{ \theta_j \cdot \frac{x}{\tau} \}$ to prevent overflow errors in numpy. Note that for digit recognition we will take $k = 10$ to represent digits $j = 0, 1,$ ... $9$.

# # A Naive Numpy Implementation and a Lesson Learned

# Implementing softmax regression will require us to compute $h(x)$ for a given feature vector $x$. We begin by writing a function using numpy, which leads us to our discussion of a potentially costly mistake one must avoid. Let's represent each feacture vector in the training set as the rows of an $(nxd)$ feature matrix $X$ where $n$ represents the number of data points in the training set and $d$ represents the dimension of our feature space. Specifically for MNIST, $n=60000$ and $d=784$. Also notice that for the dot products in $h(x)$ to make sense it must be the case that $\theta_j$ for $j \in 1$, $2$, ...$k-1$ is a $d$ dimensional vector. So let's call the associated $(kxd)$ parameter matrix $\theta$. The following function takes $X$ and $\theta$ as arguments and returns the matrix $H$ where the $i$th column represents $h(x^{(i)})$ as defined above. More intuitively $H$ is a matrix whose columns are probability vectors for each data point as assigned by the current parameter matrix $\theta$. 

# In[ ]:


def naive_compute_probabilities(X, theta):
    #Arguments:
    #X is a (nxd) matrix (numpy array)
    #theta is a kxd matrix
    
    #Returns:
    #H is a (kxn) matrix (numpy array) such that each column of H represents the probabilities that the ith
    #data point takes on each label.
    
    #constructs the skeleton of the output matrix.
    k = theta.shape[0]
    n = X.shape[0]
    H = np.zeros([k,n])

     #loops over all the data points
    for i in range(n):
        dpoint = X[i]
        #constructing a vector that we can take a maximum of to obtain constant c
        c_vector = np.zeros(k)
        for j in range(k):
            c_vector[j] = (np.dot(theta[j], dpoint))
            c = np.amax(c_vector)
        summation = 0
        
        #for each data point we loop through all of the labels
        for j in range(k): 
            exponent = np.dot(theta[j], dpoint) - c
            summation += np.exp(exponent)
            H[j][i] = np.exp(exponent)

        H[:,i] = H[:,i]/summation

    return H


# The above 'naive' implementation of the function compute_probabilities was my first attempt at the task when originally completing this project. As far as I was concerned I was using numpy effectively since my solution involved numpy arrays and it was producing the correct output. But when it came time to train my model the process took upwards of 30 minutes to an hour when other students in the MITx forums were reporting sub two minute training times on inferior hardware. Let's investigate the time it takes for this function to run by computing probabilities for the training data one time for two different choices of $\theta$:

# In[ ]:


import time
 #theta initialized to zero
theta = np.zeros([10, x_train.shape[1]])

start = time.time()
probs = naive_compute_probabilities(x_train, theta)
end = time.time()
end - start


# In[ ]:


#theta randomly initialized
theta = np.random.randint(0, 10, [10,x_train.shape[1]])

start = time.time()
probs = naive_compute_probabilities(x_train, theta)
end = time.time()
end - start


# The output above is in seconds. Not too bad but we're going to run into some trouble when we call this function in every iteration of an algorithm with possibly hundreds of steps! The tradeoff for having the ease of use and accessibility of a language like python is that we must sacrifice having a programming language that operates closely with the hardware. Python is a very 'high-level' language which means there's many layers of translation that must occur before the hardware knows what to do. This makes high level object oriented programming possible, but it's unfortunately quite slow. The workaround here is that much of numpy is fundamentally written in C, a much lower level and faster language than python. Taking advantage of it entails implementing a fully 'vectorized' approach which means performing as many computations as possible using the array operations of numpy rather than loops in python. There are still loops involved but they are taken care of by numpy at the C level rather than the high level of python. A very simple example is if we would like to perform the dot product between two vectors. Given two vectors in the form of numpy arrays we could naively compute the dot product using a python list comprehension like so:

# In[ ]:


a = np.array([1,2,3,4,5])
b = np.array([8,8,8,8,8])

a_dot_b = sum([a[i]*b[i] for i in range(len(a))])
a_dot_b


# Alternatively, a fully vectorized approach using numpy might make use of numpy's np.dot function:

# In[ ]:


np.dot(a,b)


# Not only is it more concise but it will be faster in general. The vectors $a$ and $b$ are probably too small for these operations to make much of a difference for this case but machine learning often involves training on very large data sets (in this case 60,000 images which is still tiny compared to many massive data sets!). As we will see shortly, such details can make the difference between seconds and hours or more!

# # Implementing a Vectorized Approach

# It's often discussed how being skilled at machine learning requires a strong grasp of matrix operations and linear algebra, and in this section we will begin to see how that is the case. In order to implement a vectorized approach we need to be able to express the operations in naive_compute_probabilities entirely using the array operations of numpy. To this end lets try to express the function $h(x)$ in matrix format using the matrices $X$, $\theta$ and $H$ as defined above. The rows of $X$ are the feature vectors and the columns of $H$ are the vectors $h(x^{(i)})$. The first step is recognizing that we can obtain the columns of $H$ by elementwise exponentiation of the matrix $\theta X^T$. Here we're using matrix multiplication which is efficiently computed using numpy's np.matmul operation. The 'c vector' whose entries are the constants $c_i$ for each column of $H$ and the 'summation vector' whose entries are the normalization term for each column of $H$ can be obtained by taking a columnwise maximum and sum of the $\theta X^T$ and its exponentiated form respectively. Finally, the broadcasting rules of numpy make the rest straightforward:

# In[ ]:


def vectorized_compute_probabilities(X, theta):
    #Arguments:
    #X is a (nxd) matrix (numpy array)
    #theta is a kxd matrix
    
    #Returns:
    #H - a (kxn) matrix (numpy array) such that each column of H represents the probabilities that the ith
    #data point takes on each label.
    
    theta_XT = np.matmul(theta, np.transpose(X))
    #taking a columnwise max:
    c = np.amax(theta_XT, axis = 0)
    #elementwise exponentiation of theta_XT:
    exp_matrix = np.exp(theta_XT - c)
    #computing the normalization factors for each column of H:
    sum_vector = np.sum(exp_matrix, axis = 0)
    
    #broadcasting!
    return exp_matrix/sum_vector


# Let's check that this implementation is actually faster than the non vectorized approach:

# In[ ]:


theta = np.zeros([10, x_train.shape[1]])

start = time.time()
probs = vectorized_compute_probabilities(x_train, theta)
end = time.time()
end - start


# In[ ]:


theta = np.random.randint(0, 10, [10, x_train.shape[1]])

start = time.time()
probs = vectorized_compute_probabilities(x_train, theta)
end = time.time()
end - start


# The vectorized implementation out performs the naive approach by a factor of 100! Since this function will be called once for each iteration of the algorithm we can expect a huge time save! My own original model with 150 iterations of gradient descent was shortened from a half hour to 45 seconds by this optimization making more training over more iterations (and thus a better model!) possible.

# # The Cost Function

# Next we present the cost function which we will write code to minimize:

# ## $$ J(\theta) = -\frac{1}{n}\left( \sum_{i = 1}^{n} \sum_{j = 0}^{k - 1} [[y^{(i)}==j]] \log \frac{e^{\theta_j \cdot x^{(i)}-c}}{\sum_{l=0}^{k-1}e^{\theta_l \cdot x^{(i)}-c}} \right) + \frac{\lambda}{2}\sum_{j=0}^{k-1} \sum_{i=0}^{d-1} \theta_{ji}^2 $$

# Here $n$ represents the total number of data points in the training set, $k$ again represents the number of possible labels, and $d$ represents the dimension of the feature space ($n = 60000$, $k = 10$ and $d = 784$ in our case). Here '$==$' denotes the comparison operator which returns True if $y^{(i)} = j$ and False otherwise. The function $[[ \cdot ]]$ just converts a True value to 1 and a False value to 0 (for the mathematicians out there, the composition of these two functions can be thought of as a characteristic function). Here $\lambda$ represents the regularization parameter which will help to control overfitting of the data. We will initialize $\theta$ to be zero when beginning gradient descent so that a large regularization parameter will force smaller changes in $\theta$. Notice that the argument of the logarithm in $J(\theta)$ is just the assigned probability that the $i$th image is of digit $j$ which we will denote as $p \{y^{(i)}=j|\theta \}$ henceforth.

# # Using the Gradient to Update Theta

# Gradient descent works by nudging the $\theta$ parameters in '$\theta$-space' in the direction of maximum decrease of $J(\theta)$ looking for a minimum. So we will need a vector that actually points in said direction. You may remember from vector calculus that the gradient $\nabla_\theta J(\theta)$ is a vector that points in the direction of maximum increase of $J(\theta)$. To obtain a vector in the direction of maximum decrease we can simply negate $\nabla_\theta J(\theta)$ and then multiply by the 'learning rate' $\alpha$ to control the magnitude of the step size in $\theta$-space. The form of the update of each step of gradient descent is then:

# ## $$ \theta \leftarrow \theta - \alpha \nabla_\theta J(\theta) $$

# A simple notion of descending into a valley offers an intuitive picture of the importance of selecting an appropriate $\alpha$. If the learning rate is too large we may bounce back and forth on either side of a minimum without getting sufficiently close, but if it's too small we may have to wait an enormous amount of time before convergence near a minimum occurs. Also note that the valley example uses 2 (perhaps 3) dimensions but in general our parameter space may be of much higher dimension.

# # Computing the Gradient

# Since $\theta$ is a matrix, $\nabla_\theta J(\theta)$ is also a matrix. We'll hand wave away the many steps of partial differentiation and simply state that if we represent the $l$th entry of the feature vector $x^{(i)}$ as $x_l^{(i)}$ then we can express one entry of the matrix $\nabla_\theta J(\theta)$ in the following way:

# ## $$\nabla_{\theta_{jl}} J(\theta) = -\frac{1}{n} \sum_{i = 1}^{n} \left( x_l^{(i)} ([[y^{(i)}==j]] -p\{y^{(i)}=j|\theta\}) \right) + \lambda \theta_{jl}$$

# If we use $M$ to denote the $(kxn)$ sparse matrix of zeroes and ones whose entries are defined by $M_{ij} = [[y^{(i)} == j]]$ then we can obtain the following wonderfully concise matrix representation of the update rule for $\theta$:

# ## $$ \theta \leftarrow \theta - \alpha \nabla_\theta J(\theta) = \theta + \frac{\alpha}{n}(M-H)X - \alpha \lambda \theta $$

# thus facilitating a slick vectorized implementation of a step in the gradient descent algorithm:

# In[ ]:


import scipy.sparse as sparse

def gradient_descent_iteration(X, Y, theta, alpha, lambda_factor):
    
    n = len(Y)
    k = theta.shape[0]
    data = [1]*n
    
    H = vectorized_compute_probabilities(X, theta)
    #more efficient way to implement large sparse arrays:
    M = sparse.coo_matrix((data, (Y, range(n))), shape=(k,n)).toarray()
    
    first_term = np.matmul(M-H, X)*(-1/n)
    second_term = lambda_factor * theta

    return theta - alpha * (first_term + second_term)


# In[ ]:


def predict(X, theta):

    X = augment_feature_vector(X)
    probabilities = vectorized_compute_probabilities(X, theta)
    return np.argmax(probabilities, axis = 0)


# In[ ]:


def compute_accuracy(X, Y, theta):
    predictions = predict(X, theta)
    return np.mean(predictions == Y)


# Finally we write a function that initializes $\theta$ and performs as many steps in gradient descent as we ask it to:

# In[ ]:


def softmax_regression(X, Y, alpha, lambda_factor, k, num_iterations):   
    
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    for i in range(num_iterations):
        theta = gradient_descent_iteration(X, Y, theta, alpha, lambda_factor)
    
    return theta


# In[ ]:


#using parameters suggested in prompt for original project but extending number of iterations of gradient
#descent to 1000 instead of 150

theta_final = softmax_regression(x_train, y_train, alpha = .3, 
                           lambda_factor = 1.0e-4, k = 10, num_iterations = 1000)


# In[ ]:


#accuracy on test set

compute_accuracy(x_test, y_test, theta_final)


# Our coded-from-scratch softmax regression model has predicted MNIST digits with 91 percent accuracy on the test set. It won't be winning any Kaggle competitions, but this is acceptable since we've employed no techniques such as hyperparameter tuning, data set augmentation, cross-validation, or experimenting with other models. I hope this notebook has helped to improve your confidence in building models using packages such as scikit-learn by helping you to understand what's going on under the hood, but we're not quite done with digit recognition yet! In posts to come, we will visit one of the most powerful models for image recognition available. Convolutional Neural Networks!
