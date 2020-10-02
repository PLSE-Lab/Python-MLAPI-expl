#!/usr/bin/env python
# coding: utf-8

# # Softmax regression using kernel

# ## Load MNIST data

# In[ ]:


import numpy as np # linear algebra
import matplotlib.pyplot as plt
import os.path
import gc
import pickle, gzip


def get_MNIST_data():
    """
    Reads mnist dataset from file

    Returns:
        train_x - 2D Numpy array (n, d) where each row is an image
        train_y - 1D Numpy array (n, ) where each row is a label
        test_x  - 2D Numpy array (n, d) where each row is an image
        test_y  - 1D Numpy array (n, ) where each row is a label

    """
    f = gzip.open(r"../input/mnist.pkl.gz", 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    train_x = np.vstack((train_x, valid_x))
    train_y = np.append(train_y, valid_y)
    test_x, test_y = test_set
    return (train_x, train_y, test_x, test_y)

train_x, train_y, test_x, test_y = get_MNIST_data()


# ## Utilities functions

# In[ ]:


# PCA utilities
def project_onto_PC(X, pcs, n_components):
    """
    Given principal component vectors pcs = principal_components(X)
    this function returns a new data array in which each sample in X
    has been projected onto the first n_components principcal components.
    """
    # TODO: first center data using the centerData() function.
    # TODO: Return the projection of the centered dataset
    #       on the first n_components principal components.
    #       This should be an array with dimensions: n x n_components.
    # Hint: these principal components = first n_components columns
    #       of the eigenvectors returned by principal_components().
    #       Note that each eigenvector is already be a unit-vector,
    #       so the projection may be done using matrix multiplication.
    return (pcs[:,:n_components].T@center_data(X).T).T

def center_data(X):
    """
    Returns a centered version of the data, where each feature now has mean = 0

    Args:
        X - n x d NumPy array of n data points, each with d features

    Returns:
        n x d NumPy array X' where for each i = 1, ..., n and j = 1, ..., d:
        X'[i][j] = X[i][j] - means[j]
    """
    feature_means = X.mean(axis=0)
    return(X - feature_means)


def principal_components(X):
    """
    Returns the principal component vectors of the data, sorted in decreasing order
    of eigenvalue magnitude. This function first caluclates the covariance matrix
    and then finds its eigenvectors.

    Args:
        X - n x d NumPy array of n data points, each with d features

    Returns:
        d x d NumPy array whose columns are the principal component directions sorted
        in descending order by the amount of variation each direction (these are
        equivalent to the d eigenvectors of the covariance matrix sorted in descending
        order of eigenvalues, so the first column corresponds to the eigenvector with
        the largest eigenvalue
    """
    centered_data = center_data(X)  # first center data
    scatter_matrix = np.dot(centered_data.transpose(), centered_data)
    eigen_values, eigen_vectors = np.linalg.eig(scatter_matrix)
    # Re-order eigenvectors by eigenvalue magnitude:
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    return eigen_vectors


def plot_PC(X, pcs, labels):
    """
    Given the principal component vectors as the columns of matrix pcs,
    this function projects each sample in X onto the first two principal components
    and produces a scatterplot where points are marked with the digit depicted in
    the corresponding image.
    labels = a numpy array containing the digits corresponding to each image in X.
    """
    pc_data = project_onto_PC(X, pcs, n_components=2)
    text_labels = [str(z) for z in labels.tolist()]
    fig, ax = plt.subplots()
    ax.scatter(pc_data[:, 0], pc_data[:, 1], alpha=0, marker=".")
    for i, txt in enumerate(text_labels):
        ax.annotate(txt, (pc_data[i, 0], pc_data[i, 1]))
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    plt.show()


def reconstruct_PC(x_pca, pcs, n_components, X):
    """
    Given the principal component vectors as the columns of matrix pcs,
    this function reconstructs a single image from its principal component
    representation, x_pca.
    X = the original data to which PCA was applied to get pcs.
    """
    feature_means = X - center_data(X)
    feature_means = feature_means[0, :]
    x_reconstructed = np.dot(x_pca, pcs[:, range(n_components)].T) + feature_means
    return x_reconstructed

def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))


# ### Kernel functions

# In[ ]:


def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
#     return (X @ Y.T + c).astype(np.float32) ** p
    return (X @ Y.T + c) ** p

def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    return np.exp(-gamma * ((X ** 2).sum(axis=1)[None].T + (Y ** 2).sum(axis=1) - 2 * X @ Y.T))


# ## Define kernel and hyperparameter

# In[ ]:


kernel = lambda X, Y: polynomial_kernel(X, Y, 1, 3)
# # kernel = lambda X, Y: X@Y.T
alpha = 1
temp_parameter = 10
lambda_factor = 1e-4


# In[ ]:


import time
start = time.time()
n_size = 25000
n_components = 50
pcs = principal_components(train_x)
train_x = project_onto_PC(train_x, pcs, n_components)[:n_size]
test_x = project_onto_PC(test_x, pcs, n_components)
train_y = train_y[:n_size]
print(time.time()-start)
train_x = augment_feature_vector(train_x)
test_x = augment_feature_vector(test_x)
K = kernel(train_x, train_x)  # full
print(time.time()-start)
Ktest = kernel(train_x, test_x)
print(time.time()-start)


# With kernel trick, define $\theta$ as below:
# $\alpha$, kernel coefficient, $ym$:label matrix, (-1,1)
# 
# $\theta=\begin{bmatrix} 
# \theta_0= \sum _{i=1}^{n} \alpha^{(0,i)} ym^{(0,i)} \phi (x^{(i)})\\
# \theta_1= \sum _{i=1}^{n} \alpha^{(1,i)} ym^{(1,i)} \phi (x^{(i)})\\
# ...\\
# \theta_k= \sum _{i=1}^{n} \alpha^{(k,i)} ym^{(k,i)} \phi (x^{(i)})\\
# \end{bmatrix}$
# 
# $\theta = (\alpha \circ ym ) \phi(x)$
# 
# Then convert cost function $J(\theta)$ using above defination.
# 
# $J(\theta ) = -\frac{1}{n}\Bigg[\sum _{i=1}^ n \sum _{j={0}}^{k-1} [[y^{(i)} == j]] \log {\frac{e^{\theta _ j \cdot x^{(i)} / \tau }}{\sum _{l={0} }^{{k-1} } e^{\theta _ l \cdot x^{(i)} / \tau }}}\Bigg] + \frac{\lambda }{2}\sum _{j=0}^{k- }\sum _{i=0}^{d-1} \theta _{ji}^2$
# 
# Gradient become:
# 
# $\nabla _{\alpha_j} J(\alpha ) = -\frac{1}{\tau n} \sum _{i = 1} ^{n} [([[y^{(i)} == j]] - p(y^{(i)} = j | x^{(i)}, \alpha_j ))]K(x^{(j)},x^{(i)})\circ ym^{(j)} + \lambda \alpha_j$

# In[ ]:


use_ym = True
# use_ym = False

def compute_probabilities_kernel(a, temp_parameter, K, YM):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        a - (k, n) NumPy array, where row j represents the parameters of kernel coeffection for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
        K - Kernel matrix, training time: train_x against train_x; test time: train_x against test_x
        YM - (k, n) Label matrix, -1: false, +1: true
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    if use_ym:
        w = (a * YM) @ K / temp_parameter
    else:
        w = a @ K / temp_parameter
    w = np.exp(w - w.max(axis=0))
    return w / w.sum(axis=0)


def compute_log_probabilities_kernel(a, temp_parameter, K, YM):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        a - (k, n) NumPy array, where row j represents the parameters of kernel coeffection for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
        K - Kernel matrix, training time: train_x against train_x; test time: train_x against test_x
        YM - (k, n) Label matrix, -1: false, +1: true
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the log probability that X[i] is labeled as j
    """
    if use_ym:
        w = (a * YM) @ K / temp_parameter
    else:
        w = a @ K / temp_parameter
    w = w - w.max(axis=0)
    return w - np.log(np.exp(w).sum(axis=0))


def compute_cost_function_kernel(A, a, lambda_factor, temp_parameter, K, YM):
    """
    Computes the total cost over every datapoint.

    Args:
        A - (n, k) NumPy array containing the onehot for labels (a number from 0-9) for each
            data point
        a - (n,) NumPy array, where row j represents the parameters of kernel perceptron
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)
        K - Kernel matrix, training time: train_x against train_x; test time: train_x against test_x
        YM - (k, n) Label matrix, -1: false, +1: true
    Returns
        c - the cost value (scalar)
    """
    w = compute_log_probabilities_kernel(a, temp_parameter, K, YM)
    return -(A * w).sum() / A.shape[0]



def run_gradient_descent_iteration_kernel(A, a, alpha, lambda_factor, temp_parameter, K, YM):
    """
    Runs one step of batch gradient descent

    Args:
        A - (n, k) NumPy array containing the onehot for labels (a number from 0-9) for each
            data point
        a - (n,) NumPy array, where row j represents the parameters of kernel perceptron
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)
        K - Kernel matrix, training time: train_x against train_x; test time: train_x against test_x
        YM - (k, n) Label matrix, -1: false, +1: true
    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """

    if use_ym:
        return (1 - lambda_factor * alpha) * a + YM * (
                K @ (A - compute_probabilities_kernel(a, temp_parameter, K, YM)).T).T \
               / (temp_parameter * A.shape[0] / alpha)
    else:
        return (1 - lambda_factor * alpha) * a + (
                K @ (A - compute_probabilities_kernel(a, temp_parameter, K, YM)).T).T \
               / (temp_parameter * A.shape[0] / alpha)



def get_classification_kernel(a, temp_parameter, K, YM):
    """
    Makes predictions by classifying a given dataset

    Args:
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    probabilities = compute_probabilities_kernel(a, temp_parameter, K, YM)
    return np.argmax(probabilities, axis=0)


def compute_test_error_kernel_inside(test_y, a, temp_parameter, K, YM):
    assigned_labels = get_classification_kernel(a, temp_parameter, K, YM)
    return 1 - np.mean(assigned_labels == test_y)

def compute_test_error_kernel(train_y, a, temp_parameter, Ktest, test_y):
    # label matrix with -1 and 1
    YM = -1 * np.ones(a.shape)
    YM[train_y, np.arange(train_y.shape[0])] = 1
    return compute_test_error_kernel_inside(test_y, a, temp_parameter, Ktest, YM)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()


# In[ ]:


def softmax_regression_kernel(K, train_y, Ktest, test_y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        train_y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    a = np.zeros((k, train_y.shape[0]))  # (k,n) k column for each label

    
    cost_function_progression = []
    test_error_progression = []

    # label onehot matrix
    A = np.zeros(a.shape)
    A[train_y, np.arange(train_y.shape[0])] = 1

    # label matrix with -1 and 1
    YM = -1 * np.ones(a.shape)
    YM[train_y, np.arange(train_y.shape[0])] = 1

    for i in range(1, num_iterations + 1):
        loss = compute_cost_function_kernel(A, a, lambda_factor, temp_parameter, K, YM)
        cost_function_progression.append(loss)
        a = run_gradient_descent_iteration_kernel(A, a, alpha, lambda_factor, temp_parameter, K, YM)
        test_error = compute_test_error_kernel_inside(test_y, a, temp_parameter, Ktest, YM)
        test_error_progression.append(test_error)
    return a, cost_function_progression, test_error_progression


# ## Run regression, takes a long time. Due to numpy in this notebook is not using Intel MKL.

# In[ ]:



a, cost_function_history, test_error_hist = softmax_regression_kernel(K, train_y, Ktest, test_y, temp_parameter, alpha=alpha,
                                                     lambda_factor=lambda_factor, k=10, num_iterations=400)


# In[ ]:


test_error = compute_test_error_kernel(train_y, a, temp_parameter, Ktest, test_y)


# In[ ]:


test_error


# In[ ]:


plot_cost_function_over_time(cost_function_history)
plot_cost_function_over_time(test_error_hist )

