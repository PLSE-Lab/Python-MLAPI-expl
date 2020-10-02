"""Data Utility Functions."""
# pylint: disable=invalid-name
import os
import pickle as pickle

import numpy as np


def load_cifar_batch(filename):
    """Load single batch of CIFAR-10."""
    X_raw =[]
    Y_raw = []
    for i in range(1):
        with open(filename + str(i+1)+'.p', 'rb') as f:
        # load with encoding because file was pickled with Python 2
            data_dict = pickle.load(f, encoding='latin1')
            X_raw.extend(data_dict['data'])
            Y_raw.extend(data_dict['labels'])
    X = np.array(X_raw)
    Y = np.array(Y_raw)
    X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    return X, Y


def load_CIFAR10(root_dir):
    """Load all of CIFAR-10."""
    f = os.path.join(root_dir, 'cifar10_train')
    X_batch, y_batch = load_cifar_batch(f)
    return X_batch, y_batch


def scoring_function(x, lin_exp_boundary, doubling_rate):
    """Computes score function values.

        The scoring functions starts linear and evolves into an exponential
        increase.
    """
    assert np.all([x >= 0, x <= 1])
    score = np.zeros(x.shape)
    lin_exp_boundary = lin_exp_boundary
    linear_region = np.logical_and(x > 0.1, x < lin_exp_boundary)
    exp_region = np.logical_and(x >= lin_exp_boundary, x <= 1)
    score[linear_region] = 100.0 * x[linear_region]
    c = doubling_rate
    a = 100.0 * lin_exp_boundary / np.exp(lin_exp_boundary * np.log(2) / c)
    b = np.log(2.0) / c
    score[exp_region] = a * np.exp(b * x[exp_region])
    return score

def get_CIFAR10_data_full(num_training=48000, num_validation=1000, num_test=1000, num_dev=500):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier. 
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = '../input/cifar10/'
    X, y = load_CIFAR10_full(cifar10_dir)
    
    # Our training set will be the first num_train points from the original
    # training set.
    mask = range(num_training)
    X_train = X[mask]
    y_train = y[mask]
    
    
    # Our validation set will be num_validation points from the original
    # training set.
    mask = range(num_training, num_training + num_validation)
    X_val = X[mask]
    y_val = y[mask]
    
    # We use a small subset of the training set as our test set.
    mask = range(num_training + num_validation, num_training + num_validation + num_test)
    X_test = X[mask]
    y_test = y[mask]
    
    # We will also make a development set, which is a small subset of
    # the training set. This way the development cycle is faster.
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image
    
    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    return X, y, X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev


def load_CIFAR10_full(root_dir):
    """Load all of CIFAR-10."""
    filename = os.path.join(root_dir, 'cifar10_train.p')
    with open(filename, 'rb') as f:
    # load with encoding because file was pickled with Python 2
        data_dict = pickle.load(f, encoding='latin1')
        X = np.array(data_dict['data'])
        y = np.array(data_dict['labels'])
        X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        return X, y
