#!/usr/bin/env python
# coding: utf-8

# ## Understanding The Metric: Quadratic Weighted Kappa (QWK)

# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/8089/logos/header.png?t=2018-01-10-17-54-22)

# In this kernel we will take a deep dive into the metric for the Data Science Bowl 2019: Quadratic Weighted Kappa (QWK). This is a popular metric in Kaggle competitions and is especially useful for classification tasks where the classes are hierarchical. For these kind of classification tasks a simple accuracy score does not make much sense. 
# 
# 
# P.S. Feel free to check out ["Episode 1" of Understanding The Metric on Root Mean Squared Logarithmic Error (RSMLE)](https://www.kaggle.com/carlolepelaars/understanding-the-metric-rmsle)
# 

# ## Table of Contents

# - [Dependencies](#1)
# - [Preparation](#2)
# - [The Metric](#3)
# - [Best Baselines](#4)
# - [Optimizing QWK](#5)
# - [Submission](#6)

# ## Dependencies <a id="1"></a>

# In[ ]:


# Standard Dependencies
import os
import scipy as sp
import numpy as np
import random as rn
import pandas as pd
from numba import jit
from functools import partial

# The metric in question
from sklearn.metrics import cohen_kappa_score

# Machine learning
import tensorflow as tf
import keras.backend as K
from keras.callbacks import Callback

# Set seed for reproducability
seed = 1234
rn.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# Specify paths
PATH = "../input/data-science-bowl-2019/"
TRAIN_PATH = PATH + "train_labels.csv"
SUB_PATH = PATH + "sample_submission.csv"


# In[ ]:


# File sizes and specifications
print('\n# Files and file sizes')
for file in os.listdir(PATH):
    print('{}| {} MB'.format(file.ljust(30), 
                             str(round(os.path.getsize(PATH + file) / 1000000, 2))))


# ## Preparation <a id="2"></a>

# In[ ]:


# Load in data
df = pd.read_csv(TRAIN_PATH)


# In[ ]:


df.head(3)


# ## The Metric <a id="3"></a>

# Kaggle's explanation of Quadratic Weighted Kappa on the [Data Science Bowl 2019 Evaluation page](https://www.kaggle.com/c/data-science-bowl-2019/overview/evaluation):
# 
# The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix $O$ is constructed, such that $O_{i,j}$ corresponds to the number of installation_ids $i$ (actual) that received a predicted value $j$. An N-by-N matrix of weights, $w$, is calculated based on the difference between actual and predicted values:
# 
# $w_{i,j} = \frac{\left(i-j\right)^2}{\left(N-1\right)^2}$
# 
# An N-by-N histogram matrix of expected outcomes, $E$, is calculated assuming that there is no correlation between values.  This is calculated as the outer product between the actual histogram vector of outcomes and the predicted histogram vector, normalized such that $E$ and $O$ have the same sum.
# 
# From these three matrices, the quadratic weighted kappa is calculated as: 
# 
# $\kappa=1-\frac{\sum_{i,j}w_{i,j}O_{i,j}}{\sum_{i,j}w_{i,j}E_{i,j}}.$
# 
# ---------------------------------------------------
# 
# Note that Quadratic Weighted Kappa score is a ratio that can take a value between -1 and 1. A negative QWK score implies that the model is "worse than random". A random model should give a score of close to 0. Lastly, perfect predictions will yield a score of 1.
# 

# Instead of implementing Quadratic Weighted Kappa from scratch we can also get the metric (almost) out-of-the-box from scikit-learn. The only thing we need to specify is that the weights are quadratic.

# In[ ]:


def sklearn_qwk(y_true, y_pred) -> np.float64:
    """
    Function for measuring Quadratic Weighted Kappa with scikit-learn
    
    :param y_true: The ground truth labels
    :param y_pred: The predicted labels
    
    :return The Quadratic Weighted Kappa Score (QWK)
    """
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


# However, Scikit-learn's implementation can be relatively slow. Luckily, [Kaggle Grandmaster CPMP](https://www.kaggle.com/cpmpml) implemented a really fast method to calculate Quadratic Weighted Kappa using the open-source compiler [Numba](http://numba.pydata.org/).
# 
# [Source](https://www.kaggle.com/cpmpml/ultra-fast-qwk-calc-method)
# 
# [Discussion Topic](https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-657027)

# In[ ]:


@jit
def cpmp_qwk(a1, a2, max_rat=3) -> float:
    """
    A ultra fast implementation of Quadratic Weighted Kappa (QWK)
    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133
    
    :param a1: The ground truth labels
    :param a2: The predicted labels
    :param max_rat: The maximum target value
    
    return: A floating point number with the QWK score
    """
    assert(len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


# ## Best Baselines <a id="4"></a>

# One of the most naive predictions that we can make on this dataset to predict the class that occurs the most. On this dataset that will be 3. The Quadratic Weighted Kappa score will be 0 and therefore no better than random. QWK has a robustness that we also see with a metric such as [The Area under the ROC Curve (AUC)](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5).

# In[ ]:


# Get the ground truth labels
true_labels = df['accuracy_group']


# In[ ]:


# Check which labels are present
print("Label Distribution:")
df['accuracy_group'].value_counts()


# In[ ]:


# Calculate scores for very naive baselines
dumb_score = sklearn_qwk(true_labels, np.full(len(true_labels), 3))
random_score = round(sklearn_qwk(true_labels, np.random.randint(0, 4, size=len(true_labels))), 5)
print(f"Simply predicting the most common class will yield a QWK score of:\n{dumb_score}\n")
print(f"Random predictions will yield a QWK score of:\n{random_score}")


# When we take a closer look at the data we readily notice that there are five different assessments for which we have to predict the accuracy group. To make a good naive prediction we can groupby this assignment and take the mode for each assignment as our prediction. It seems like taking the mean and rounding out will yield good naive predictions. However, this will not yield as good a score as taking the mode.

# In[ ]:


print("Assessment types in the training data:")
list(set(df['title']))


# In[ ]:


# Group by assessments and take the mode
mode_mapping = df.groupby('title')['accuracy_group'].agg(lambda x:x.value_counts().index[0])
mode_preds = df['title'].map(mode_mapping)

# Group by assessments and take the rounded mean
mean_mapping = df.groupby('title')['accuracy_group'].mean().round()
mean_preds = df['title'].map(mean_mapping)


# In[ ]:


# Check which a score a less naive baseline would give
grouped_mode_score = round(sklearn_qwk(true_labels, mode_preds), 5)
grouped_mean_score = round(sklearn_qwk(true_labels, mean_preds), 5)
print(f"The naive grouping of the assessments and taking the mode will yield us a QWK score of:\n{grouped_mode_score}")
print(f"The naive grouping of the assessments and taking the rounded mean will yield us a QWK score of:\n{grouped_mean_score}")


# ## Optimizing QWK <a id="5"></a>

# The most naive way to maximize QWK is to optimize the accuracy. This will give suboptimal results because the accuracy does not take into account small deviations from the target variable. 
# 
# So how do optimize QWK in a smart way?
# 
# In general there are two valid ways:
# 
# 1. Approach the modeling as a regression problem. Minimize the [Mean Squared Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error) and round the predictions from the model. Ideally, optimize the rounding thresholds.
# 2. Use QWK Directly as a loss function.

# ### 1. Approach the modeling as a regression problem.

# While Quadratic Weighted Kappa (QWK) fundamentally is a classification metric. It can be very beneficial to build regression models by minimizing MSE and round the predictions afterwards. This will in general give better results and is a much simples method than trying to implement QWK as a loss function.
# 
# 

# For example, let's take the mean predictions from the last section but with the rounding. We will use the mode predications to optimize the rounding thresholds and try to improve on the mean predictions.

# In[ ]:


# Map the mean based on the assessment title
raw_mean_mapping = df.groupby('title')['accuracy_group'].mean()
raw_mean_preds = df['title'].map(raw_mean_mapping)


# We can now optimize the round thresholds as to maximize the QWK score. When doing this in practice be careful not to use the validation data to optimize the thresholds as this can lead to target leakage.
# 
# Credits to [Kaggle Grandmaster Abhishek Thakur](https://www.kaggle.com/abhishek) for creating this "OptimizedRounder" class. The original class can be found in [this Kaggle kernel](https://www.kaggle.com/abhishek/optimizer-for-quadratic-weighted-kappa).

# In[ ]:


class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    """
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']


# In[ ]:


# Optimize rounding thresholds (No effect since we have naive baselines)
optR = OptimizedRounder()
optR.fit(mode_preds, true_labels)
coefficients = optR.coefficients()
opt_preds = optR.predict(raw_mean_preds, coefficients)
new_score = sklearn_qwk(true_labels, opt_preds)


# In[ ]:


print(f"Optimized Thresholds:\n{coefficients}\n")
print(f"The Quadratic Weighted Kappa (QWK)\nwith optimized rounding thresholds is: {round(new_score, 5)}\n")
print(f"This is an improvement of {round(new_score - grouped_mean_score, 5)} over the unoptimized rounding.")


# Unfortunately, in this case we will not improve because the predictions themselves are naive. However, in practice the optimized rounding will slightly improve the final QWK score.

# You can also use this Keras custom Callback to save the model with the highest QWK score.

# In[ ]:


class QWK(Callback):
    """
    A custom Keras callback for saving the best model
    according to the Quadratic Weighted Kappa (QWK) metric
    """
    def __init__(self, model_name="model.h5"):
        self.model_name = model_name
    
    def on_train_begin(self, logs={}):
        """
        Initialize list of QWK scores on validation data
        """
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Gets QWK score on the validation data
        
        :param epoch: The current epoch number
        """
        # Get predictions and convert to integers
        y_pred, labels = get_preds_and_labels(model, val_generator)
        y_pred = np.rint(y_pred).astype(np.uint8).clip(0, 4)
        _val_kappa = cpmp_qwk(labels, y_pred)
        self.val_kappas.append(_val_kappa)
        print(f"val_kappa: {round(_val_kappa, 4)}")
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save(self.model_name)
        return
    
def get_preds_and_labels(model, generator):
    """
    Get predictions and labels from the generator
    """
    preds = []
    labels = []
    for _ in range(int(np.ceil(generator.samples / BATCH_SIZE))):
        x, y = next(generator)
        preds.append(model.predict(x))
        labels.append(y)
    # Flatten list of numpy arrays
    return np.concatenate(preds).ravel(), np.concatenate(labels).ravel()


# ### 2. Use QWK Directly as a loss function.

# If you like you can also directly optimize the QWK by using it as a loss function. Here is an implementation for Tensorflow/Keras models.
# 
# [Source](https://stackoverflow.com/questions/54831044/how-can-i-specify-a-loss-function-to-be-quadratic-weighted-kappa-in-keras)

# In[ ]:


def _cohen_kappa(y_true, y_pred, num_classes, weights=None, metrics_collections=None, updates_collections=None, name=None):
    kappa, update_op = tf.contrib.metrics.cohen_kappa(y_true, y_pred, num_classes, weights, metrics_collections, updates_collections, name)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([update_op]):
        kappa = tf.identity(kappa)
    return kappa

def cohen_kappa_loss(num_classes, weights=None, metrics_collections=None, updates_collections=None, name=None):
    """
    A loss function that measures the Quadratic Weighted Kappa (QWK) score
    and can be used in a Tensorflow / Keras model
    """
    def cohen_kappa(y_true, y_pred):
        return -_cohen_kappa(y_true, y_pred, num_classes, weights, metrics_collections, updates_collections, name)
    return cohen_kappa


# ## Submission <a id="6"></a>

# The test data has some assessments that are in the training data, while other assessments are totally new. For the overlapping assessments we will predict the mode calculated on the training data. Once we aggregate on each installation id we get valid naive predictions that we submit to Kaggle.

# In[ ]:


# Read in Test Data
test_df = pd.read_csv(PATH + "test.csv")

# Map the mode to the test data and create the final predictions through aggregation
test_df['preds'] = test_df['title'].map(mode_mapping)
final_preds = test_df.groupby('installation_id')['preds'].agg(lambda x:x.value_counts().index[0])


# In[ ]:


# Make submission for Kaggle
sub_df = pd.read_csv(SUB_PATH)
sub_df['accuracy_group'] = list(final_preds.fillna(0).astype(np.uint8))
sub_df.to_csv("submission.csv", index=False);


# In[ ]:


print('Final predictions:')
sub_df.head(2)


# If you want to learn more about Quadratic Weighted Kappa I suggest watching [this video from the Coursera course "How to win Data Science Competitions"](https://www.coursera.org/lecture/competitive-data-science/classification-metrics-review-EhJzY). The part on QWK starts at 13:00.
# 
# 
# **That's it! If you like this Kaggle kernel, feel free to give an upvote and leave a comment! Your feedback is also very welcome! I will try to implement your suggestions in this kernel!**
