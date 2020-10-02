#!/usr/bin/env python
# coding: utf-8

# ## Understanding Root Mean Squared Logarithmic Error (RMSLE)

# In this kernel we take a deep dive into Root Mean Squared Logarithmic Error (RMSLE). This is the metric used for the [ASHRAE Energy Prediction competition](https://www.kaggle.com/c/ashrae-energy-prediction) and a common metric for regression problems. It is an extension on [Mean Squared Error (MSE)](https://en.wikipedia.org/wiki/Root-mean-square_deviation) that is mainly used when predictions have large deviations, which is the case with this energy prediction competition. Values range from 0 up to millions and we don't want to punish deviations in prediction as much as with MSE.
# 
# We will explore the metric itself as well as some of the best naive predictions that are possible.
# 
# P.S. This kernel is the first in a series of kernels on competition metrics. Feel free to check out ["Episode Two" of Understanding the metric on Quadratic Weighted Kappa](https://www.kaggle.com/carlolepelaars/understanding-the-metric-quadratic-weighted-kappa) and ["Episode Three" of Understanding the metric on Spearman's Rho](https://www.kaggle.com/carlolepelaars/understanding-the-metric-spearman-s-rho).

# ## Table Of Contents

# - [Preparation](#1)
# - [The Metric (RMSLE)](#2)
# - [Best Baselines](#3)
# - [Submission with best constant](#4)

# ## Preparation <a id="1"></a>

# In[ ]:


# Standard libraries
import os
import numpy as np
import pandas as pd
import random as rn
import tensorflow as tf

# Visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# For calculating the metric
from sklearn.metrics import mean_squared_log_error

# Path specifications
BASE_PATH = "../input/ashrae-energy-prediction/"
TRAIN_PATH = BASE_PATH + "train.csv"
SAMP_SUB_PATH = BASE_PATH + "sample_submission.csv"

# Seed for reproducability
seed = 1234
rn.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


# In[ ]:


# Read in data
df = pd.read_csv(TRAIN_PATH)
# Remove outliers
df = df[df['meter_reading'] < 250000]


# ## The metric (RMSLE) <a id="2"></a>

# The Root Mean Squared Log Error (RMSLE) can be defined using a slight modification on sklearn's mean_squared_log_error function, which itself a modification on the familiar [Mean Squared Error (MSE)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) metric.
# 
# The formula for RMSLE is represented as follows:
# 
# RMSLE = $\sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+1))^2 }$
# 
# Where:
# 
# $n$ is the total number of observations in the (public/private) data set,
# 
# $p_i$ is your prediction of target, and
# 
# $a_i$ is the actual target for $i$.
# 
# $log(x)$ is the natural logarithm of $x$ ($log_e(x)$.
# 
# [Formula source: Kaggle Evaluation page for the ASHRAE 2019 competition](https://www.kaggle.com/c/ashrae-energy-prediction/overview/evaluation)

# In[ ]:


def RMSLE(y_true:np.ndarray, y_pred:np.ndarray) -> np.float64:
    """
        The Root Mean Squared Log Error (RMSLE) metric 
        
        :param y_true: The ground truth labels given in the dataset
        :param y_pred: Our predictions
        :return: The RMSLE score
    """
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# In[ ]:


def NumPyRMSLE(y_true:list, y_pred:list) -> float:
    """
        The Root Mean Squared Log Error (RMSLE) metric using only NumPy
        N.B. This function is a lot slower than sklearn's implementation
        
        :param y_true: The ground truth labels given in the dataset
        :param y_pred: Our predictions
        :return: The RMSLE score
    """
    n = len(y_true)
    msle = np.mean([(np.log(y_pred[i] + 1) - np.log(y_true[i] + 1)) ** 2.0 for i in range(n)])
    return np.sqrt(msle)


# RMSLE For Tensorflow (From [Shashi Prakash Tripathi](https://www.kaggle.com/shishu1421) in the comments)

# In[ ]:


def RMSLETF(y_pred:tf.Tensor, y_true:tf.Tensor) -> tf.float64:
    '''
        The Root Mean Squared Log Error (RMSLE) metric for TensorFlow / Keras
        
        :param y_true: The ground truth labels given in the dataset
        :param y_pred: Predicted values
        :return: The RMSLE score
    '''
    y_pred = tf.cast(y_pred, tf.float64)
    y_true = tf.cast(y_true, tf.float64) 
    y_pred = tf.nn.relu(y_pred) 
    return tf.sqrt(tf.reduce_mean(tf.squared_difference(tf.log1p(y_pred), tf.log1p(y_true))))


# ## Best baselines <a id="3"></a>

# ### 1. Constant predictions

# Making a constant prediction is a good way to get a sense of what it means to have good performance on model. You can build a complex model that can get 95% accuracy, but if a constant prediction will give you that as well than 95% doesn't look so good anymore. A metric like [AUC (Area under ROC Curve)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) will for example give a more honest representation for constant predictions, which is 0.5 and the same for random predictions.
# 
# The best constant score for RMSLE is the exponential of the mean of the log target values. It can be expressed as a formula in the following way.
# 
# #### Best constant for RMSLE = $\mathrm{e}^{mean(log_e(targets))}$
# 
# We will derive this conclusion for ourselves by iteratively increasing the constant value and checking where the RMSLE score is the lowest.

# In[ ]:


mean = np.mean(df['meter_reading'])

print(f"RMSLE for predicting only 0: {round(RMSLE(df['meter_reading'], np.zeros(len(df))), 5)}")
print(f"RMSLE for predicting only 1: {round(RMSLE(df['meter_reading'], np.ones(len(df))), 5)}")
print(f"RMSLE for predicting only 50: {round(RMSLE(df['meter_reading'], np.full(len(df), 50)), 5)}")
print(f"RMSLE for predicting the mean ({round(mean, 2)}): {round(RMSLE(df['meter_reading'], np.full(len(df), mean)), 5)}")


# In[ ]:


const_rmsles = dict()
for i in range(75):
    const = i*2
    rmsle = round(RMSLE(df['meter_reading'], np.full(len(df), const)), 5)
#     print(f"RMSLE for predicting only {const}: {rmsle}")
    const_rmsles[const] = rmsle

xs = list(const_rmsles.keys())
ys = list(const_rmsles.values())

pd.DataFrame(ys, index=xs).plot(figsize=(15, 10), legend=None)
plt.scatter(min(const_rmsles, key=const_rmsles.get), sorted(ys)[0], color='red')
plt.title("RMSLE scores for constant predictions", fontsize=18, weight='bold')
plt.xticks(fontsize=14)
plt.xlabel("Constant", fontsize=14)
plt.ylabel("RMSLE", rotation=0, fontsize=14);


# In the graph above we can see that the RMSLE score is the lowest around 60. However, you probably would like to know the exact value. This can easily be calculate using familiar NumPy functions.

# In[ ]:


# Formulate the best constant for this metric
best_const = np.expm1(np.mean(np.log1p(df['meter_reading'])))


# In[ ]:


print(f"The best constant for our data is: {best_const}...")
print(f"RMSLE for predicting the best possible constant on our data: {round(RMSLE(df['meter_reading'], np.full(len(df), best_const)), 5)}\n")

print("This is the optimal RMSLE score that we can get with only a constant prediction and using all data available.\nWe therefore call it the best 'Naive baseline'\nA model should at least perform better than this RMSLE score.")


# ### 2. Random Predictions

# It can be interesting to check the score that random predictions will give you. This will for example help you identify when your predictions are shuffled accidentally or when there is something wrong with your model. For this competition there is no clear "maximum value" that would make sense to predict. We therefore will try out different maximum thresholds for random predictions. The minimum value will stay 0 for our experiment.

# In[ ]:


# Random predictions
rand_rmsles = dict()
for i in range(15):
    magn = 10**(0.2*(i+1))
    rand_preds = np.random.randint(0, magn, len(df))
    rmsle = round(RMSLE(df['meter_reading'], rand_preds), 5)
    rand_rmsles[magn] = rmsle

xs = list(rand_rmsles.keys())
ys = list(rand_rmsles.values())  
    
pd.DataFrame(ys, index=xs).plot(figsize=(15, 10), legend=None)
plt.scatter(min(rand_rmsles, key=rand_rmsles.get),sorted(ys)[0],color='red')
plt.title("RMSLE scores for random predictions", fontsize=18, weight='bold')
plt.xticks(fontsize=14)
plt.xlabel("Maximum value", fontsize=14)
plt.ylabel("RMSLE", rotation=0, fontsize=14);


# We can conclude that selecting random values between 0 and 160 will yield close to optimal performance regarding random predictions. Note that the best RMSLE score for random predictions (around 2.34) is not better than the best constant prediction. It can nonetheless still be interesting to analyze what we can expect from predicting random values.

# ## Submission with best constant <a id="4"></a>

# In[ ]:


# Read in sample submission and fill all predictions with the best constant
samp_sub = pd.read_csv(SAMP_SUB_PATH)
samp_sub['meter_reading'] = best_const
samp_sub.to_csv("best_constant_submission.csv", index=False)


# In[ ]:


# Check Final Submission
print("Final Submission:")
samp_sub.head(2)


# **That's it! If you like this Kaggle kernel, feel free to give an upvote and leave a comment! Your feedback is also very welcome! I will try to implement your suggestions in this kernel!**
