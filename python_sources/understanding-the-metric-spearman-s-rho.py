#!/usr/bin/env python
# coding: utf-8

# ### Understanding the Metric: Spearman's Rank Correlation Coefficient (Spearman's Rho)

# ![](http://i.hurimg.com/i/hdn/75/0x0/59c9a5f845d2a027e83ddaf9.jpg)

# In this kernel we explore the competition metric for the [Google QUEST Q&A Labeling 2019 competition](https://www.kaggle.com/c/google-quest-challenge). The competition metric is called [Spearman's Rank Correlation Coefficient (or Spearman's Rho)](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient). This metric is similar to Pearson correlation, but used the ranks of the data instead of the raw data.

# P.S. This kernel is the third in a series of kernels on metrics. Feel free to check out the previous ones on [Root Mean Square Logirithmic Error (RMSLE)](https://www.kaggle.com/carlolepelaars/understanding-the-metric-rmsle) and [Quadratic Weighted Kappa (QWK)](https://www.kaggle.com/carlolepelaars/understanding-the-metric-quadratic-weighted-kappa).

# ## Table of Contents

# - [Dependencies](#1)
# - [Preparation](#2)
# - [The Metric](#3)
# - [Speed Comparison](#4)
# - [Naive Baselines](#5)
# - [Optimizing Spearman's Rho](#6)
# - [Submission](#7)

# ## Dependencies <a id="1"></a>

# In[ ]:


# Standard dependencies
import os
import numpy as np
import random as rn
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import Callback

# Scipy's implementation of Spearman's Rho 
from scipy.stats import spearmanr

# Set seed for reproducability
seed = 1234
rn.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# Paths for easy data access
BASE_PATH = "../input/google-quest-challenge/"
TRAIN_PATH = BASE_PATH + "train.csv"
TEST_PATH = BASE_PATH + "test.csv"
SUB_PATH = BASE_PATH + "sample_submission.csv"


# In[ ]:


# File sizes and specifications
print('\n# Files and file sizes')
for file in os.listdir(BASE_PATH):
    print('{}| {} MB'.format(file.ljust(30), 
                             str(round(os.path.getsize(BASE_PATH + file) / 1000000, 2))))


# ## Preparation <a id="2"></a>

# In[ ]:


# All 30 targets
target_cols = ['question_asker_intent_understanding',
       'question_body_critical', 'question_conversational',
       'question_expect_short_answer', 'question_fact_seeking',
       'question_has_commonly_accepted_answer',
       'question_interestingness_others', 'question_interestingness_self',
       'question_multi_intent', 'question_not_really_a_question',
       'question_opinion_seeking', 'question_type_choice',
       'question_type_compare', 'question_type_consequence',
       'question_type_definition', 'question_type_entity',
       'question_type_instructions', 'question_type_procedure',
       'question_type_reason_explanation', 'question_type_spelling',
       'question_well_written', 'answer_helpful',
       'answer_level_of_information', 'answer_plausible', 'answer_relevance',
       'answer_satisfaction', 'answer_type_instructions',
       'answer_type_procedure', 'answer_type_reason_explanation',
       'answer_well_written']


# In[ ]:


# Read in training data
df = pd.read_csv(TRAIN_PATH)


# In[ ]:


print("Target variables:")
df[target_cols].head()


# ## The Metric <a id="3"></a>

# [Spearman's Rho](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) is one of the most popular ways to evaluate the correlation between variables. It is an appropriate metric for both continuous and discrete ordinal variables. The Spearman's Rho score will always be between -1 (perfect negative correlation) and 1 (Perfect correlation). The original formula can look quite daunting but for completeness we present it here:
# 
# ![](https://www.statisticshowto.datasciencecentral.com/wp-content/uploads/2015/01/tied-ranks-1.png)
# 
# R(x) and R(y) are the ranks.
# 
# R(x)bar and R(y)bar are the mean ranks.
# 
# [Image Source](https://www.statisticshowto.datasciencecentral.com/wp-content/uploads/2015/01/tied-ranks-1.png)
# 
# We can clean up the formula a bit if you are familiar with how covariance and the standard deviation are calculated. We define Spearman's Rho as the Covariance of the ranks of the variables divided by the multiplied standard deviations of the ranks:
# 
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/a8dda555d22080d721679401fa13181cad3863f6) 
# 
# If all ranks are distinct integers we can use a popular simplified formula to calculate Spearman's Rho:
# 
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/b69578f3203ecf1b85b1a0929772b376ae07a3ce)
# 
# 

# We can use an optimized implementation from Scipy that uses [Cython](https://cython.org/) and can already calculate Spearman's Rho pretty efficiently. It will also provide the p-value for the calculation.

# In[ ]:


def spearmans_rho(y_true, y_pred, axis=0):
    """
        Calculates the Spearman's Rho Correlation between ground truth labels and predictions 
    """
    return spearmanr(y_true, y_pred, axis=axis)


# Let's implement Spearman's R using only NumPy to get better insight in the structure of the formula and see how you can implement it yourself in Python.

# In[ ]:


def _get_ranks(arr: np.ndarray) -> np.ndarray:
    """
        Efficiently calculates the ranks of the data.
        Only sorts once to get the ranked data.
        
        :param arr: A 1D NumPy Array
        :return: A 1D NumPy Array containing the ranks of the data
    """
    temp = arr.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(arr))
    return ranks

def spearmans_rho_custom(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
        Efficiently calculates the Spearman's Rho correlation using only NumPy
        
        :param y_true: The ground truth labels
        :param y_pred: The predicted labels
    """
    # Get ranked data
    true_rank = _get_ranks(y_true)
    pred_rank = _get_ranks(y_pred)
    
    return np.corrcoef(true_rank, pred_rank)[1][0] 


# Also, [Abhishek Thakur](https://www.kaggle.com/abhishek) provided us with a nice implementation of Spearman's Rho as a callback compatible with Tensorflow/Keras. Use this callback if you are training a Tensorflow/Keras model and want to keep score on Spearman's Rho.
# 
# [The implementation was copied from this Kaggle kernel](https://www.kaggle.com/abhishek/distilbert-use-features-oof)

# In[ ]:


class SpearmanRhoCallback(Callback):
    def __init__(self, training_data, validation_data, patience, model_name):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        
        self.patience = patience
        self.value = -1
        self.bad_epochs = 0
        self.model_name = model_name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        rho_val = np.mean([spearmanr(self.y_val[:, ind], y_pred_val[:, ind] + np.random.normal(0, 1e-7, y_pred_val.shape[0])).correlation for ind in range(y_pred_val.shape[1])])
        if rho_val >= self.value:
            self.value = rho_val
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            print("Epoch %05d: early stopping Threshold" % epoch)
            self.model.stop_training = True
            #self.model.save_weights(self.model_name)
        print('\rval_spearman-rho: %s' % (str(round(rho_val, 4))), end=100*' '+'\n')
        return rho_val

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


# ## Speed Comparison <a id="4"></a>

# Our custom implementation only needs to sort once, but it still takes advantage of the optimized function from NumPy. It also calculates only the correlation without p-values. This is probably why it runs a little faster than Scipy's version. We will show this with an example using sampled linear data with noise.

# In[ ]:


# Sample two times from distributions that are highly correlated
samp_size = 1000000
norm_num = np.arange(samp_size) + np.random.normal(0, 10, samp_size)
norm_num2 = np.arange(samp_size) + np.random.normal(0, 100000, samp_size)


# ### Speed Test Scipy's Implementation

# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'spearmanr(norm_num, norm_num2)[0]')


# ### Speed Test Custom Implementation

# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'spearmans_rho_custom(norm_num, norm_num2)')


# ## Naive Baselines <a id="5"></a>

# ### 1. Predict using random uniform predictions

# The most basic benchmark is to sample from a random uniform distribution ([0,1]). This will give us a score that is close to 0.

# In[ ]:


corrs = []
# Make random predictions
for col in target_cols:
    naive_preds = np.random.rand(len(df))
    corr = spearmans_rho_custom(naive_preds, df[col])
    corrs.append(corr)
rand_baseline = np.mean(corrs)
print(f"Spearman's Rho Score for random uniform predictions: {round(rand_baseline, 6)}")


# ### 2. Predict mean rank with noise

# A second way to formulate a naive baseline is to predict the mean of the column. However, we have to add a little noise to avoid a division by zero error. Remember that in order to calculate Spearman's Rho we have to divide by standard deviation of the first column multiplied by the standard deviation of the second column. If our prediction is completely constant than the standard deviation will be zero and hence we will get an error.
# 
# Unfortunately, this will not improve compared to the random predictions.

# In[ ]:


corrs = []
# Predict the mean and a small amount of noise to avoid division by zero
for col in target_cols:
    probs = df[col].value_counts().values / len(df)
    vals = list(df[col].value_counts().index)
    naive_preds = df[col].mean() + np.random.normal(0, 1e-15, len(df))
    corr = spearmans_rho_custom(naive_preds, df[col])
    corrs.append(corr)
mean_baseline = np.mean(corrs)
print(f"Spearman's Rho Score for predicting the mean with some noise: {round(mean_baseline, 6)}")


# ### 3. Predict using distribution based on the data

# Thirdly, we can create a naive baseline using the probability that a value occurs and use these probabilities to create a new distribution to sample from. Unfortunately, there will be no increase in score compared to the random baseline.

# In[ ]:


corrs = []
# Calculate probability of some prediction and sample according to those probabilities
for col in target_cols:
    probs = df[col].value_counts().values / len(df)
    vals = list(df[col].value_counts().index)
    naive_preds = np.random.choice(vals, len(df), p=probs)
    corr = spearmanr(naive_preds, df[col])[0]
    corrs.append(corr)
dist_baseline = np.mean(corrs)
print(f"Spearman's Rho Score for sampling from calculated distribution: {round(dist_baseline, 6)}")


# ## Optimizing Spearman's Rho <a id="6"></a>

# In essence there are two choices to optimize for the Spearman's Rho score (in this competition):
# 
# 1. Use binary_crossentropy since it is essentially a binary classification problem.
# 2. Implement a custom loss function to optimize Spearman's Rho directly.

# ### 1. Use binary_crossentropy

# Binary Crossentropy is a common loss function that is already implemented in most libraries. Just select "binary_crossentropy" as a loss function in the Machine Learning library of your choice. ;)

# ### 2. Optimize Spearman's Rho directly

# Work in Progress: Loss Function to optimize Spearman's Rho directly.

# ## Submission <a id="7"></a>

# In[ ]:


# Read in sample submission file
sub_df = pd.read_csv(SUB_PATH)

# Make random predictions
for col in target_cols:
    naive_preds = np.random.rand(len(sub_df))
    sub_df[col] = naive_preds.round(6)
    
sub_df.to_csv('submission.csv', index=False)


# In[ ]:


print('Final predictions:')
sub_df.head(2)


# Unfortunately I have not found a way to get a naive baseline that performs better than random uniform. Please let me know in the comments if there is a way to do better than random. I will implement it in this kernel.

# **That's it! If you like this Kaggle kernel, feel free to give an upvote and leave a comment! Your feedback is also very welcome! I will try to implement your suggestions in this kernel!**
