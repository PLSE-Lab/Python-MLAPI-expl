#!/usr/bin/env python
# coding: utf-8

# In iMaterialist Furniture Challenge we have classical problem when class distribution is different for train and test.
# 
# Recently on Kaggle in Quora Question Pairs we had the same problem and you can read about it here https://www.kaggle.com/c/quora-question-pairs/discussion/31179 . But I'll try to explain solution in slightly different way and show how to apply it for multiclass problem.

# In this competition 128 classes and they distributed like this for train and validation:

# In[29]:


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')

train_json = json.load(open('../input/imaterialist-challenge-furniture-2018/train.json'))
train_df = pd.DataFrame(train_json['annotations'])

f, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(15, 5))
ax0.hist(train_df.label_id.value_counts())
ax0.set_xlabel("# images per class")
ax0.set_ylabel("# classes")
ax0.set_title('Class distribution for Train')

val_pred = np.loadtxt('../input/furniture2018val/furniture_val_true.csv')
ax1.hist(pd.Series(val_pred).value_counts())
ax1.set_xlabel("# images per class")
ax1.set_title('Class distribution for Validation')
f;


# In train some classes have 4k examples other only 500. It's 4x difference. But validation has balanced dataset - 50 images per class. We also can assume from number of images in test (12800=128*100) - it's balanced too.

# I've tried oversampling and using weigths for CrossEntropy loss to solve the problem. But the best score I got with calibration.

# ## Probability calibration

# Let's start with 2 class problem. From Bayesian perspective our final predicted probability from unbalanced train dataset could be seen as:
# 
# $$
# (0)
# \left\{
# \begin{array}{rl}
#  P(y_0|X) \propto Pr(y_0) L(X|y_0)  \\
#  P(y_1|X) \propto Pr(y_1) L(X|y_1)
# \end{array}
# \right. 
# $$
# 
# $P(y_0|X)$ - predicted probability for class $y_0$; $Pr(y_0)$ - prior probability for $y_0$; $L(X|y_0)$ some likelihood of some data for $y_0$. I'm using $L$ and $Pr$ to not confuse you with bunch of $P$.
# 
# In words, our predicted probability is multiplication of some prior probability and some function from X.
# 
# For different distribution, likelihood should be the same, but because prior is different, we get different probability:
# 
# $$
# (1)
# \left\{
# \begin{array}{rl}
#  P(y_0|X)' \propto Pr(y_0)' L(X|y_0)  \\
#  P(y_1|X)' \propto Pr(y_1)' L(X|y_1)
# \end{array}
# \right.
# $$
# 
# $P(y_0|X)'$ is our corrected probability which we need to calculate. And we know desire priors (1/2 for balanced), and becase likelihood is the same we can get it from (0):
# 
# $$
# (2)
# \left\{
# \begin{array}{rl}
#  L(X|y_0)  \propto \frac{P(y_0|X)}{Pr(y_0)}   \\
#  L(X|y_1)  \propto \frac{P(y_1|X)}{Pr(y_1)}
# \end{array}
# \right. 
# $$
# 
# and insert likelihood in (1) to get corrected probability:
# $$
# (3)
# \left\{
# \begin{array}{rl}
#  P(y_0|X)' \propto Pr(y_0)' \frac{P(y_0|X)}{Pr(y_0)})  \\
#  P(y_1|X)' \propto Pr(y_1)' \frac{P(y_1|X)}{Pr(y_1)}
# \end{array}
# \right.
# $$
# and it's almost our final formula, we just need to normalize probability so they sum to 1.

# Changing this formula to multiclass is quite easy. For every class we just represent other 127 classes as one class.

# We're ready to move to practice.
# 
# Where I'll also use:
# $$
# P(y_1|X) = 1 - P(y_0|X) \\
# Pr(y_1) = 1 - Pr(y_0) \\
# Pr(y_1)' = 1 - Pr(y_0)' \\
# Pr(y_0)' = \frac{1}{128} \\
# $$

# In[30]:


def calibrate(prior_y0_train, prior_y0_test,
              prior_y1_train, prior_y1_test,
              predicted_prob_y0):
    predicted_prob_y1 = (1 - predicted_prob_y0)
    
    p_y0 = prior_y0_test * (predicted_prob_y0 / prior_y0_train)
    p_y1 = prior_y1_test * (predicted_prob_y1 / prior_y1_train)
    return p_y0 / (p_y0 + p_y1)  # normalization


# In[31]:


prior_y0_test = 1/128
prior_y1_test = 1 - prior_y0_test

def calibrate_probs(prob):
    calibrated_prob = np.zeros_like(prob)
    nb_train = train_df.shape[0]
    for class_ in range(128): # enumerate all classes
        prior_y0_train = ((train_df.label_id - 1) == class_).mean()
        prior_y1_train = 1 - prior_y0_train
        
        for i in range(prob.shape[0]): # enumerate every probability for a class
            predicted_prob_y0 = prob[i, class_]
            calibrated_prob_y0 = calibrate(
                prior_y0_train, prior_y0_test,
                prior_y1_train, prior_y1_test,                
                predicted_prob_y0)
            calibrated_prob[i, class_] = calibrated_prob_y0
    return calibrated_prob


# In[32]:


# let's apply calibration to validation 
val_prob = np.loadtxt('../input/furniture2018val/furniture_val_prob.csv', delimiter=',')
calibrated_val_prob = calibrate_probs(val_prob)


# In[33]:


val_predicted = np.argmax(val_prob, axis=1)
calibrated_val_predicted = np.argmax(calibrated_val_prob, axis=1)


# In[34]:


f, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
ax0.hist(pd.Series(val_predicted).value_counts(), bins=20)
ax0.set_xlabel('# images per class')
ax0.set_ylabel("# classes")
ax0.set_title('Before')

ax1.hist(pd.Series(calibrated_val_predicted).value_counts(), bins=20)
ax1.set_xlabel('# images per class')
ax0.set_ylabel("# classes")
ax1.set_title('After')

ax2.hist(pd.Series(list(range(128))*50).value_counts())
ax2.set_xlabel('# images per class')
ax0.set_ylabel("# classes")
ax2.set_title('Ideal')
f;


# So our calibrated probability is slightly close to our ideal distribution.
# 
# Let's see how good calibration for score:

# In[35]:


val_true = np.loadtxt('../input/furniture2018val/furniture_val_true.csv', delimiter=',')
print('Score for raw probability:', (val_true != val_predicted).mean())
print('Score for calibrated probability:', (val_true != calibrated_val_predicted).mean())


# So we got 0.005 improvement.

# Happy Kaggling!
