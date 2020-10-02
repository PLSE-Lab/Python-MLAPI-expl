#!/usr/bin/env python
# coding: utf-8

# # Resampling Methods
# 
# Process of repeatedly drawing samples from a data set and refitting a given model on each sample with the goal of learning more about the fitted model. Resampling methods can be expensive since they require repeatedly performing the same statistical methods on N different subsets of the data.
# 
# **Following are types of resampling methods:**
# * Bootstrap Sampling
# * K Fold Cross Validation
# * Leave One Out Cross Validation

# # 1. Load packages and observe dataset

# In[ ]:


#Import numerical libraries
import numpy as np
from numpy import array
import pandas as pd

#Import graphical plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Import resampling and modeling algorithms
from sklearn.utils import resample # for Bootstrap sampling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#KFold CV
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')


# # 2. Bootstrap sampling method

# ### Agenda - Bootstrap sampling
# 
# * Configure bootstrap
# * Run bootstrap in train, test data and obtain accuracy scores
# * Visual representation using Histogram and derive Confidence levels 

# Here a random function is used to create samples from original data. Within a sample set, there could be duplicates or more however 2 sample sets are unlikely to be 100% same. Due to the drawing with replacement, a bootstrapped data set may contain multiple instances of the same original cases, and may completely omit other original cases.
# 
# This technique is used in machine learning to estimate the skill of machine learning models when making predictions on data not included in the training set. This technique helps to fine tune the model even before we give it access to test data (real world data). This allows us to tweak the model hyperparameters to achieve the best score.
# 
# Bootstrapping is primarily used to establish empirical distribution functions for a widespread range of statistics.
# 
# A desirable property of the results from estimating machine learning model skills is that the estimated skill can be presented with confidence intervals, a feature not readily available with other methods such as Cross Validation.

# In[ ]:


data = pd.read_csv('../input/handson-pima/Hands on Exercise Feature Engineering_ pima-indians-diabetes (1).csv')
data.head()

values = data.values


# In[ ]:


#Lets configure Bootstrap

n_iterations = 10  #No. of bootstrap samples to be repeated (created)
n_size = int(len(data) * 0.50) #Size of sample, picking only 50% of the given data in every bootstrap sample


# In[ ]:


#Lets run Bootstrap
stats = list()
for i in range(n_iterations):

    #prepare train & test sets
    train = resample(values, n_samples = n_size) #Sampling with replacement..whichever is not used in training data will be used in test data
    test = np.array([x for x in values if x.tolist() not in train.tolist()]) #picking rest of the data not considered in training sample
    
    #fit model
    model = DecisionTreeClassifier()
    model.fit(train[:,:-1], train[:,-1]) #model.fit(X_train,y_train) i.e model.fit(train set, train label as it is a classifier)
    
    #evaluate model
    predictions = model.predict(test[:,:-1]) #model.predict(X_test)
    score = accuracy_score(test[:,-1], predictions) #accuracy_score(y_test, y_pred)
    #caution, overall accuracy score can mislead when classes are imbalanced
    
    print(score)
    stats.append(score)


# Here each Bootstrap iteration sample would create one model and this model is tested against the Out of Bag (test data) of that sample, i.e we will test that sample with the test sample not part of that sample.
# 
# **Thus we obtain accuracy scores for 10 samples.**

# In[ ]:


#Lets plot the scores to better understand this visually

plt.hist(stats)
plt.figure(figsize = (10,5))


# ### Confidence Intervals
# Confidence intervals refers to the % of all possible samples that can be expected to include the true population parameter.
# 
# For eg. 95% of all samples would be found in this interval range.

# In[ ]:


#Lets find Confidence intervals

a = 0.95 # for 95% confidence
p = ((1.0 - a)/2.0) * 100 #tail regions on right and left .25 on each side indicated by P value (border)
                          # 1.0 is total area of this curve, 2.0 is actually .025 thats the space we would want to be 
                            #left on either side
lower = max(0.0, np.percentile(stats,p))

p = (a + ((1.0 - a)/ 2.0)) * 100 #p is limits
upper = min(1.0, np.percentile(stats,p))
print('%.1f confidence interval %.1f%% and %.1f%%' %(a*100, lower*100, upper*100))


# **As the no. of iterations are increased, this histogram tends to acquire a Normal distribution. Increasing the no. of iterations from 10 to 100 or 500 would give a better histogram.**

# # 3.a Cross Validation - KFold

# Cross validation resamples without replacement and thus produces surrogate data sets that are smaller than the original. These data sets are produced in a systematic way so that after a pre-specified number k of surrogate data sets, each of the n original cases has been left out exactly once. This is called k-fold cross validation or leave-x-out cross validation with x = n/k, e.g. leave-one-out cross validation omits 1 case for each surrogate set, i.e. k=n.
# 
# Primary purpose of CV is measuring performance of a model

# ### Agenda - KFold CV
# 
# * Fit Logistic Regression and compute cross_val_score
# * Calculate accuracy of this model

# In[ ]:


#Create separate arrays such that only values are considered as X, y
values = data.values
X = values[:,0:8]
y = values[:,8]

#Split the data into train,test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.50, random_state = 1)


# In[ ]:


#Lets configure Cross Validation
#default value of n_splits = 10
kfold = KFold(n_splits = 50, random_state = 7)
model = LogisticRegression()
results = cross_val_score(model,X,y,cv = kfold)


# In[ ]:


print(results)


# Since 50 n_folds is requested hence you received 50 iterations of this test set. For 50 test sets, 50 training sets are created and it gives 50 different accuracy scores.

# In[ ]:


#What's the accuracy of this model using KFold CV

print('Accuracy:  %.3f%% (%.3f%%)' % (results.mean()*100.0, results.std()*100.0))


# **This model is likely to give an accuracy score of 77.017 +- 10.621(std dev). Putting this in normal distribution, you will get the score as per Central Limit theorem**

# # 3.b Cross Validation - Leave One Out CV

# Here the dataset k is split into k-1 train sets and 1 test set.

# ### Agenda - LOOCV
# 
# * Provide a dataset with values
# * Fit LOOCV and print the train, test set values
# * Calculate accuracy of this model

# In[ ]:


# scikit-learn k-fold cross-validation

# data sample
data = array([10,20,30,40,50,60,70,80,90,100])
# prepare cross validation
loocv = LeaveOneOut()
# enumerate splits
for train, test in loocv.split(data):
    print('train: %s, test: %s' % (data[train], data[test]))


# Here the data comprises of an array from 10 - 100, LOOCV will leave one of these values out as a test value.

# *For further info: https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/*
# *and https://datascience.stackexchange.com/questions/32264/what-is-the-difference-between-bootstrapping-and-cross-validation *
