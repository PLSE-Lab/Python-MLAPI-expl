#!/usr/bin/env python
# coding: utf-8

# This Notebook is inspired by **fast.ai's Machine Learning course**. I really liked the teaching methodology as its completely application oriented, in oppose to other theorotical approaches in academic course. I learnt a lot of practical things,  like how to actually work with categorical features, how to normalize continuous data, how to handle missing values in your dataset, how to make the best use of timestamps in your dataset, best practices while creating trainning, validation and test datasets, what actually random forest do (intuitionally) and realized how easy it is to tuning hyperparameters and many more things.
# You might have noted that I have used "how to" in front of almost all the things that I learnt because this is what I actually learnt: "how to do something", instead of plain theory.
# 
# This notebook is **meant for people's who want to learn all these thing but cannnot manage time to watch 1-1:30 hrs long videos** (there are 7 lectures in the ML course, in total). I tried my best:
# - To put in as much stuff as possible.
# - To keep it simple and intuitional.
# - To avoid using fastai library (I personally think it requires you to learn a lot of context).
#     
# I have taken the trouble (rather, I enjoyed doing it) of going through the code and tried replicating the codes as normal python functions. So, you can see how the things are actually implemented instead of looking at some random function name, knowing what it does but not having any clue of how its implemented.
#  
# > Note: This notebook is not complete and exhaustive, I have not implemented everything that was covered in the course but I tried my best to make the notebook as end-to-end as possible.
# 
# > Note: This is pretty long notebook, you might want to bookmark it. The notebook will also have a [Part-2](https://www.kaggle.com/ankursingh12/lessons-learnt-from-fast-ai-lectures-part-2) covering the rest of the lessons learnt.

# ### Introduction to Blue Book for Bulldozers
# 
# We will be looking at the Blue Book for Bulldozers Kaggle Competition: "The goal of the contest is to predict the sale price of a particular piece of heavy equiment at auction based on it's usage, equipment type, and configuaration. The data is sourced from auction result postings and includes information on usage and equipment configurations."
# 
# This is a very common type of dataset and prediciton problem, and similar to what you may see in your project or workplace.
# 
# 
# 
# Kaggle provides info about some of the fields of our dataset; on the [Kaggle Data info page](https://www.kaggle.com/c/bluebook-for-bulldozers/data) they say the following:
# 
# For this competition, you are predicting the sale price of bulldozers sold at auctions. The data for this competition is split into three parts:
# 
# - **Train.csv** is the training set, which contains data through the end of 2011.
# - **Valid.csv** is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.
# - **Test.csv** is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.
# 
# The key fields are in train.csv are:
# 
# - SalesID: the uniue identifier of the sale
# - MachineID: the unique identifier of a machine. A machine can be sold multiple times
# - saleprice: what the machine sold for at auction (only provided in train.csv)
# - saledate: the date of the sale
# 
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import math
import re

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In any sort of data science work, **it's important to look at your data**, to make sure you understand the format, how it's stored, what type of values it holds, etc. Even if you've read descriptions about your data, the actual data may not be what you expect. So, lets start by look at out data.

# In[ ]:


path = '../input/train/'


# In[ ]:


get_ipython().system('ls {path}')


# In[ ]:


df_org = pd.read_csv(f'{path}Train.csv', low_memory=False, parse_dates=['saledate']); df_org.head().T


# In[ ]:


df_org.describe(include='all').T


# It's important to note what metric is being used for a project. Generally, selecting the metric(s) is an important part of the project setup. However, in this case Kaggle tells us what metric to use: **RMSLE** (root mean squared log error) **between the actual and predicted auction prices**. Therefore we take the log of the prices, so that RMSE will give us what we need.

# In[ ]:


df_org.SalePrice = np.log(df_org.SalePrice) #taking log, now over metric is simple RMSE instead of RMSLE.


# ### Intializing the model
# Once you have your dataset and metric in place, first thing you should always do is try training a model (here Random Forest, with default parameters) on your data.

# In[ ]:


m = RandomForestRegressor()
m.fit(df_org.drop('SalePrice', axis=1), df_org.SalePrice)


# Ooops, you have error !!
# First thing, DON'T PANIC. Scroll to the very end of the error log and read the last line. It precisely tells you what the error is; **could not convert string to float: 'Low'**. It can not convert string to float, that means you will have to do it because all your Machine Learning model demand numbers(only) as input.
# 
# ### Categorical Features
# 
# This dataset contains a mix of **continuous** and **categorical** variables. We need to separate the categorical variables from continuous once and then encode(i.e convert) them into some numerical form. Simple ..huh??
# 
# > **Lesson 1:** Don't be afraid of errors, instead LOVE them. Look at them as an opportunity to learn something, you don't already know.
# 
# Now, we know the problem, the categorical variables are currently stored as strings, which is unamiable and inefficient, and doesn't provide the numeric coding required for a random forest. Therefore we need to convert strings to pandas categories. One can do it manually, I personally would prefer to write a python code which does it for me. There are several other advantages of writing a python code, for example it will work even if you have different dataset. Also, Imagine if you have 1000's of features, doing it manually would be a huge waste of time.

# In[ ]:


df_raw = df_org.copy() # keeping a copy of our original dataset aside


# In[ ]:


categorical = []
for col in df_raw.columns[:]:
    if df_raw[col].dtype == 'object' : categorical.append(col)  # pandas treat "str" as "object"
categorical # list of all the variables which are strings


# In[ ]:


for col in categorical: df_raw[col] = df_raw[col].astype("category").cat.as_ordered()


# In[ ]:


# you can select a column and have alook at the categories
df_raw.UsageBand.cat.categories


# In[ ]:


# We can specify the order to use for categorical variables if we wish:
df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)


# Normally, pandas will continue displaying the text categories, while treating them as numerical data internally. Optionally, we can replace the text categories with numbers, which will make this variable non-categorical, like so:

# In[ ]:


df_raw.UsageBand = df_raw.UsageBand.cat.codes


# > **Note:** When you convert categories into codes, all the missing values are replaced with "-1". 

# Now I guess we can feed the dataset into the RF model for training. So lets see if it works.

# In[ ]:


m = RandomForestRegressor()
m.fit(df_raw.drop('SalePrice', axis=1), df_raw.SalePrice)


# Oops, Another error !!
# 
# First thing, DON'T PANIC. Read the last line, it says "timestamp" is not supported. We will have to convert the "timestamp" variable into some numerical format. There are two possible ways of handling this situation.
# - You can delete the timestamp column (i.e. saledate) completely. OR
# - You can do some feature engineering on that column (our approach).
# 
# ### Time Stamp
# 
# We will extract particular date fields from a complete datetime for the purpose of constructing categoricals. You should always consider this feature extraction step when working with date-time. Without expanding your date-time into these additional fields, you can't capture any trend/cyclical behavior as a function of time at any of these granularities.
# 
# > **Lesson 2:** Always use "add_datepart" or similar method for feature extraction when working with date-time.

# In[ ]:


def add_datepart(df, fldname, drop=True, time=False):
    "Helper function that adds columns relevant to a date in the column `fldname` of `df`."
    fld = df[fldname]
    fld_dtype = fld.dtype
    
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtypeType):
        fld_dtype = np.datetime64
        
    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
         
    prefix = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: 
        attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: 
        df[prefix + n] = getattr(fld.dt, n.lower())
    df[prefix + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
        
    if drop: df.drop(fldname, axis=1, inplace=True)


# In[ ]:


add_datepart(df_raw, 'saledate'); df_raw.head().T


# ### Missing values
# We're still not quite done - for instance we have lots of missing values, which we can't pass directly to a random forest. 

# In[ ]:


df_raw.isnull().mean().sort_index()


# Missing values in categorical variables are replaced by "-1" by pandas. You only have to take care of missing values in continuous variables. One easy and intuitional way is to fill all the missing values with  the column's mean value.

# In[ ]:


def fix_missing(df, na_dict):
    """ Fill missing data in a column of df with the median, and add a {name}_na column
    which specifies if the data was missing."""
    for name,col in df.items():
        if is_numeric_dtype(col):
            if pd.isnull(col).sum():
                df[name+'_na'] = pd.isnull(col)
                filler = na_dict[name] if name in na_dict else col.median()
                df[name] = col.fillna(filler)
                na_dict[name] = filler
    return na_dict


# In[ ]:


na_dict = fix_missing(df_raw, {})


# > **Lesson 3**: For continuous variables, fill all the missing values with mean() of the column. Creating a new column specifying if data was missing is really a good practice and can help you to achieve better scores.

# ### Putting it all together
# We are almost done with the preprocessing part. We have taken care of Missing values, time stamp and categorical variables. These are the most common problems that we come across in almost every dataset. So we can try writing a function which does all of it at one once. The function would look like:**

# In[ ]:


def numericalize(df, max_cat):
    """ Changes the column col from a categorical type to it's integer codes."""
    for name, col in df.items():
        if hasattr(col, 'cat') and (max_cat is None or len(col.cat.categories)>max_cat):
            df[name] = col.cat.codes+1
            
def get_sample(df,n):
    """ Gets a random sample of n rows from df, without replacement."""
#     idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[-n:].copy()


# In[ ]:


def process_df(df_raw,y_fld=None, subset=None, na_dict={}, max_cat=None,):
    if subset: df = get_sample(df_raw,subset)
    else: df = df_raw.copy()
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = df[y_fld].cat.codes
        y = df[y_fld].values
    df.drop(y_fld, axis=1, inplace=True)
    
    # Missing continuous values
    na_dict = fix_missing(df, na_dict)
    
    # Normalizing continuous variables
    means, stds = {}, {}
    for name,col in df.items():
        if is_numeric_dtype(col) and col.dtype not in ['bool', 'object']:
            means[name], stds[name] = col.mean(), col.std()
            df[name] = (col-means[name])/stds[name] 
    
    # categorical variables
    categorical = []
    for col in df.columns:
        if df[col].dtype == 'object' : categorical.append(col)  # pandas treat "str" as "object"
    for col in categorical: 
        df[col] = df[col].astype("category").cat.as_ordered()
        
    # converting categorical variables to integer codes.
    numericalize(df, max_cat) # features with cardinality more than "max_cat".
    
    df = pd.get_dummies(df, dummy_na=True) # one-hot encoding for features with cardinality lower than "max_cat".
    
    return df, y#, na_dict, means, stds


# In[ ]:


add_datepart(df_org, 'saledate')


# In[ ]:


df, y = process_df(df_org,'SalePrice')


# In[ ]:


df.head().T # final dataset, you can save it if you want!


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
m.fit(df,y)
m.score(df, y)


# In statistics, the coefficient of determination, denoted R2 or r2 and pronounced "R squared", is the proportion of the variance in the dependent variable that is predictable from the independent variable(s). [https://en.wikipedia.org/wiki/Coefficient_of_determination](https://en.wikipedia.org/wiki/Coefficient_of_determination)
# 
# Wow, an r^2 of 0.98 - that's great, right? Well, perhaps not...
# 
# Possibly the most important idea in machine learning is that of having separate training & validation data sets. As motivation, suppose you don't divide up your data, but instead use all of it. And suppose you have lots of parameters:
# 
# ![Underfitting and Overfitting](https://i.stack.imgur.com/t0zit.png)
# [Refer this link](https://datascience.stackexchange.com/questions/361/when-is-a-model-underfitted)
# 
# The error for the pictured data points is lowest for the model on the far right (the blue curve passes through the red points almost perfectly), yet it's not the best choice. Why is that? If you were to gather some new data points, they most likely would not be on that curve in the graph on the right, but would be closer to the curve in the middle graph.
# 
# This illustrates how using all our data can lead to overfitting. A validation set helps diagnose this problem.

# In[ ]:


def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 12000  # same as Kaggle's test set size
n_trn = len(df)-n_valid
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape


# ### Base Model
# 
# Let's try our model again, this time with separate training and validation sets

# In[ ]:


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# An r^2 in the high-80's isn't bad at all (and the RMSLE puts us around rank 100 of 470 on the Kaggle leaderboard), but we can see from the validation set score that we're over-fitting badly. 
# 
# > **Note:** Here, we are processing the complete data first and then splitting it into training and validation set i.e we have some how already seen (learnt some part of) the validation set. This is called **"Data Leakage"**. Its a fundamentally false parctice and should be avoided. You can learn more about it [here](https://www.coursera.org/lecture/python-machine-learning/data-leakage-ois3n). In this notebook we will not talk about it. 
# 
# Lets intuitionally try to understand the reasons for overfitting.
# 
# ### Being interactive
# Let's start by speeding things up. We will take a subset of our data to work with

# In[ ]:


df, y = process_df(df_org, 'SalePrice', subset=30000)

X_train, X_valid = split_vals(df, 20000)
y_train, y_valid = split_vals(y, 20000) 


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# ### Single tree

# In[ ]:


df, y = process_df(df_org,'SalePrice')

n_valid = 12000  # same as Kaggle's test set size
n_trn = len(df)-n_valid
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)


# In[ ]:


m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# Let's see what happens if we create a bigger tree.

# In[ ]:


m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# The training set result looks great! But the validation set is worse than our original model. This is why we need to use bagging of multiple trees to get more generalizable results.
# 
# ## Bagging
# ### Introduction

# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# We'll grab the predictions for each individual tree, and look at one example.

# In[ ]:


preds = np.stack([t.predict(X_valid) for t in m.estimators_])
preds[:,0], np.mean(preds[:,0]), y_valid[0]


# In[ ]:


preds.shape


# In[ ]:


plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)]);


# The shape of this curve suggests that adding more trees isn't going to help us much. Let's check. (Compare this to our original model on a sample)

# In[ ]:


m = RandomForestRegressor(n_estimators=20, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


m = RandomForestRegressor(n_estimators=40, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


m = RandomForestRegressor(n_estimators=80, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# As guessed, icreasing the number of estimators after a certain number will not help. A good range to look for is between 20-40 estimators.
# 
# ### Out-of-bag (OOB) score
# **Is our validation set worse than our training set because we're over-fitting, or because the validation set is for a different time period, or a bit of both?** The questions are really subtle so read them twice before you move ahead. With the existing information we've shown, we can't tell. However, random forests have a very clever trick called out-of-bag (OOB) error which can handle this (and more!)
# 
# The idea is to calculate error on the training set, but only include the trees in the calculation of a row's error where that row was not included in training that tree. This allows us to see whether the model is over-fitting, without needing a separate validation set.
# 
# This also has the benefit of allowing us to see whether our model generalizes, even if we only have a small amount of data so want to avoid separating some out to create a validation set.
# 
# This is as simple as adding one more parameter to our model constructor. We print the OOB error last in our print_score function below.
# 

# In[ ]:


## Also a BASELINE MODEL to which all the other models will be compared.
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# There's a huge difference between validation_score and oob_score. In this scenario, your model is performing better on oob_set, which is take directly from your training dataset. Indicating, validation_set is for a different time period. (say training_set has records for the month of "January" and validation_set has records for the month of "July"). So, more than a test for model's performance, oob_score is test for "how representative is your Validation_set".
# 
# You should always make sure that you have a good representative validation_set, because it's score is used as an indicator for our model's performance. So your goal should be, to have as little difference between oob_score and valid_score as possible.
# 
# The **huge difference between Train_score and valid_score**,  is an indicator that our model is over-fitting very badly.
# 
# ## Fighting Over fitting
# ### Using tree parameters
# We will use the last trained random forest as a baseline model to which all the other models will be compared.

# In[ ]:


def dectree_max_depth(tree):
    children_left = tree.children_left
    children_right = tree.children_right

    def walk(node_id):
        if (children_left[node_id] != children_right[node_id]):
            left_max = 1 + walk(children_left[node_id])
            right_max = 1 + walk(children_right[node_id])
            return max(left_max, right_max)
        else: # leaf
            return 1

    root_node_id = 0
    return walk(root_node_id)


# In[ ]:


dectree_max_depth(m.estimators_[0].tree_)


# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


dectree_max_depth(m.estimators_[0].tree_)


# #### 1. *min_samples_leaf*
# One way to reduce over-fitting is to grow our trees less deeply. We do this by specifying (with *min_samples_leaf*) that we require some minimum number of rows in every leaf node. This has two benefits:    
# - There are less decision rules for each leaf node; simpler models should generalize better.
# - The predictions are made by averaging more rows in the leaf node, resulting in less volatility.
# 
# #### 2. *max_features*
# We can also increase the amount of variation amongst the trees by using a different sample of columns for each split. We do this by specifying *max_features*, which is the proportion of features to randomly select from at each split.
# 
# 

# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# We can't compare our results directly with the Kaggle competition, since it used a different validation set (and we can no longer to submit to this competition) - but we can at least see that we're getting similar results to the top 100-ish based on the dataset we have.
# 
# The sklearn docs show an example of different max_features methods with increasing numbers of trees - as you see, using a subset of features on each split requires using more trees, but results in better models:
# 
# ![](https://scikit-learn.org/stable/_images/sphx_glr_plot_ensemble_oob_001.png)
# 
# ### End note
# This was a long notebook but I am glad that you followed it to the very end. You should be proud of yourself for doing it. We have briefly talked about the following:
# - How to extract features from timestamps
# - How to work with categorical features
# - How to work with continuous features
# - How to handle missing values in both continuous and categorical features
# - Need to have a validation set
# - OOB score and its importances
# - How we can fine tune parameters of random forest
# 
# In practice, random forest is the best model for most data because most data is not random. A random-forest decision tree works for almost every structured-data problem. You just have to tune it properly for the data in-hand.
# 
# We have [Part-2](https://www.kaggle.com/ankursingh12/lessons-learnt-from-fast-ai-lectures-part-2) of this notebook lined up. So, stay TUNED!!

# In[ ]:




