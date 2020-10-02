#!/usr/bin/env python
# coding: utf-8

# An important part when applying Machine Learning is picking an approriate training and test set. In particular they should have the same characteristics. In this competition this could mean that the same kinds of customers and merchants appear in both train and set set. If that is not the case then we'd do a poor job at predicting outside the training set. We would be training the model on a fundamentally different kind of data than the one against which we will be testing it. We wouldn't want to train the model on young professionals and then test our predictions on pensioners. 
# 
# What I want to do in this kernel is to test this assumption by using a simple RandomForestClassifier to check whether there is something particular about the train and test set. If the two data sets are similar then I would expect the classification prediction to be random, i.e. the ROC_AUC score would be around 0.5. 

# # Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
import re


# # Load Data

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.info()


# In[ ]:


test.info()


# # Clean Data

# Let's delete the single row with the missing date. It doesn't matter for our purposes. When submitting tp the Kaggle challenge then obviously you can't delete rows from the test set.

# In[ ]:


test.dropna(axis=0,how='any', inplace=True)


# Let's drop 'target' from train as I won't be needing it. Instead I will set up a new target variable, which I call 'train'. 'train' indicates whether or not the row belongs to the training set.

# In[ ]:


# Drop target
train.drop('target', axis=1, inplace=True)

# Create the new target
train['train'] = 1
test['train'] = 0


# Next, I concatenate the two data sets.

# In[ ]:


train_test = pd.concat([train, test], axis=0)


# What does the ratio of train to test set look like? Turns out that 62% of the combined data set is made up of train data.

# In[ ]:


print(np.mean(train_test['train']))


# ## Feature Enginerring

# The two features 'first_active_month' and 'card_id' are not in a usable format for our classification yet. So, I will have to make some adjustments. 
# 
# 1. I first convert 'first_active_month' to datetime and then extract each subcomponent from it such as year, month etc.
# 2. I convert 'card_id' to a categorical variable so that it becomes a number, which I can actually use.

# In[ ]:


# Convert to datetime
train_test['first_active_month'] = pd.to_datetime(train_test['first_active_month'])

# Set up the function to extract the subcomponents of date
def add_datepart(df, fldname, drop=True, time=False):
    fld = df[fldname]
    #if not np.issubdtype(fld.dtype, np.datetime64):
    #    df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Hour', 'Minute',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    #df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)
        
# Extract subcomponent of date
add_datepart(train_test, 'first_active_month')


# In[ ]:


# Convert 'card_id' to categorical, i.e. number
train_test['card_id'] = train_test['card_id'].astype('category').cat.as_ordered()
train_test['card_id'] = train_test['card_id'].cat.codes


# # Modelling

# ## Prepare Data for Modelling

# Here I split the data into dependent and independent variables. There is no need to split this into a new train and validation set as cross_val_score automatically does stratified K-Folds. In other words, the ratio of 62% of training to test data will be preserved when doing the cross-validation. 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# In[ ]:


# Split up the dependent and independent variables
X = train_test.drop('train', axis=1)
y= train_test['train']


# ## Run the Model

# I will run a simple RandomForestClassifier using cross-validation.

# In[ ]:


rfc = RandomForestClassifier(n_estimators=10,
                            random_state=1)


# In[ ]:


cv_results = cross_val_score(rfc, X, y, cv=5, scoring='roc_auc')


# In[ ]:


print(cv_results)
print(np.mean(cv_results))


# # Evaluation

# And indeed the distributions/behaviours of train and test set are very similar. The ROC_AUC score is 0.5 indicating that the Random Forest is as good as randomly guessing whether a row belongs to train or test.
# 
# The results might look different if we joined all tables together and do the same analysis. But doing this quick-and-dirty exercise goes a long way of sense checking whether or not train and test data are similar. We wouldn't want to train the model on youn professionals and then test our predictions on pensioners. We would most likely get a pretty bad prediction.
# 
# **I hope you found this useful and I would like to hear from whether you use a similar approach or whether there are bits that I can improve.**
