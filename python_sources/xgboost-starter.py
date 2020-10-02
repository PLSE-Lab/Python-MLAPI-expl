#!/usr/bin/env python
# coding: utf-8

# ## XGBoost script for begginer
# * Must install XGBoost - http://xgboost.readthedocs.io/en/latest/build.html
# * If using Anaconda here is another way to install - https://stackoverflow.com/questions/35139108/how-to-install-xgboost-in-anaconda-python-windows-platform   <--- This worked better for me
# * You can start with some data with missing values as there is built in handling of missing values.

# In[2]:


import numpy as np
import pandas as pd
import timeit   
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer


# #### Import Model from XGBoost
# * not sure about the depreciation warning.  I can't seem to get a newer version, and it doesn't seem to impact the model

# In[3]:


from xgboost import XGBClassifier


# #### Data Transformation
# * note, put data in a subfolder called input.  Current syntax is for MAC, for windows add one more .
#     example: '../input/train.csv' 

# In[6]:


df = pd.read_csv('../input/train.csv',low_memory=False)
print(df.shape)
df.head()


# #### Could Drop NaN's but XGBoost has Missing data handling.  
# * doesn't seem to work for free form text fields as it's trying to convert to float.  
# * Dropping all columns with "OTHERS", "Oth",  &"REC" in name 
# * Imputing -1 for NaN

# In[4]:


#uncomment the item below to drop NaNs.  Use instead of regex 
#df2 = df.dropna(axis=1)

df=df[df.columns.drop(list(df.filter(regex='OTHERS')))]
df=df[df.columns.drop(list(df.filter(regex='Oth')))]
df2=df[df.columns.drop(list(df.filter(regex='REC')))]

#Force numeric on remaining
df2=df2.apply(pd.to_numeric, errors='coerce')

#Fill NaN with -1
df2=df2.fillna(-1) 

print (df2.shape)


# In[5]:


# to get all columns names
df2.columns


# ### let's create training data
# * drop train_id and is_female  for X
# * take "is_female" as Y

# In[6]:


X = df2.drop(['train_id','is_female'], axis=1)
Y = df2.is_female


# ### Let's check Test data
# * Matching to training columns
# * update to numeric
# * Fill NaN with -1

# In[ ]:


test = pd.read_csv('../input/test.csv',low_memory=False)
print(test.shape)


# In[8]:


# keep those columns from training data X
test3 = test.reindex(columns=X.columns, fill_value=-1)

#Force numeric on remaining
test3=test3.apply(pd.to_numeric, errors='coerce')

#Fill NaN with -1
test3=test3.fillna(-1) 

test3.shape


# In[9]:


print (Y.shape, X.shape, test3.shape)
test3.head()


# ### start training
# * let's split the data to know how difficult this task is
# * Use Imputers to transform
# * Start Model

# In[10]:


train_X, test_X, train_Y, test_Y = train_test_split(X.as_matrix(), Y.as_matrix(), test_size=0.33)
print (train_X.shape, test_X.shape, train_Y.shape, test_Y.shape)


# In[11]:


# evaluate the effect of the number of threads
results = []
num_threads = [6,8,10]
for n in num_threads:
    start = timeit.default_timer()
    model = XGBClassifier(nthread=n, n_estimators=100)
    model.fit(train_X, train_Y)
    elapsed = timeit.default_timer() - start
    print(n, elapsed)
    results.append(elapsed)


# In[11]:


#Starter Model
#added start_time so I can show how much time this take to run
start_time = timeit.default_timer()

n_est = 100
learning_rate = 0.05
seed = 1234
my_model_t1 = XGBClassifier(n_estimators=n_est, learning_rate=learning_rate, seed=seed)
eval_set = [(test_X, test_Y)]
my_model_t1.fit(train_X, train_Y, verbose=False)

elapsed = timeit.default_timer() - start_time
print(elapsed)


# In[21]:


#added start_time so I can show how much time this take to run
start_time = timeit.default_timer()

#Tuned Model
my_model_t2 = XGBClassifier(
    learning_rate =0.01,
    n_estimators= 1500,
    max_depth=7,
    min_child_weight=3,
    gamma=0,
    subsample=.9,
    colsample_bytree=0.65,
    reg_alpha=0.00001,
    reg_lambda=1.59,
    objective= 'binary:logistic',
    tree_method = "hist",
    scale_pos_weight=1,
    seed=27)
eval_set = [(test_X, test_Y)]
my_model_t2.fit(train_X, train_Y, verbose=1)

elapsed = timeit.default_timer() - start_time
print(elapsed)


# elapsed time: 1114.6977803190002

# In[13]:


#added start_time so I can show how much time this take to run
start_time = timeit.default_timer()

#Tuned Model
my_model_t3 = XGBClassifier(
 learning_rate =0.2,
 n_estimators=1311,
 max_depth=8,
 min_child_weight=4,
 gamma=0.4,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
eval_set = [(test_X, test_Y)]
my_model_t3.fit(train_X, train_Y, verbose=True)

elapsed = timeit.default_timer() - start_time
print(elapsed)


# ### Run prediction and get accuracy 

# In[22]:


pred_Y = my_model_t2.predict(test_X)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_Y, pred_Y)
print (cm)


# ### However, to get AUC, we need probablity score

# In[23]:


y_prob1 = my_model_t1.predict_proba(test_X)
y_prob2 = my_model_t2.predict_proba(test_X)
y_prob3 = my_model_t3.predict_proba(test_X)


# In[24]:


from sklearn import metrics


# In[25]:


# let check which score do we need,   we need column 1  not column 0
y_prob1, y_prob2, y_prob3,


# In[26]:


print("my_model_t1 : {}".format(metrics.roc_auc_score(test_Y, y_prob1[:,1])))
print("my_model_t2 : {}".format(metrics.roc_auc_score(test_Y, y_prob2[:,1])))
print("my_model_t3 : {}".format(metrics.roc_auc_score(test_Y, y_prob3[:,1])))


# ### WHAT?! .969
# 
# * Uses 2/3 of data as training
# * 1/3 as validation
# * now we can use all data as training to predict test set for submission

# In[ ]:


my_model2 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=2133,
 max_depth=8,
 min_child_weight=4,
 gamma=4,
 subsample=1.0,
 colsample_bytree=0.65,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
my_model2.fit(X, Y)
y_submit = my_model2.predict_proba(test3)[:,1]
test['is_female'] = y_submit
ans = test[['test_id','is_female']]
ans.to_csv('./input/submit.csv', index=None)


# In[ ]:


my_model3 = XGBClassifier(
    learning_rate =0.01,
    n_estimators=1822,
    max_depth=7,
    min_child_weight=3,
    gamma=0,
    subsample=.9,
    colsample_bytree=0.65,
    reg_alpha=0.00001,
    reg_lambda=1.59,
    objective= 'binary:logistic',
    tree_method = "hist",
    scale_pos_weight=1,
    seed=27)
my_model3.fit(X, Y)
y_submit = my_model3.predict_proba(test3)[:,1]
test['is_female'] = y_submit
ans = test[['test_id','is_female']]
ans.to_csv('./input/submit.csv', index=None)

