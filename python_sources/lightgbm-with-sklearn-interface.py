#!/usr/bin/env python
# coding: utf-8

# This is the 3rd stop of the Kaggle Challenge

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore")


# #  Import and cleaning

# In[ ]:


train=pd.read_csv('../input/act_train.csv',parse_dates=['date'])
test=pd.read_csv('../input/act_test.csv',parse_dates=['date'])
people=pd.read_csv('../input/people.csv',parse_dates=['date'])


# In[ ]:


train.head(1)


# In[ ]:


test.head(1)


# In[ ]:


people.head(1)


# Merge train with people and test with people.

# In[ ]:


train=pd.merge(train,people,on='people_id')
test=pd.merge(test,people,on='people_id')


# In[ ]:


train.info()


# In[ ]:


test.info()


# # EDA

# ## Check if it's a class imblance problem

# In[ ]:


train.outcome.plot(kind='hist')
plt.show()


# According to the result, we can use "accuracy" as the measurement.

# ## Check null values

# In[ ]:


def checkMissing(df):
    column_names=[]
    null_count=[]
    for i in df.columns:
        if df[i].isnull().sum()!=0:
            column_names.append(i)
            null_count.append(df[i].isnull().sum())
    if len(null_count)==0:
        print('There is no missing values!')
    else:
        plt.figure(figsize=(8,4))
        sns.barplot(x=null_count,y=column_names,color='C0')
        plt.show()
checkMissing(train)


# ## The char_38
# 
# The char_38 is the only numeric feature except outcome. Let's look into it.

# In[ ]:


train.char_38.nunique()


# In[ ]:


plt.hist(train[train['outcome']==1]['char_38'],color='C1',alpha=0.7)
plt.hist(train[train['outcome']==0]['char_38'],color='C0',alpha=0.7)
plt.show()
print('Number of 0 value:',train[train .char_38==0]['char_38'].count())


# It's obvious that the char_38 is larger when the outcome equales to 1. And there are a lot of 0 values.

# ## Categorical Features

# In[ ]:


train.describe(include=['object']).transpose()


# group_1 has a large amount of unique numbers and is the only "group" category

# In[ ]:


len(set(test['group_1'])-set(train['group_1']))


# There are 4325 group_1s that are in the test but not in the train. It may create null value when we do feature engineering later. 

# ## Checking date

# In[ ]:


for d in ['date_x', 'date_y']:
    print('Start of ' + d + ': ' + str(train[d].min().date()))
    print('  End of ' + d + ': ' + str(train[d].max().date()))
    print('Range of ' + d + ': ' + str(train[d].max() - train[d].min()) + '\n')


# ## Missing Value Imputation

# In[ ]:


# Impute the missing values with type 0
for i in train.columns:
    if np.dtype(train[i])==np.dtype('object'):
        train[i].fillna('type 0',inplace=True)


# In[ ]:


for i in test.columns:
    if np.dtype(test[i])==np.dtype('object'):
        test[i].fillna('type 0',inplace=True)


# In[ ]:


checkMissing(train)
checkMissing(test)


# # Feature Engineering

# ## Mean and median outcome group by group_1

# In[ ]:


# Mean and Median outcome group by group_1 for train
outcomeMeanGroupbyGroup_1=train.groupby(['group_1'])['outcome'].mean().to_frame().reset_index()
outcomeMedianGroupbyGroup_1=train.groupby(['group_1'])['outcome'].median().to_frame().reset_index()
dict_outcomeMeanGroupbyGroup_1=dict(zip(outcomeMeanGroupbyGroup_1['group_1'],outcomeMeanGroupbyGroup_1['outcome']))
dict_outcomeMedianGroupbyGroup_1=dict(zip(outcomeMedianGroupbyGroup_1['group_1'],outcomeMedianGroupbyGroup_1['outcome']))
train['outcomeMeanGroupbyGroup_1']=train['group_1'].map(lambda x: dict_outcomeMeanGroupbyGroup_1.get(x))
train['outcomeMedianGroupbyGroup_1']=train['group_1'].map(lambda x: dict_outcomeMedianGroupbyGroup_1.get(x))

# Mean and Median outcome group by group_1 for test
test['outcomeMeanGroupbyGroup_1']=test['group_1'].map(lambda x: dict_outcomeMeanGroupbyGroup_1.get(x))
test['outcomeMedianGroupbyGroup_1']=test['group_1'].map(lambda x: dict_outcomeMedianGroupbyGroup_1.get(x))


# In[ ]:


checkMissing(test)


# In[ ]:


# Impute missing value for test
test['outcomeMeanGroupbyGroup_1']=test.groupby(['activity_category'])['outcomeMeanGroupbyGroup_1'].transform(lambda x: x.fillna(x.mean()))
test['outcomeMedianGroupbyGroup_1']=test.groupby(['activity_category'])['outcomeMedianGroupbyGroup_1'].transform(lambda x: x.fillna(x.mean()))


# In[ ]:


checkMissing(test)


# In[ ]:


def featureEngineering(df):
    # feature engineering for dates
    listDate=['year', 'month', 'week', 'day', 'dayofweek', 'dayofyear','is_month_end', 'is_month_start', 'is_quarter_end', 'is_quarter_start', 'is_year_end', 'is_year_start']
    for n in listDate:
        df[n.upper()]=df['date_x'].map(lambda x: getattr(x,n))
        df[n.upper()]=df['date_y'].map(lambda x: getattr(x,n))
    
    # Extract numbers from cateforical data and convert boolean to int
    for i in df.columns:
        if np.dtype(df[i])==np.dtype('object') and i not in ['activity_id','people_id']:
            df[i]=df[i].map(lambda x:int(x.split(' ')[1]))
        elif np.dtype(df[i])==np.dtype('bool'):
            df[i]=df[i].map(lambda x:int(x))

    return df


# In[ ]:


# Feature Engineering for train and test
train=featureEngineering(train)
test=featureEngineering(test)


# In[ ]:


# Check feature number of train and test after feature engineering
missingfeatures=list(set(train.columns.tolist())-set(test.columns.tolist()))
missingfeatures.remove('outcome')
print(missingfeatures)

Train and test have same number of features
# In[ ]:


train.head()


# # Modeling

# ## Data Pre-processing

# In[ ]:


X_train=train.drop(['outcome','date_x','date_y','activity_id','people_id'],axis=1).copy()
y_train=train['outcome'].copy()
X_test=test.drop(['date_x','date_y','activity_id','people_id'],axis=1).copy()


# In[ ]:


X_train.head()


# In[ ]:


# To save some time. I used reduced dataset.


# In[ ]:


# X_train_demo=X_train.iloc[:50000,:].copy()
# y_train_demo=y_train.iloc[:50000].copy()


# In[ ]:


from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score


# ## Model tunning

# In[ ]:


# Cross validate model with Kfold cross validation
kfold=StratifiedKFold(n_splits=2)


# In[ ]:


# I've narrow down the range of the parameters after some tryings in order to save some time.
lgbm = LGBMClassifier(random_state=8)

lgbm_param_grid = {'num_leaves' : [2,3],
                'learning_rate' :   [0.004,0.005,0.006,0.007],
                'n_estimators': [50,100]}

lgbm = GridSearchCV(lgbm,param_grid = lgbm_param_grid, cv=kfold, scoring="accuracy",n_jobs=2, verbose = 1)

lgbm.fit(X_train,y_train)
# How to use the output of GridSearch? Please see https://datascience.stackexchange.com/questions/21877/how-to-use-the-output-of-gridsearch
# Best score
print(lgbm.best_score_)
print(lgbm.best_params_)


# ## Prediction

# In[ ]:


results = pd.DataFrame({ 'activity_id' : test['activity_id'].values, 
                       'outcome': lgbm.predict_proba(X_test)[:,1] })


# In[ ]:


results.to_csv('red_hat_LightGBM.csv',index=False)

