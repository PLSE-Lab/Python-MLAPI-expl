#!/usr/bin/env python
# coding: utf-8

# # HR_Analytics-AV Hackathon

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Importing Libraries

# In[ ]:


import os
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import math as m
from scipy import stats
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv('/kaggle/input/avhranalytics/train_jqd04QH.csv')
test = pd.read_csv('/kaggle/input/avhranalytics/test_KaymcHn.csv')


# In[ ]:


train.head()


# In[ ]:


train.shape,test.shape


# In[ ]:


train.apply(lambda x: (len(x.unique()))) 


# In[ ]:


test.apply(lambda x: (len(x.unique()))) 


# Train and Test having the same number of unique values, we can combine the dataset and use for further Analysis

# In[ ]:


combine = train.append(test,sort=False)
combine.shape


# In[ ]:


combine.isnull().sum()


# > ### **EDA - Target Exploration**

# In[ ]:


train['target'].value_counts(normalize=True)


# In[ ]:


sns.countplot(train['target'])


# ### Univariate Analysis

# In[ ]:


plt.figure(figsize=(24, 6))
plt.subplot(121)
sns.countplot(combine['company_size'],order = combine['company_size'].value_counts(dropna=False).index)
plt.subplot(122)
sns.countplot(combine['company_type'],order = combine['company_type'].value_counts(dropna=False).index)


# The company_size Missing Value is greater than any of sum of other values, it is good to consider as a seperate variable

# In[ ]:


combine['company_size'].fillna('unknown', inplace=True)


# The Company_Type more 30 % of values are Missing Value is greater, it is good to consider as a seperate variable

# In[ ]:


combine['company_type'].fillna('unknown', inplace=True)


# In[ ]:


plt.figure(figsize=(20, 6))
plt.subplot(121)
sns.countplot(combine['gender'],order = combine['gender'].value_counts(dropna=False).index)
plt.subplot(122)
sns.countplot(combine['relevent_experience'],order = combine['relevent_experience'].value_counts(dropna=False).index)


# As the data contains Almost 90% of males, Nan Values can be replaces as Males, else we can use Not revelaed(I have tried but there is no change in Model Performance)

# In[ ]:


combine['gender'].fillna('Male', inplace=True)


# In[ ]:


combine["gender"] = combine["gender"].map({'Male':2,  'Female':1, 'Other':0})


# In[ ]:


plt.figure(figsize=(22, 6))
plt.subplot(121)
sns.countplot(combine['last_new_job'],order = combine['last_new_job'].value_counts(dropna=False).index)
plt.subplot(122)
sns.countplot(combine['experience'],order = combine['experience'].value_counts(dropna=False).index)


# In[ ]:


combine['last_new_job'].fillna('1', inplace=True) #using Mode Option for fill Nan Values as there are very less null values


# In[ ]:


combine['last_new_job'].replace('>4','6', inplace=True)
combine['last_new_job'].replace('never','0' ,inplace=True)
combine['last_new_job']=combine['last_new_job'].astype(int)


# In[ ]:


combine['experience'].fillna('>20', inplace=True) #using Mode Option for fill Nan Values as there are very less null values


# In[ ]:


combine['experience'].replace('>20','25', inplace=True)
combine['experience'].replace('<1','0' ,inplace=True)
combine['experience']=combine['last_new_job'].astype(int)


# In[ ]:


plt.figure(figsize=(22, 6))
plt.subplot(131)
sns.countplot(combine['education_level'],order = combine['education_level'].value_counts(dropna=False).index)
plt.subplot(132)
sns.countplot(combine['enrolled_university'],order = combine['enrolled_university'].value_counts(dropna=False).index)
plt.subplot(133)
sns.countplot(combine['major_discipline'],order = combine['major_discipline'].value_counts(dropna=False).index)


# ### Bivariate Analysis

# In[ ]:


combine['education_level'].value_counts(dropna=False)


# In[ ]:


combine['education_level'].fillna(train['education_level'].mode()[0], inplace=True) #Missing Value is less than 10% of Mode Value Itself


# In[ ]:


plt.figure(figsize=(22, 6))
city_tier_counts = (combine.groupby(['target'])['education_level'].value_counts(dropna=False,normalize=True).rename('percentage').mul(100).reset_index().sort_values('target'))
sns.barplot(x="education_level", y="percentage", hue="target", data=city_tier_counts)


# In[ ]:


combine["education_level"] = combine["education_level"].map({'Graduate':0, 'Masters':1, 'High School':2, 'Phd':3, 'Primary School':4})


# In[ ]:


combine.enrolled_university.value_counts(dropna=False)


# In[ ]:


combine['enrolled_university'].fillna(train['enrolled_university'].mode()[0], inplace=True) #Missing Value is less than 3% of Mode Value Itself


# In[ ]:


plt.figure(figsize=(22, 6))
city_tier_counts = (combine.groupby(['target'])['enrolled_university'].value_counts(dropna=False,normalize=True).rename('percentage').mul(100).reset_index().sort_values('target'))
sns.barplot(x="enrolled_university", y="percentage", hue="target", data=city_tier_counts)


# In[ ]:


combine["enrolled_university"] = combine["enrolled_university"].map({'no_enrollment':1, 'Full time course':4, 'Part time course':2})


# In[ ]:


combine.major_discipline.value_counts(dropna=False)


# Missing Value is nearly 10% of data, so we can keep Nan values as Unknow,  but here the other datas are less than 3% of Mode of major_discipline, so I am replacing Nan as Mode Values
# 

# In[ ]:


combine['major_discipline'].fillna(train['major_discipline'].mode()[0], inplace=True)


# In[ ]:


plt.figure(figsize=(20, 6))
city_tier_counts = (combine.groupby(['target'])['major_discipline'].value_counts(dropna=False,normalize=True).rename('percentage').mul(100).reset_index().sort_values('target'))
sns.barplot(x="major_discipline", y="percentage", hue="target", data=city_tier_counts)


# In[ ]:


combine.isnull().sum()


# ### Label Encoding

# In[ ]:


cat_col = combine.dtypes.loc[combine.dtypes=='object'].index
categorical_variables=cat_col.tolist()
categorical_variables


# In[ ]:


from sklearn import metrics, preprocessing, model_selection
for col in categorical_variables:
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(combine[col].values.astype('str')))
    combine[col] = lbl.transform(list(combine[col].values.astype('str')))


# In[ ]:


display(combine.columns),train.shape


# In[ ]:


train_features = combine.drop(['enrollee_id', 'target'], axis = 1)[:18359]
target = combine['target'][:18359]
test_features = combine.drop(['enrollee_id','target'], axis = 1)[18359:]


# In[ ]:


train_features.shape,target.shape,test_features.shape


# In[ ]:


from sklearn import metrics, preprocessing, model_selection
import lightgbm as lgb


# In[ ]:


train_X=train_features
train_y=target
test_X=test_features


# In[ ]:


def runLGB(train_X, train_y, test_X, test_y=None, test_X2=None, dep=8, seed=0, data_leaf=200):
    params = {}
    params["objective"] = "binary"
    params['metric'] = 'auc'
    params["max_depth"] = dep
    params["num_leaves"] = 31
    params["min_data_in_leaf"] = data_leaf
    params["learning_rate"] = 0.01
    params["bagging_fraction"] = 0.9
    params["feature_fraction"] = 0.5
    params["feature_fraction_seed"] = seed
    params["bagging_freq"] = 1
    params["bagging_seed"] = seed
    params["lambda_l2"] =5
    params["lambda_l1"] = 5
    params["verbosity"] = -1
    num_rounds = 25000

    plst = list(params.items())
    lgtrain = lgb.Dataset(train_X, label=train_y)

    if test_y is not None:
        lgtest = lgb.Dataset(test_X, label=test_y)
        model = lgb.train(params, lgtrain, num_rounds, valid_sets=[lgtest], early_stopping_rounds=200, verbose_eval=500)
    else:
        lgtest = lgb.DMatrix(test_X)
        model = lgb.train(params, lgtrain, num_rounds)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)

    loss = 0
    if test_y is not None:
        loss = metrics.roc_auc_score(test_y, pred_test_y)
        print(loss)
        return model, loss, pred_test_y, pred_test_y2
    else:
        return model, loss, pred_test_y, pred_test_y2


# In[ ]:


print("Building model..")
cv_scores = []
pred_test_full = 0
pred_train = np.zeros(train_X.shape[0])
n_splits = 5
#kf = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=7988)
gkf = model_selection.GroupKFold(n_splits=n_splits)
model_name = "lgb"
for dev_index, val_index in gkf.split(train_X, combine['target'][:18359].values, combine['enrollee_id'][:18359].values):
    dev_X, val_X = train_X.iloc[dev_index,:], train_X.iloc[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val = 0
    pred_test = 0
    n_models = 0.

    model, loss, pred_v, pred_t = runLGB(dev_X, dev_y, val_X, val_y, test_X, dep=8, seed=2019)
    pred_val += pred_v
    pred_test += pred_t
    n_models += 1
    
    model, loss, pred_v, pred_t = runLGB(dev_X, dev_y, val_X, val_y, test_X, dep=7, data_leaf=100, seed=9873)
    pred_val += pred_v
    pred_test += pred_t
    n_models += 1
    
    model, loss, pred_v, pred_t = runLGB(dev_X, dev_y, val_X, val_y, test_X, dep=9, data_leaf=150, seed=4568)
    pred_val += pred_v
    pred_test += pred_t
    n_models += 1
    
    pred_val /= n_models
    pred_test /= n_models
    
    loss = metrics.roc_auc_score(val_y, pred_val)
        
    pred_train[val_index] = pred_val
    pred_test_full += pred_test / n_splits
    cv_scores.append(loss)
#     break
print(np.mean(cv_scores))


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
lgb.plot_importance(model, max_num_features=100, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()


# In[ ]:


sample = pd.read_csv('/kaggle/input/avhranalytics/sample_submission_sxfcbdx.csv')
sample["target"] = pred_test_full
sample.to_csv("Solution.csv", index=False)

