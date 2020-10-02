#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Reading the given dataset

# In[ ]:


train_df = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/train.csv')
test_df = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/test.csv')
sample_sub_df = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/sample_submission.csv')


# ## Highlights of the dataset
# 
# <pre>
# Accident_ID:              	unique id assigned to each row
# Accident_Type_Code:     	  the type of accident (factor, not numeric)
# Cabin_Temperature:      	  the last recorded temperature before the incident, measured in degrees fahrenheit
# Turbulence_In_gforces:	    the recorded/estimated turbulence experienced during the accident
# Control_Metric:               an estimation of how much control the pilot had during the incident given the factors at play
# Total_Safety_Complaints: 	 number of complaints from mechanics prior to the accident
# Days_Since_Inspection:  	  how long the plane went without inspection before the incident
# Safety_Score:           	  a measure of how safe the plane was deemed to be
# Severity:	                 a description (4 level factor) on the severity of the crash [Target]
# </pre>

# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


sample_sub_df.head()


# In[ ]:


print(f'Shape of training data: {train_df.shape}')
print(f'Shape of testing data: {test_df.shape}')


# ## Analysis of missing values (NaNs)

# In[ ]:


train_df.isna().sum()


# In[ ]:


test_df.isna().sum()


# So there are no missing values in the dataset.

# ## Exploratory Data Analysis

# In[ ]:


X_train = train_df.drop(['Severity', 'Accident_ID'], axis=1)
Y_train = train_df['Severity']


# In[ ]:


Y_train.unique()


# So there are 4 classes of accidents (out target variable). Let's map those classes to integers

# In[ ]:


class_map = {
    'Minor_Damage_And_Injuries': 0,
    'Significant_Damage_And_Fatalities': 1,
    'Significant_Damage_And_Serious_Injuries': 2,
    'Highly_Fatal_And_Damaging': 3
}
inverse_class_map = {
    0: 'Minor_Damage_And_Injuries',
    1: 'Significant_Damage_And_Fatalities',
    2: 'Significant_Damage_And_Serious_Injuries',
    3: 'Highly_Fatal_And_Damaging'
}


# In[ ]:


Y_train = Y_train.map(class_map).astype(np.uint8)


# ### 1. Distribution of Target Variable 

# In[ ]:


plt.figure(figsize=(13,8))
ax = sns.barplot(np.vectorize(inverse_class_map.get)(pd.unique(Y_train)), Y_train.value_counts().sort_index())
ax.set(xlabel='Accident Severity', ylabel='# of records', title='Meter type vs. # of records')
ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")
plt.show()


# ### 2. Distribution of safety score

# In[ ]:


plt.figure(figsize=(13,8))
sns.distplot(X_train['Safety_Score'], kde=False)
plt.show()


# ### 3. Distribution of days till Last inspection

# In[ ]:


plt.figure(figsize=(13,8))
sns.distplot(X_train['Days_Since_Inspection'], kde=False)
plt.show()


# ### 3. Distribution of total safety complaints

# In[ ]:


plt.figure(figsize=(13,8))
sns.distplot(X_train['Total_Safety_Complaints'], kde=False)
plt.show()


# ### 4. Distribution of control metric

# In[ ]:


plt.figure(figsize=(13,8))
sns.distplot(X_train['Control_Metric'], kde=False)
plt.show()


# ### 5. Distribution of Turbulence

# In[ ]:


plt.figure(figsize=(13,8))
sns.distplot(X_train['Turbulence_In_gforces'], kde=False)
plt.show()


# ### 6. Distribution of Cabin Temperature (deg. F)

# In[ ]:


plt.figure(figsize=(13,8))
sns.distplot(X_train['Cabin_Temperature'], kde=False)
plt.show()


# ### 7. Distribution of Max Elevation

# In[ ]:


plt.figure(figsize=(13,8))
sns.distplot(X_train['Max_Elevation'], kde=False)
plt.show()


# ### 8. Distribution of number of violations

# In[ ]:


plt.figure(figsize=(13,8))
sns.distplot(X_train['Violations'], kde=False)
plt.show()


# ### 9. Distribution of adverse weather metric

# In[ ]:


plt.figure(figsize=(13,8))
sns.distplot(X_train['Adverse_Weather_Metric'], kde=False)
plt.show()


# In[ ]:


X_train['Total_Safety_Complaints'] = np.power(1.2, X_train['Total_Safety_Complaints'])
X_train['Days_Since_Inspection'] = np.power(1.2, X_train['Days_Since_Inspection'])
X_train['Safety_Score'] = np.power(1.2, X_train['Safety_Score'])


# In[ ]:


categoricals = ['Accident_Type_Code']


# In[ ]:


params = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class': 4,
            'metric': {'multi_logloss'},
            'subsample': 0.4,
            'subsample_freq': 1,
            'learning_rate': 0.2,
            'num_iterations': 2000,
            'num_leaves': 100,
            'feature_fraction': 0.8,
            'lambda_l1': 1,
            'lambda_l2': 1
            }

folds = 10
seed = 666

kf = StratifiedKFold(n_splits=folds, shuffle=False, random_state=seed)

# Score: 84.69896
models = []
for train_index, val_index in kf.split(X_train, Y_train):
    train_X = X_train.iloc[train_index]
    val_X = X_train.iloc[val_index]
    train_y = Y_train.iloc[train_index]
    val_y = Y_train.iloc[val_index]
    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categoricals)
    lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categoricals)
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=500,
                valid_sets=(lgb_train, lgb_eval),
                early_stopping_rounds=100,
                verbose_eval = 100)
    models.append(gbm)


# In[ ]:


test_df['Total_Safety_Complaints'] = np.power(1.2, test_df['Total_Safety_Complaints'])
test_df['Days_Since_Inspection'] = np.power(1.2, test_df['Days_Since_Inspection'])
test_df['Safety_Score'] = np.power(1.2, test_df['Safety_Score'])


# In[ ]:


preds = np.argmax(sum([model.predict(test_df.drop(['Accident_ID'], axis=1)) for model in models])/folds, axis=1)


# In[ ]:


submission = pd.DataFrame([test_df['Accident_ID'], np.vectorize(inverse_class_map.get)(preds)], index=['Accident_ID', 'Severity']).T
submission.to_csv('submission.csv', index=False)
submission.head()


# In[ ]:


from IPython.display import FileLink, FileLinks

FileLink('submission.csv')


# In[ ]:




