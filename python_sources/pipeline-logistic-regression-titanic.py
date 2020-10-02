#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


import re
from scipy.stats import iqr, zscore,norm

# to handle datasets
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
get_ipython().run_line_magic('matplotlib', 'inline')


# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import StandardScaler

# to build the models
from sklearn.linear_model import LogisticRegression

# to evaluate the models
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import log_loss

# to persist the model and the scaler
import joblib

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)


# In[ ]:


train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


# Display all dataset samples and info
display(train_df.head())
display(test_df.tail())
display(train_df.info())
display(test_df.info())


# In[ ]:


# Converting Object data type to String for regex expression to be used
train_df['Cabin']= train_df['Cabin'].astype('str')
# Convert pclass and sibsp into object type since it is not a continuous numerical value
train_df['Pclass']= train_df['Pclass'].astype('object')
train_df['SibSp']= train_df['SibSp'].astype('object')
train_df['Survived']= train_df['Survived'].astype('Int64')
train_df['Parch']= train_df['Parch'].astype('object')
train_df['Survived']= train_df['Survived'].astype('object')


# In[ ]:


# Converting Object data type to String for regex expression to be used
test_df['Cabin']= test_df['Cabin'].astype('str')
# Convert pclass and sibsp into object type since it is not a continuous numerical value
test_df['Pclass']= test_df['Pclass'].astype('object')
test_df['SibSp']= test_df['SibSp'].astype('object')
test_df['Parch']= test_df['Parch'].astype('object')


# In[ ]:


# replace interrogation marks by NaN values

train_df = train_df.replace('?', np.nan)
test_df = test_df.replace('?', np.nan)


# In[ ]:


# retain only the first cabin if more than
# 1 are available per passenger

def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return np.nan
    
train_df['Cabin'] = train_df['Cabin'].apply(get_first_cabin)
test_df['Cabin'] = test_df['Cabin'].apply(get_first_cabin)


# In[ ]:


# extracts the title (Mr, Ms, etc) from the name variable

def get_title(passenger):
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'
    
train_df['title'] = train_df['Name'].apply(get_title)
test_df['title'] = test_df['Name'].apply(get_title)


# In[ ]:


train_df.drop(labels=['Name','Ticket'], axis=1, inplace=True)
test_df.drop(labels=['Name','Ticket'], axis=1, inplace=True)


# In[ ]:


def varss(data):
    vars_num = list(data.select_dtypes([np.number]).columns)
    vars_cat = list(data.select_dtypes(include=['object', 'category']).columns)
    return [vars_num, vars_cat]
    

var = varss(train_df)
vars_num = (var[0])
vars_cat = (var[1])
print('Number of numerical variables: {}'.format(len(vars_num)))
print('Number of categorical variables: {}'.format(len(vars_cat)))


# In[ ]:


vars_cat.remove('Survived')
vars_num.remove('PassengerId')


# In[ ]:


train_df['Cabin']= train_df['Cabin'].apply(lambda x: re.split('(\d)', x)[0])
test_df['Cabin']= test_df['Cabin'].apply(lambda x: re.split('(\d)', x)[0])


# In[ ]:


# Find missing values in train dataset

#missing data => Find total rows having missing values and calculate the percentage of missing values for each field.
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:


# Find missing values in test dataset

#missing data => Find total rows having missing values and calculate the percentage of missing values for each field.
total = test_df.isnull().sum().sort_values(ascending=False)
percent = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:


for var in vars_cat:
        plt.figure()
        g = sns.FacetGrid(train_df, col='Survived')
        g.map(plt.hist, var)


# In[ ]:


for var in vars_num:
        plt.figure()
        g = sns.FacetGrid(train_df, col='Survived')
        g.map(plt.hist, var)


# ### Replace Missing Value for Numerical Variables in Train dataset

# In[ ]:


X_train = train_df.drop("Survived",axis=1)


# In[ ]:


# make a list with the numerical variables that contain missing values
vars_with_na = [
    var for var in X_train.columns
    if X_train[var].isnull().sum() > 0 and X_train[var].dtypes != 'O']

# print percentage of missing values per variable
X_train[vars_with_na].isnull().mean()


# In[ ]:


# replace engineer missing values as we described above

for var in vars_with_na:

    # calculate the mode using the train set
    mode_val = X_train[var].mode()[0]

    # add binary missing indicator (in train and test)
    #X_train[var+'_na'] = np.where(X_train[var].isnull(), 1, 0)
    #X_test[var+'_na'] = np.where(X_test[var].isnull(), 1, 0)

    # replace missing values by the mode
    # (in train and test)
    X_train[var] = X_train[var].fillna(mode_val)
    

# check that we have no more missing values in the engineered variables
X_train[vars_with_na].isnull().sum()


# ### Replace Missing Value for Numerical Variables in Test dataset

# In[ ]:


X_test = test_df


# In[ ]:


# make a list with the numerical variables that contain missing values
vars_with_na = [
    var for var in X_test.columns
    if X_test[var].isnull().sum() > 0 and X_test[var].dtypes != 'O']

# print percentage of missing values per variable
X_test[vars_with_na].isnull().mean()


# In[ ]:


# replace engineer missing values as we described above

for var in vars_with_na:

    # calculate the mode using the train set
    mode_val = X_test[var].mode()[0]

    # add binary missing indicator (in train and test)
    #X_train[var+'_na'] = np.where(X_train[var].isnull(), 1, 0)
    #X_test[var+'_na'] = np.where(X_test[var].isnull(), 1, 0)

    # replace missing values by the mode
    # (in train and test)
    X_test[var] = X_test[var].fillna(mode_val)
    

# check that we have no more missing values in the engineered variables
X_test[vars_with_na].isnull().sum()


# In[ ]:


scaler = StandardScaler()
X_train[vars_num] = scaler.fit_transform(X_train[vars_num])
X_test[vars_num] = scaler.fit_transform(X_test[vars_num])


# In[ ]:


X_train = pd.get_dummies(X_train, drop_first=True, dummy_na=False,prefix=None,
    prefix_sep='_',columns=vars_cat)


# In[ ]:


X_test = pd.get_dummies(X_test, drop_first=True, dummy_na=False,prefix=None,
    prefix_sep='_',columns=vars_cat)


# In[ ]:


y = train_df['Survived']


# In[ ]:


log_reg = LogisticRegression(penalty="l2",max_iter=10000)


# In[ ]:


log_reg.fit(X_train,y.astype(int))


# In[ ]:


y_pred = log_reg.predict(X_test)


# In[ ]:


from sklearn import model_selection
print(model_selection.cross_val_score(log_reg,X_train,y.astype(int),cv=6).mean())


# In[ ]:


passenger_id = test_df["PassengerId"]
print(y_pred.shape)
output=pd.DataFrame({"PassengerId":passenger_id,"Survived":y_pred})
output


# In[ ]:



output.to_csv('my_submission.csv', index=False)
print("Submission was successfully saved.")

