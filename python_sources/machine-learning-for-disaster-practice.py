#!/usr/bin/env python
# coding: utf-8

# ### Importing the libraries

# In[ ]:


import math
import datetime
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from scipy import stats
from scipy.stats import skew

np.random.seed(2020)

get_ipython().run_line_magic('matplotlib', 'inline')

print("done")


# ### Importing the data

# In[ ]:




def read_and_concat_dataset(training_path, test_path):
    train = pd.read_csv(training_path)
    train['train'] = 1
    test = pd.read_csv(test_path)
    test['train'] = 0
    data = train.append(test, ignore_index=True)
    return train, test, data

train, test, data = read_and_concat_dataset('../input/titanic/train.csv', '../input/titanic/test.csv')
data = data.set_index('PassengerId')


# In[ ]:


train, test, data = read_n_conc(train_path,test_path)
data.set_index('PassengerId', inplace=True)


# ### Exploring the data

# In[ ]:


data.head(5)


# In[ ]:


data.describe()


# In[ ]:


sns.heatmap(data[['Survived','SibSp','Parch','Age','Fare']].corr(),annot=True,cmap='coolwarm')


# In[ ]:


data[data['Parch'].isnull()==True] #No passenger has their Parch value missing 


# In[ ]:


data[data['Survived'].isnull()==True] # A lot passenger have their Survived value missing


# In[ ]:


data.dtypes


# In[ ]:


def comparing(data,variable1, variable2):
    print(data[[variable1, variable2]][data[variable2].isnull()==False].groupby([variable1], as_index=False).mean().sort_values(by=variable2, ascending=False))
    g = sns.FacetGrid(data, col=variable2).map(sns.distplot, variable1)
def counting_values(data, variable1, variable2):
    return data[[variable1, variable2]][data[variable2].isnull()==False].groupby([variable1], as_index=False).mean().sort_values(by=variable2, ascending=False)


# In[ ]:


comparing(data, 'Parch','Survived')


# In[ ]:


import warnings
warnings.filterwarnings("ignore")

def compare(data,var1,var2):
    comp = data[[var1,var2]][data[var2].isnull()==False] # getting the required columns and removing NaN entries
    comp = comp.groupby([var1],as_index=False).mean()
    comp = comp.sort_values(by=var2, ascending=False)
    print(comp)
    g= sns.FacetGrid(comp,col=var2).map(sns.distplot,var1)


# In[ ]:


compare(data, 'Parch', 'Survived')


# In[ ]:




