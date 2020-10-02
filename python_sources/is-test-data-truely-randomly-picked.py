#!/usr/bin/env python
# coding: utf-8

# #### This Notebook aim to demonstrate that the test data is selected base on some conditions, while the description does not mention how they pick the test set. By looking at the distribution of certain features and use a RandomForest to classify whether a row come from train/test set, it suggests they are not randomly selected, therefore, certain row may be more useful than others. A random selected validation will not have similar distribution as test set, thus it may not reflect the model accuracy accurately.

# In[72]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[3]:


data = Path('../input')
application_train = pd.read_csv(data/'application_train.csv')
application_test = pd.read_csv(data/'application_test.csv')

tables = [application_train, application_test]


# In[6]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns",1000):       
            display(df)


# The train / test example seems not picked randomly. I just pick a few features that differentiate train/test test, there may be more.

# In[96]:


for i in tables:
    display_all(i.head())


# In[95]:


cols = [i for i in application_train.columns.values if 'AMT' in i]


# In[10]:


cols


# In[14]:


train = application_train[cols]
test = application_test[cols]


# In[15]:


from sklearn.ensemble import  RandomForestClassifier


# ### Now I create a fake column to identify train/testset

# In[16]:


train['is_test'] = 0
test['is_test'] = 1


# ### Since the original dataset has unbalance data, we will sample the data from train set

# In[24]:


n = len(test)


# In[55]:


train = train.iloc[np.random.permutation(len(train))[:n]]


# In[56]:


train.shape,test.shape


# In[57]:


df = pd.concat([train,test])


# In[58]:


df = df.iloc[np.random.permutation(len(df))]


# In[59]:


df.fillna(0,inplace=True) # Fill na with 0 as RandomForest do not accept NaN


# In[60]:


df.head()


# In[61]:


df.is_test.mean() # Now the dataset is balanced


# In[62]:


y = df['is_test'].copy()


# In[63]:


df.drop('is_test',1,inplace=True)


# In[64]:


def split_vals(a,n): return a[:n].copy(), a[n:].copy()


# In[65]:


n_trn = int(len(df)*0.75) # reserve 25% of data as validation


# In[66]:


X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape


# ## Build a RandomForestClassifier to identify is a row coming from train/test seta

# In[67]:


m = RandomForestClassifier(n_estimators=100, max_depth=8)


# ### We now have a balanced dataset, the chance of a particular row is a train/test set should be 0.5 if they are splited randomly.

# In[92]:


m.fit(X_train,y_train)


# In[93]:


print('Accuracy: ',m.score(X_train,y_train))


# In[94]:


print('Accuracy: ',m.score(X_valid,y_valid))


# We can also see different distribution in these features in  train/test set

# In[89]:


for i in cols:
    plt.figure()
    plt.title(i)
    plt.hist([application_train[i],application_test[i]],alpha=0.3,log=True,density=True)
    plt.xticks( rotation='vertical')
    plt.legend(['train','test'])
    plt.show()
   


# In[ ]:




