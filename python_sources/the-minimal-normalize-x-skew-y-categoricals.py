#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


# Adding needed libraries and reading data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

# Prints R2 and RMSE scores
def get_score(prediction, lables):    
    print('R2: {}'.format(r2_score(prediction, lables)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))

# Shows scores for train and validation sets    
def train_test(estimator, x_trn, x_tst, y_trn, y_tst):
    prediction_train = estimator.predict(x_trn)
    # Printing estimator
    print(estimator)
    # Printing train scores
    get_score(prediction_train, y_trn)
    prediction_test = estimator.predict(x_tst)
    # Printing test scores
    print("Test")
    get_score(prediction_test, y_tst)


# In[15]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test.head()


# In[16]:


train.head()


# In[17]:


ntrain = train.shape[0]
y_train = train["SalePrice"]
train = train[train.columns[:-1]]
train = train.append(test)


# In[18]:


obj_cols = train.columns[train.dtypes == "object"]
train[obj_cols] = train[obj_cols].fillna("None") # No pool ?
train = pd.get_dummies(train,columns=obj_cols)


# In[19]:



#for col_i in obj_cols:
#    print("Col_name : ", col_i)
#    #print("values : \n", train[col_i].value_counts())
#    print("NAs : ", train[col_i].isna().sum()) 


# In[20]:


train = train.select_dtypes(include=['float64','int64','uint8'])
train.fillna(train.median(),inplace=True)
print(train.columns)
train.info()


# Try using features you think that will improve the Model

# In[21]:


# get a list of columns
#cols = list(train)
# move the column to head of list using index, pop and insert
#cols.append(cols.pop(cols.index('SalePrice')))
#cols
# use ix to reorder
#train = train.ix[:, cols]
#train.head()


# In[22]:


feat_cols = train.columns
train = train[feat_cols]
train.shape


# In[23]:


#train["SalePrice"].hist()
y_train.hist()


# In[24]:


skewed_columns = train.columns[abs(train.skew()) > 1.7]
skewed_columns


# In[ ]:





# In[25]:


np.log2(y_train).hist()


# In[26]:


train_bkp = train.copy()
train[skewed_columns] = np.log2(train[skewed_columns])

inf_issue = train.columns[np.isinf(train).sum() > 0]
#inf_issue
def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]
cols_2_log = diff(skewed_columns,inf_issue)
cols_2_log


# In[27]:


train = train_bkp.copy()
train[cols_2_log] = np.log2(train[cols_2_log])


# In[28]:


train.shape


# In[29]:


y_train = np.log2(y_train)


# In[30]:


x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(train[:ntrain],y_train,train_size=0.7)
print(x_train_st.shape)
print(x_test_st.shape)


# In[31]:


#y_test_st = np.log10(y_test_st)
#y_train_st = np.log10(y_train_st)


# In[ ]:





# In[ ]:





# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

# In[32]:


LR_Test = linear_model.ElasticNetCV(normalize=True).fit(x_train_st.values, y_train_st)
train_test(LR_Test, x_train_st, x_test_st, y_train_st, y_test_st)


# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
# 
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html

# In[33]:


pred = LR_Test.predict(x_test_st)
act = y_test_st
x = act
y = pred
#plt.plot(act,pred,"o")

fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
# fit_fn is now a function which takes in x and returns an estimate for y

plt.plot(x,y, 'o', x, fit_fn(x), '--k')


# In[34]:


test = train[ntrain:]


# In[36]:


test.shape


# In[37]:


kag_pred = LR_Test.predict(test)


# In[38]:


kag_pred = np.power(2,kag_pred)
kag_pred


# In[39]:


sub = pd.DataFrame()
sub['Id'] = test["Id"]
sub['SalePrice'] = kag_pred
sub.to_csv('submission.csv',index=False)


# In[ ]:




