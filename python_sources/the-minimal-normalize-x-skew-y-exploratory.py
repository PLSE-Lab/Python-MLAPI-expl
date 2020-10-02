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


# In[41]:


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


# In[42]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[43]:


train.head()


# In[44]:


train = train.select_dtypes(include=['float64','int64'])
train.fillna(train.median(),inplace=True)
#train = train[train.columns[1:]] # descarding "Id"
print(train.columns)
train.info()


# In[45]:


#separate variables into new data frames
numeric_data = train.select_dtypes(include=[np.number])
cat_data = train.select_dtypes(exclude=[np.number])
print ("There are {} numeric and {} categorical columns in train data".format(numeric_data.shape[1],cat_data.shape[1]))


# In[46]:


#correlation plot
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15.0, 15.0)
import seaborn as sns
corr = numeric_data.corr()
sns.heatmap(corr)


# In[47]:


corr


# In[48]:


low_var_cols = numeric_data.columns[numeric_data.var() < 0.2]
print(low_var_cols)
numeric_data[low_var_cols.tolist()+["SalePrice"]].corr()


# Try using features you think that will improve the Model

# In[49]:


train.isna().sum()


# In[50]:


selected_cols = train.columns.tolist()
print(selected_cols.pop(selected_cols.index("Id")))
print(selected_cols.pop(selected_cols.index("BsmtHalfBath")))
#selected_cols = ["YearBuilt","LotArea","SalePrice"]
feat_cols = selected_cols[:-1]
train = train[selected_cols]
train.shape


# In[51]:


train["SalePrice"].hist()


# In[52]:


skewed_columns = train.columns[abs(train.skew()) > 1.7]
skewed_columns


# In[53]:


train_bkp = train.copy()
train[skewed_columns] = np.log2(train[skewed_columns])


# In[54]:


np.log2(train["SalePrice"]).hist()


# In[55]:


inf_issue = train.columns[np.isinf(train).sum() > 0]
inf_issue


# In[56]:


def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]
cols_2_log = diff(skewed_columns,inf_issue)
cols_2_log


# In[57]:


train = train_bkp.copy()
train[cols_2_log] = np.log2(train[cols_2_log])


# In[58]:


x_train_st , x_test_st = train_test_split(train,train_size=0.7)
print(x_train_st.shape)
print(x_test_st.shape)
y_train_st = x_train_st["SalePrice"]
y_test_st = x_test_st["SalePrice"]
x_train_st = x_train_st.loc[:,feat_cols]
x_test_st = x_test_st.loc[:,feat_cols]


# In[59]:


#y_test_st = np.log10(y_test_st)
#y_train_st = np.log10(y_train_st)


# In[ ]:





# In[ ]:





# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

# In[60]:


LR_Test = linear_model.LinearRegression(normalize=True).fit(x_train_st.values, y_train_st)
train_test(LR_Test, x_train_st, x_test_st, y_train_st, y_test_st)


# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
# 
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html

# In[61]:


pred = LR_Test.predict(x_test_st)
act = y_test_st
x = act
y = pred
#plt.plot(act,pred,"o")

fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
# fit_fn is now a function which takes in x and returns an estimate for y

plt.plot(x,y, 'o', x, fit_fn(x), '--k')


# In[62]:


test.fillna(x_train_st.median(),inplace=True)


# In[63]:


test_bkp = test.copy()
test[cols_2_log[:-1]] = np.log2(test[cols_2_log[:-1]])
cols_2_log


# In[67]:


kag_pred = LR_Test.predict(test[feat_cols])


# In[68]:


kag_pred = np.power(2,kag_pred)
kag_pred


# In[69]:


np.isinf(kag_pred).sum()


# In[70]:


kag_pred[np.isinf(kag_pred)] = np.median(kag_pred)
np.isinf(kag_pred).sum()


# In[71]:


sub = pd.DataFrame()
sub['Id'] = test["Id"]
sub['SalePrice'] = kag_pred
sub.to_csv('submission.csv',index=False)


# In[ ]:




