#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train="/kaggle/input/predict-the-number-of-upvotes-a-post-will-get/train_NIR5Yl1.csv"
test="/kaggle/input/predict-the-number-of-upvotes-a-post-will-get/test_8i3B3FC.csv"
train=pd.read_csv(train)
test=pd.read_csv(test)


# In[ ]:


train.drop('ID',axis=1,inplace=True)


# In[ ]:


train.drop('Username',axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


X=train.iloc[:,1:-1]
Y=train.iloc[:,-1]


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[ ]:


from xgboost import XGBRegressor


# In[ ]:


X=XGBRegressor(n_estimators=1000)


# In[ ]:


X.fit(x_train,y_train,early_stopping_rounds=5,eval_set=[(x_test,y_test)],verbose=False)


# In[ ]:


y_pred=X.predict(x_test)


# In[ ]:


from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_pred,y_test)
print(mae)


# In[ ]:



print(y_pred)
print(y_test)


# In[ ]:


test.head()
test_copy=test.copy()
test_copy.head()


# In[ ]:


test.drop('Username',axis=1,inplace=True)
test.drop('ID',axis=1,inplace=True)
test.drop('Tag',axis=1,inplace=True)


# In[ ]:


test.head()


# In[ ]:


Predictions=X.predict(test)


# In[ ]:


output=pd.DataFrame({'Upvotes':Predictions})
output.to_csv('Upvotes.csv',index=False)


# In[ ]:


print(pd.read_csv('Upvotes.csv'))


# In[ ]:




