#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('/kaggle/input/diamonds/diamonds.csv')


# In[ ]:


df.shape


# In[ ]:


df.head()


# delete 'Unnamed: 0' since this is not useful in predicting the target

# In[ ]:


del df['Unnamed: 0']


# In[ ]:


df.head()


# Rather than using Label Encode I am encoding manually since the number of classes are fewer. The drawback of label encoding in Python is that it does not recognize the natural ordering between element the classes of feature.

# In[ ]:


df['cut'].value_counts()


# In[ ]:


df['cut'].replace(('Fair','Good','Very Good','Premium','Ideal'),(1,2,3,4,5), inplace = True)


# In[ ]:


df['color'].value_counts()


# In[ ]:


df['color'].replace(('D','E','F','G','H','I','J'),(7,6,5,4,3,2,1), inplace = True)


# In[ ]:


df['clarity'].replace(('I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'),(1,2,3,4,5,6,7,8),inplace = True)


# In[ ]:


df.head()


# Now every feature in the input is numerical and we want to check for any missing values in the data

# In[ ]:


df.isnull().sum()


# In[ ]:


df.dtypes


# There are no missing values and therefore we can proceed ahead with out model building activity

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score


# Feature selection with RandomForestRegressor

# In[ ]:


y = df['price']
X = df.drop('price', axis = 1)
print(y.shape)
print(X.shape)


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2)


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# In[ ]:


rf = RandomForestRegressor()
rf.fit(X_train,y_train)


# In[ ]:


for feature in zip(X. columns, rf.feature_importances_):
    print(feature)


# This indicates that only 'carat' and 'y' are the most important features in predicting target

# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
pred_lr = lr.predict(X_test)


# In[ ]:


pred_rf = rf.predict(X_test)


# In[ ]:


lr_mse = mean_squared_error(y_test,pred_lr)
rf_mse = mean_squared_error(y_test,pred_rf)
print(lr_mse)
print(rf_mse)


# In[ ]:


print(r2_score(y_test,pred_lr))
print(r2_score(y_test,pred_rf))


# If you like this article on diamond price prediction please give an upvote. Thanks
