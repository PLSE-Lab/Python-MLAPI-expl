#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import seaborn as sns
import matplotlib.pyplot as plt   #Data visualisation libraries 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# import datasets
files = glob.glob("../input/*.csv")
list = []
for f in files:
    df = pd.read_csv(f,index_col=None)
    list.append(df)
df = pd.concat(list)


# In[ ]:


# drops columns
df = df.drop(['from_address','to_address','status','date'],axis=1)


# In[ ]:


df.head()


# In[ ]:


print(df.dtypes)


# **Split tain_test with Sci-kit learn**

# split train 70% and test 30%

# In[ ]:


# import sklearn
from sklearn.model_selection import train_test_split


# In[ ]:


X = df[['open','high','low','volumefrom']]
y = df['close']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


# import Lib
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score


# In[ ]:


# build the model using sklearn
lm = LinearRegression()
lm.fit(X_train,y_train)
r2 = lm.score(X_train,y_train)
predictions = lm.predict(X_test)
print("R-squared :",r2)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print(r2_score(y_test, predictions))
    


# In[ ]:


coef = pd.DataFrame(lm.coef_, X.columns, columns = ['Coefficients'])
coef


# ****Feature selection based on correlation****

# Correlation martix

# In[ ]:


corr = X.corr()
corr


# In[ ]:


columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.7:
            if columns[j]:
                columns[j] = False
selected_columns = X.columns[columns]
data = X[selected_columns]


# In[ ]:


data.corr()


# drop columns high and low because a correlation higher than 0.7

# In[ ]:


X = df[['open','volumefrom']]
y = df['close']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# **build model without feature high and low**

# In[ ]:


# build the model using sklearn
lm = LinearRegression()
lm.fit(X_train,y_train)
r2 = lm.score(X_train,y_train)
predictions = lm.predict(X_test)
print("R-squared :",r2)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print(r2_score(y_test, predictions))


# In[ ]:


lm.coef_


# In[ ]:


sns.regplot(y_test, predictions)


# In[ ]:




