#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.set_style('darkgrid')


# In[ ]:


df_train = pd.read_csv('../input/train.csv')


# In[ ]:


df_train.shape


# In[ ]:


df_train.head()


# In[ ]:


df_train.info()


# In[ ]:


df_train.describe()


# In[ ]:


df_train.columns


# In[ ]:


sns.distplot(df_train['SalePrice'])


# In[ ]:


df_train.corr()


# In[ ]:


f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(df_train.corr(), annot = True,linewidths=.2, fmt='.1f', ax=ax)


# In[ ]:





# In[ ]:





# In[ ]:


df_train['SalePrice'].describe()


# In[ ]:





# In[ ]:


sns.pairplot(df_train,x_vars=['TotalBsmtSF','GrLivArea','GarageArea','OverallQual','FullBath'],
             y_vars=['SalePrice'],kind='reg')


# In[ ]:


df_train.columns


# In[ ]:


X = df_train[['TotalBsmtSF','GrLivArea','GarageArea','OverallQual','FullBath']]


# In[ ]:


y = df_train['SalePrice']


# In[ ]:


#from sklearn.cross_validation import train_test_split 
from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train,y_train)


# In[ ]:


print(lm.intercept_)


# In[ ]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[ ]:


predictions = lm.predict(X_test)


# In[ ]:


plt.scatter(y_test,predictions)


# In[ ]:


sns.distplot((y_test-predictions),bins=35)


# In[ ]:


from sklearn import metrics


# In[ ]:


#Mean Absolute Error 
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

#Mean Squared Error
print('MSE:', metrics.mean_squared_error(y_test, predictions))

#Root Mean Squared Error
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




