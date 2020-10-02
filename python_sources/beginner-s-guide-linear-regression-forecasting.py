#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/insurance/insurance.csv')


# In[ ]:


df.tail()


# In[ ]:


df.info


# In[ ]:


df.describe


# In[ ]:


df.columns


# # EDA

# In[ ]:


sns.pairplot(df)


# In[ ]:


sns.distplot(df['charges'])


# In[ ]:


sns.heatmap(df.corr())


# # Training a Linear Regression Model

# In[ ]:


X = df[['age', 'bmi', 'children']] 
y = df['charges']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=200)


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


# # Predictions

# In[ ]:


predictions = lm.predict(X_test)


# In[ ]:


plt.scatter(y_test,predictions)


# In[ ]:


sns.distplot((y_test-predictions),bins=20);


# In[ ]:


from sklearn import metrics


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




