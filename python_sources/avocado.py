#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd


# In[ ]:


avocado= pd.read_csv('../input/avocado-prices/avocado.csv', index_col=0)
avocado


# In[ ]:


sns.barplot(x= avocado.index, y=avocado['Total Volume'])


# In[ ]:


sns.barplot(x=avocado['Total Volume'],y= avocado['year'])


# In[ ]:


sns.barplot(x= avocado['Total Volume'], y=avocado['Date'])


# In[ ]:


features=['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags','year']

train_data=avocado[features].copy()
val=avocado['AveragePrice']


# In[ ]:




X_train, X_val, y_train, y_val= train_test_split(train_data, val, random_state=40, train_size=0.8)


# In[ ]:


X_train.info()


# In[ ]:


from sklearn import ensemble
elr= ensemble.GradientBoostingRegressor()
model=elr.fit(X_train, y_train)

model.score(X_val, y_val)


# In[ ]:


from sklearn.linear_model import LinearRegression
lr= LinearRegression()
model1=lr.fit(X_train, y_train)
model1.score(X_val, y_val)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfr= RandomForestRegressor()
model2= rfr.fit(X_train, y_train)
model2.score(X_val, y_val)


# In[ ]:


from sklearn.linear_model import  Ridge, Lasso
ridge = Ridge(alpha = 1)  # sets alpha to a default value as baseline  
model3=ridge.fit(X_train, y_train)
model3.score(X_val, y_val)


# In[ ]:


prediction=model2.predict(X_val)
prediction


# In[ ]:


prediction1=model3.predict(X_val)
prediction1


# In[ ]:


output = pd.DataFrame({'year': X_val["year"],
                       'AveragePrice': prediction1})
output.to_csv('submission.csv', index=False)


# In[ ]:




