#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


dataset=pd.read_csv('../input/50_Startups.csv')


# In[ ]:


dataset.head(5)


# In[ ]:


X=dataset.iloc[:,:-1]
Y=dataset.iloc[:,4]


# In[ ]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder


# In[ ]:


labelencoder=LabelEncoder()


# In[ ]:


X['State_n']=labelencoder.fit_transform(X['State'])


# In[ ]:


X_n=X.drop('State',axis='columns')
X_n.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_n, Y, test_size = 0.2, random_state = 0)


# In[ ]:


regressor = LinearRegression()


# In[ ]:


regressor.fit(X_train,Y_train)


# In[ ]:


Y_predict = regressor.predict(X_test)


# In[ ]:


regressor.intercept_


# In[ ]:


regressor.coef_


# In[ ]:


regressor.score(X_n,Y)


# In[ ]:


regressor.predict([[162345.47,139827,236829,5]])

