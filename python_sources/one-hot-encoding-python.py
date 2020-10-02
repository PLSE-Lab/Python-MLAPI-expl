#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_csv(r'../input/Home_price.csv')
df.head(9)


# In[ ]:


dummies = pd.get_dummies(df.town) #creating dummies variable
dummies


# In[ ]:


df_dummies= pd.concat([df,dummies],axis='columns')
df_dummies


# In[ ]:


df_dummies.drop('town',axis='columns',inplace=True)
df_dummies


# In[ ]:


'''Dummy Variable Trap
When you can derive one variable from other variables, they are known to be multi-colinear. Here if you know values of california and georgia then you can easily infer value of new jersey state, i.e. california=0 and georgia=0. There for these state variables are called to be multi-colinear. In this situation linear regression won't work as expected. Hence you need to drop one column.

NOTE: sklearn library takes care of dummy variable trap hence even if you don't drop one of the state columns it is going to work, however we should make a habit of taking care of dummy variable trap ourselves just in case library that you are using is not handling this for you'''


# In[ ]:


df_dummies.drop('west windsor',axis='columns',inplace=True)
df_dummies


# In[ ]:


X = df_dummies.drop('price',axis='columns')
y = df_dummies.price

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)


# In[ ]:


X


# In[ ]:


model.predict(X)


# In[ ]:


model.score(X,y)


# In[ ]:


model.predict([[3400,0,0]])


# In[ ]:


model.predict([[2800,0,1]])


# In[ ]:


'''Using sklearn OneHotEncoder
First step is to use label encoder to convert town names into numbers'''


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[ ]:


dfle = df
dfle.town = le.fit_transform(dfle.town)
dfle


# In[ ]:


X = dfle[['town','area']].values
X


# In[ ]:


y = dfle.price.values
y


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])


# In[ ]:


X = ohe.fit_transform(X).toarray()
X


# In[ ]:


X = X[:,1:]
X


# In[ ]:


model.fit(X,y)


# In[ ]:


model.predict([[0,1,3400]])


# In[ ]:


model.predict([[1,0,2800]])


# In[ ]:




