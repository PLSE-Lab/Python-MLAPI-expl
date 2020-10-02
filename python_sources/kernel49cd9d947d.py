#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# In[ ]:


df=pd.read_csv('../input/Fish.csv')
df.head()


# In[ ]:


df.tail()


# In[ ]:


import seaborn as sns
sns.pairplot(df,x_vars=['Length1','Length2','Length3','Height','Width'],y_vars=['Weight'],size=7,aspect=0.7,kind='reg')


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[ ]:


dfle=df
dfle.Species=le.fit_transform(dfle.Species)
dfle.tail()


# In[ ]:


X=dfle[['Species','Length1','Length2','Length3','Height','Width']]#removed height,width
y=dfle['Weight']
X.head()


# In[ ]:


y.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=0)


# In[ ]:


X_test
len(X_test)


# In[ ]:


model=LinearRegression()
model.fit(X,y)


# In[ ]:


model.predict([[0 ,28 ,30.4 ,36.0 ,14.5200 ,4.9500]])


# In[ ]:


print(model.score(X_test,y_test)*100,"%")


# In[ ]:


model.coef_


# In[ ]:


model.intercept_


# In[ ]:


df.tail(10)


# In[ ]:


from sklearn.externals import joblib


# In[ ]:


joblib.dump(model,'predicted_fish.pkl')


# In[ ]:


my_model=joblib.load('predicted_fish.pkl')


# In[ ]:


my_model.predict([[5,13.8 ,15.0 ,16.2 ,2.9322 ,1.8792]])


# In[ ]:




