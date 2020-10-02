#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[21]:


df = pd.read_csv("../input/insurance.csv")


# In[22]:


df.describe()


# In[23]:


df.boxplot()


# In[24]:


df.info()


# In[25]:


df.head()


# In[26]:


df["sex"].value_counts()


# In[27]:


df["sex"] = df.sex.replace({'male':1,'female':2})


# In[28]:


df.smoker[df.smoker=='no']=2


# In[29]:


df["smoker"] = df.smoker.replace({'yes'=1,'no'=2})


# In[30]:


df.smoker[df.smoker=='yes'] = 1
df.smoker[df.smoker=='2']=2


# In[39]:


df.head()


# In[32]:


df["region"].value_counts()


# In[38]:


df["region"] = df.region.replace({'southeast':1,'southwest':2,'northwest':3,'northeast':4})


# In[66]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[67]:


sns.pairplot(data = df)
plt.show()


# In[42]:


df.corr()


# In[43]:


df.info()


# In[44]:


y = df["expenses"]
x =  df.drop(columns=["expenses","sex","region"])


# In[45]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[46]:


train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state = 42)


# In[47]:


train_x.shape


# In[48]:


test_x.shape


# In[49]:


lm = LinearRegression()
lm.fit(train_x,train_y)


# In[50]:


plt.scatter(x["age"],y)


# In[51]:


lm.intercept_


# In[52]:


lm.coef_


# In[53]:


train_predict = lm.predict(train_x)


# In[54]:


test_predict = lm.predict(test_x)


# In[55]:


train_predict


# In[59]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print("MSE : ", mean_squared_error(train_y,train_predict))
print("R2 : " ,r2_score(train_y,train_predict))

