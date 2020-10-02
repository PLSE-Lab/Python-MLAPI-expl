#!/usr/bin/env python
# coding: utf-8

# In[238]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# In[239]:


data = pd.read_csv("../input/winequality-red.csv")

data.head()


# In[240]:


data.describe()


# In[241]:


data['quality'].value_counts()


# In[242]:


data.columns


# In[243]:


for i in data.columns[0:11]:
    print(i,stats.spearmanr(data['quality'],data[i]))


# In[244]:


data.corr()


# In[245]:


x = data[data.columns[0:11]]

y = data['quality']


# In[246]:


import seaborn as sns

sns.pairplot(data, x_vars='quality', y_vars=data.columns[0:11], kind='reg')


# In[247]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[259]:


model = LinearRegression()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.01)

model.fit(x_train,y_train)

a = model.predict(x_test)


# In[249]:


from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test,a)


# In[250]:


model.coef_


# In[251]:


for i in data.columns[0:11]:
    print(i,stats.spearmanr(data['quality'],data[i]))


# In[258]:


print(model.score(x_test,y_test))
print(model.coef_)


# In[260]:


fig,ax = plt.subplots()

ax = plt.scatter(y_test,a)

plt.plot()


# In[282]:


reg = pd.DataFrame(x_test,columns=x_test.columns)

reg['quality']=a

reg

sns.pairplot(reg,x_vars=reg.columns[0:11],y_vars='quality',kind='reg')

reg['quality']=y_test
sns.pairplot(reg,x_vars=reg.columns[0:11],y_vars='quality',kind='reg')

