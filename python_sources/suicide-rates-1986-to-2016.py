#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[ ]:


pwd


# In[ ]:


df= pd.read_csv('/kaggle/input/suicide.csv')
df.head()


# In[ ]:


gen_map={'Generation X': 1,'Silent': 2, 'Boomers': 3,'Millenials':4,'G.I. Generation':5,'Generation Z':6 }
df.generation=df.generation.map(gen_map)
df.head()


# In[ ]:


df=df.describe()
df.head()


# In[ ]:


df.isnull().count()


# In[ ]:


df=df.fillna(method='ffill')


# In[ ]:


df.info()


# In[ ]:


df.dtypes


# In[ ]:


df=df.rename(columns={'HDI for year':'HDI of year','suicides/100k pop':'suicides_per_lakhs'})


# In[ ]:


df.dtypes


# In[ ]:


Suicides_no=df.suicides_no.astype('category')
df.suicides_no=df.suicides_no.astype('category').cat.codes


# In[ ]:


Population=df.population.astype('category')
df.population=df.population.astype('category').cat.codes


# In[ ]:


Suicides_per_lakhs=df.suicides_per_lakhs.astype('category')
df.suicides_per_lakhs=df.suicides_per_lakhs.astype('category').cat.codes


# In[ ]:


df.info()


# In[ ]:


df=df.drop(['HDI of year'], axis=1)
df.head()


# In[ ]:


x=df['suicides_no']
y=df['population']
sns.barplot(x,y,color='red')
plt.xlabel('suicides_no')
plt.ylabel('population')
plt.title('suicide_no vs population')
plt.show()


# In[ ]:


x=df['suicides_no']
y=df['population']
sns.residplot(x,y,data=df)
plt.xlabel('suicides_no')
plt.ylabel('generation')
plt.title('suicide_no vs generation')
plt.show()


# In[ ]:


x=df['population']
y=df['gdp_per_capita ($)']
sns.barplot(x,y,color='green')
plt.xlabel('population')
plt.ylabel('gdp_per_capita ($)')
plt.title('suicide_no vs gdp_per_capita ($)')
plt.show()


# In[ ]:


df.dtypes


# In[ ]:


x=df['suicides_no']
y=df['population']
plt.scatter(x,y)
plt.xlabel('suicides_no')
plt.ylabel('population')
plt.title('comparision between population and suicide no')


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[ ]:


x=df.drop(['suicides_no'],axis=1)
y=df['suicides_no']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=2,test_size=0.2)


# In[ ]:


lr= LinearRegression(fit_intercept=True,n_jobs=1)
lr.fit(x_train,y_train)
lr.predict(x_test)
lr.score(x_train,y_train)


# In[ ]:


support= SVR(degree=7, cache_size=400)
support.fit(x_train,y_train)
support_predict=support.predict(x_test)
support.score(x_train,y_train)


# In[ ]:





# In[ ]:




