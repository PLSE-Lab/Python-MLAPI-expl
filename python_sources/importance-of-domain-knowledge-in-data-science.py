#!/usr/bin/env python
# coding: utf-8

# ## I wrote a medium post to explain why and how domain knowledge is important in data science. This kernel is a part of it. You can read about it [here](https://medium.com/@anand0427/why-domain-knowledge-is-important-in-data-science-anand0427-3002c659c0a5). 

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt


# ## Without feature engineering.

# In[ ]:


df=pd.read_csv('../input/GDP_CAT.csv')


# In[ ]:


df = df.iloc[::-1]


# In[ ]:


df = df.set_index('Year')


# In[ ]:


df.head()


# In[ ]:


X=['Consumer expenditure household','Consumer public adm','Equip. Goods others','Const.',
   'Total exports goods and services','Total imports goods and services']
X=df[X]
y=df.GDP
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.score(X_test,y_test))
print(sqrt(mean_squared_error(y_test,lm.predict(X_test))))


# ## With feature engineering.

# In[ ]:


Cons_per_GDP=df['Const.']/df.GDP*100
Exports_per_GDP=df['Total exports goods and services']/df.GDP*100
df['Cons_per_GDP']=Cons_per_GDP
df['Exports_per_GDP']=Exports_per_GDP
Domestic_Demand_per_GDP_wc=(df['Domestic demand']-df['Const.'])/df.GDP*100
df['Domestic_Demand_per_GDP_wc']=Domestic_Demand_per_GDP_wc
df['trad_op']= (df['Total exports goods and services']+df['Total imports goods and services'])
df['trad_op']= (df['trad_op']/df.GDP*100) 
df['pct_change']=(df.GDP.pct_change()*100)        


# In[ ]:


X=['Consumer expenditure household','Consumer public adm','Equip. Goods others','Const.',
   'Total exports goods and services','Total imports goods and services','trad_op',
   'Domestic_Demand_per_GDP_wc'] 
X=df[X]
y=df.GDP
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.score(X_test,y_test))
print(sqrt(mean_squared_error(y_test,lm.predict(X_test))))


# ### Check out this amazing kernel by xavier14 [here](http://https://www.kaggle.com/xavier14/catalonia-gdp-insights-and-regression-analysis) which is used to create this kernel to display the advantage of domain knowledge. 

# In[ ]:




