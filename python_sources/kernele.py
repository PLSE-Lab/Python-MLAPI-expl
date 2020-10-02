#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn


# In[ ]:


oecd_bli=pd.read_csv('E:/data/oecd_bli_2017.csv',thousands=',')


# In[ ]:


gdp_per_capita=pd.read_csv('E:/data/gdp_per_capita.csv',thousands=',',delimiter='\t',encoding='latin1',na_values="n/a")


# In[ ]:


country_stats=(oecd_bli,gdp_per_capita)


# In[ ]:


x=np.c_[country_stats["GDP per capita"]]
y=np.c_[country_stats["Life satisfaction"]]


# In[ ]:


country_stats.plot(kind='scatter',x="GDP per capita",y='Life satisfaction') 
plt.show()


# In[ ]:


lin_reg_model=sklearn.linear_model.LinearRegression()


# In[ ]:


lin_reg_model.fit(X,y)


# In[ ]:


#Make a prediction for Cyprus
X_new=[[22587]] #cyprus'GDP per capita print(lin_reg_model.predict(X_new)

