#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df= pd.read_csv("../input/House_file.csv")
df.head()


# > Linear regression with single variable  and Gradient boosting

# In[ ]:


reg=LinearRegression()
reg


# In[ ]:


reg.fit(df[["SqFt"]],df.Price)


# In[ ]:


y_pred=reg.predict([[2100]])


# Model accuracy with linear regression

# In[ ]:


reg.score(df[["SqFt"]],df.Price)


# gradient boost model for regression

# In[ ]:


gb=GradientBoostingRegressor()  


# In[ ]:


gb.fit(df[["SqFt"]],df.Price)


# In[ ]:


gb_predict=gb.predict([[2100]])


# Model accuracy with Gradient boosting model

# In[ ]:


gb.score(df[["SqFt"]],df.Price)


# Linear Regression with multi variable and Gradient boosting

# In[ ]:


mreg=LinearRegression()


# In[ ]:


mreg.fit(df[["SqFt","Bedrooms"]],df.Price)


# In[ ]:


mreg_predict=mreg.predict([[2000,2]])
mreg_predict


# In[ ]:


mreg.score(df[["SqFt","Bedrooms"]],df.Price)


# Lets see gradient boosting for the same.

# In[ ]:


mgb=GradientBoostingRegressor()
mgb


# In[ ]:


mgb.fit(df[["SqFt","Bedrooms"]],df.Price)


# In[ ]:


mgb_predict=mgb.predict([[2000,2]])
mgb_predict


# In[ ]:


mgb.score(df[["SqFt","Bedrooms"]],df.Price)

