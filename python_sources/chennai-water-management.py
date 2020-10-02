#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


res_levels=pd.read_csv("../input/chennai-water-management/chennai_reservoir_levels.csv")
res_rainfall=pd.read_csv("../input/chennai-water-management/chennai_reservoir_rainfall.csv")


# In[ ]:


res_levels.head(3)


# In[ ]:


res_rainfall.head(5)


# In[ ]:


res_rainfall.isna().sum()


# In[ ]:


res_levels.isna().sum()


# In[ ]:


res_levels.dtypes


# In[ ]:


res_rainfall.dtypes


# In[ ]:


df=res_rainfall.merge(res_levels,on='Date',how="inner",suffixes=("_rain","_res"))


# In[ ]:


df.head(4)


# In[ ]:


res_rainfall.Date=pd.to_datetime(res_rainfall.Date,format="%d-%m-%Y")
res_rainfall.dtypes
res_rainfall.index=res_rainfall.Date
res_rainfall=res_rainfall.drop(['Date'],axis=1)


# In[ ]:


plt.figure(figsize=(12,5))
plt.plot(res_rainfall.POONDI)
plt.title("Rain Fall At POONDI")
plt.xlabel("year")
plt.ylabel("rain fall in Cm")


# In[ ]:


plt.figure(figsize=(12,5))
plt.plot(res_rainfall.POONDI,color='r',label='Poondi')
plt.plot(res_rainfall.CHOLAVARAM,color='y',label='CHOLAVARAM')
plt.plot(res_rainfall.REDHILLS,color='b',label='REDHILLS')
plt.plot(res_rainfall.CHEMBARAMBAKKAM,color='g',label='CHEMBARAMBAKKAM')
plt.legend()
plt.title("Rain Fall At POONDI")
plt.xlabel("year")
plt.ylabel("rain fall in Cm")


# In[ ]:


res_rainfall.head()


# In[ ]:


res_rainfall["Total"]=(res_rainfall.POONDI+res_rainfall.CHOLAVARAM+res_rainfall.REDHILLS+res_rainfall.CHEMBARAMBAKKAM)/4


# In[ ]:


res_rainfall


# In[ ]:


plt.figure(figsize=(12,5))
plt.plot(res_rainfall.Total,color='r')
plt.title("avg rainfall")
plt.xlabel("year")
plt.ylabel("rain in Cm")

