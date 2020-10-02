#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os 
import pandas as pd

el = pd.read_csv('../input/elnino.csv')
el.columns = [col.strip() for col in el.columns]
el.columns = [col.replace(' ','_') for col in el.columns]


# In[ ]:


el.head()


# In[ ]:


el.describe()


# In[ ]:


plt.matshow(el.corr())
plt.colorbar()
plt.show()


# In[ ]:


p = el.hist(figsize = (20,20))


# In[ ]:


el['Zonal_Winds'] = pd.to_numeric(el['Zonal_Winds'], errors='coerce')
el['Zonal_Winds'].describe()


# In[ ]:


el['Meridional_Winds'] = pd.to_numeric(el['Meridional_Winds'], errors='coerce')
el['Meridional_Winds'].describe()


# In[ ]:


sns.jointplot(x="Zonal_Winds", y="Meridional_Winds", data=el)


# In[ ]:


el['Air_Temp'] = pd.to_numeric(el['Air_Temp'], errors='coerce')
el['Air_Temp'].describe()


# In[ ]:


sns.jointplot(x="Zonal_Winds", y="Air_Temp", data=el)


# In[ ]:


sns.jointplot(x="Meridional_Winds", y="Air_Temp", data=el)


# In[ ]:


sns.lineplot(x='Zonal_Winds', y='Meridional_Winds', data=el)


# In[ ]:


sns.lineplot(x='Zonal_Winds', y='Air_Temp', data=el)


# In[ ]:


sns.lineplot(x='Meridional_Winds', y='Air_Temp', data=el)

