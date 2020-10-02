#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:



import numpy as np
import pandas as pd
import folium
from folium import plugins





df_acc = pd.read_csv("../input/report_12_01_2018_2.csv", dtype=object)



# In[ ]:



df_acc.head()
             


# In[ ]:


import folium
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 7))
sns.heatmap(df_acc.isnull(), cbar = False, cmap = "viridis")


# In[ ]:


pd.value_counts(df_acc['fuerocomun_desapentidad'])[:10]


# In[ ]:


pd.value_counts(df_acc['fuerocomun_desaphora'])[:10]


# In[ ]:


pd.value_counts(df_acc['fuerocomun_desapfecha'])[:20]


# In[ ]:


pd.value_counts(df_acc['fuerocomun_desapmunicipio'])[:20]


# In[ ]:


pd.value_counts(df_acc['fuerocomun_edad'])[:20]


# In[ ]:


df_acc.info()


# In[ ]:


plt.figure(figsize = (15, 10))
sns.countplot(y= 'fuerocomun_desapentidad', data = df_acc, order = df_acc['fuerocomun_desapentidad'].value_counts().iloc[:10].index)


# In[ ]:


locations = df_acc.groupby('fuerocomun_desapentidad')


# In[ ]:



    


# In[ ]:


m = folium.Map(location=[23.6260333, -102.5375005], zoom_start = 5)

