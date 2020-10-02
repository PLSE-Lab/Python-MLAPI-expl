#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd #libreria pandas para utilizar dataframes


# In[ ]:


data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv') # lectura del directorio y archivo en kaggle


# In[ ]:


data #ver los datos


# In[ ]:


#import statistics as stats  


# In[ ]:


sum(data.Deaths)


# In[ ]:


sum(data.Confirmed)


# In[ ]:


sum(data.Recovered)


# In[ ]:


#import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


x=[sum(data.Confirmed),sum(data.Recovered),sum(data.Deaths)]


# In[ ]:


fig = plt.figure()
plt.bar(range(3), x, edgecolor='black')
etiquetas = ['Confirmados', 'Recuperados', 'Muertes']
plt.xticks(range(3), etiquetas)
plt.title("Casos de Covid19")
plt.show()


# In[ ]:


x

