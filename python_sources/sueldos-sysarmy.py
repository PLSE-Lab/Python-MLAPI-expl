#!/usr/bin/env python
# coding: utf-8

# # PodemosAprender 
# 
# ## Encuesta de sueldos SysArmy

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


## Import dataset
data=pd.read_csv('../input/encuesta.csv',skiprows=8)    ## First rows are commentaries not data
data.drop_duplicates(keep = 'first', inplace = True)    ##remove duplicates


# In[ ]:


data.head()


# In[ ]:


##columns names
data.columns


# In[ ]:


data.describe()


# Veamos el promedio de salarios segun el nivel de estudios alcanzados

# In[ ]:


mean_salario = data[['Salario mensual NETO (en tu moneda local)','Nivel de estudios alcanzado']].groupby('Nivel de estudios alcanzado').mean()
mean_salario


# In[ ]:


mean_salario.plot(kind='bar')


# Tiene sentido este resultado? Hay que limpiar datos, quizas sacar outliers, etc 

# In[ ]:




