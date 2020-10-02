#!/usr/bin/env python
# coding: utf-8

# In[22]:


#PASO: arranco con lo basico, veo que archivos tengo
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print("Available files")
print(os.listdir("../input/crunch2013"))


# In[23]:


#PASO: cargo un archivo, uso head para ver que columnas/que pinta tienen los datos
companies= pd.read_csv("../input/crunch2013/crunchbase-companies.csv", encoding = "ISO-8859-1")
#TRUCO: aunque los datos vienen "lindos" en csv, en este caso si no le agrego el encoding da UNICODEERROR
companies.head()


# In[24]:


#PASO: uso value_counts por columna para ver si los datos son surtidos o que sesgos pueden tener, cuantos son, etc.
companies.country_code.value_counts()


# In[25]:


#PASO: para el rubro tengo algo un poco mas interesante
companies.category_code.value_counts()


# In[52]:


#PASO: me interesa la columna de funding, de que numeros hablamos? Empiezo por la facil...
pd.to_numeric(companies.funding_total_usd, errors= 'coerce').dropna().describe().apply(lambda x: '%.f' % (x/1000))
#TRUCO: tenian un poco de datos basura, hasta "dropna" limpie
#TRUCO: con apply le cambie el formato Y DIVIDI POR MIL (podia hacer cualquier cuenta que quiera)


# In[ ]:




