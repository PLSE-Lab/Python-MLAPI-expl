#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


dados_2015 =pd.read_csv('/kaggle/input/world-happiness/2015.csv')
dados_2016 =pd.read_csv('/kaggle/input/world-happiness/2016.csv')
dados_2017 =pd.read_csv('/kaggle/input/world-happiness/2017.csv')


# In[ ]:


print('2015:', dados_2015.shape)
print('2016:', dados_2016.shape)
print('2017:', dados_2017.shape)


# In[ ]:


dados_2015.head()


# In[ ]:


dados_2016.head()


# In[ ]:


import pandas_profiling


# In[ ]:


pandas_profiling.ProfileReport(dados_2017)


# In[ ]:


dados_concat =  result = pd.concat([dados_2015, dados_2016, dados_2017], axis=1)


# In[ ]:


dados_concat.info()


# In[ ]:


dados_concat.head().T


# In[ ]:


dados2 = pd.merge(dados_2015, dados_2016, on = 'Country')
 #result = pd.merge(left, right, on='key')


# In[ ]:


dados2.head(5).T


# In[ ]:


dftotal = pd.merge(dados2,dados_2017, on = 'Country', how='full')


# In[ ]:


dftotal.head(5).T


# In[ ]:


dftotal.shape


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


dados_2015.info()


# In[ ]:


dados_2015.plot(title = '2015', kind='scatter',x='Economy (GDP per Capita)',y='Happiness Score',color='red')


# In[ ]:


dados_2015.plot(title = '2015', kind='scatter',x='Economy (GDP per Capita)',y='Happiness Score',color='red')
dados_2016.plot(title = '2016', kind='scatter',x='Economy (GDP per Capita)',y='Happiness Score',color='blue')
dados_2017.plot(title = '2017', kind='scatter',x='Economy..GDP.per.Capita.',y='Happiness.Score',color='green')


# In[ ]:


dados_2017.info()


# In[ ]:


sns.stripplot(x='Region', y='Happiness Score', data = dados_2015)
plt.xticks(rotation=90)


# In[ ]:


pip install pygal


# In[ ]:


sns.stripplot(x='Region', y='Happiness Score', data = dados_2015)


# In[ ]:


dados_2016.corr()


# In[ ]:



f,ax = plt.subplots(figsize=(16,6))
sns.heatmap(dados_2016.corr(),annot=True, fmt='.2f',linecolor='black',lw=.7,ax=ax,center=0 )


# In[ ]:


q3 = dados_2016['Happiness Score'].quantile(0.75)
q2= dados_2016['Happiness Score'].quantile(0.5)
q1= dados_2016['Happiness Score'].quantile(0.25)

happy_quality = []

# Percorrer a coluna do adtaframe e determinar as categorias
for valor in dados_2016['Happiness Score']:
    if valor >= q3:
        happy_quality.append('Muito Alto')
    elif valor < q3 and valor >= q2:
        happy_quality.append('Alto')
    elif valor < q2 and valor >=q1:
        happy_quality.append('Normal')
    else:
        happy_quality.append('Baixo')
    
dados_2016['happy_quality']= happy_quality


# In[ ]:


dados_2016.tail()


# In[ ]:


#box plot
sns.boxplot(dados_2016['happy_quality'],dados_2016['Economy (GDP per Capita)'])


# In[ ]:


dados_2016.nsmallest(5,'Economy (GDP per Capita)')


# In[ ]:


#plt.figure(figsize)

sns.swarmplot(dados_2016['happy_quality'],dados_2016['Economy (GDP per Capita)'])


# In[ ]:


#correlacao entre health e PIB

plt.figure(figsize=(7,7))

sns.scatterplot(dados_2016['Health (Life Expectancy)'],
               dados_2016['Economy (GDP per Capita)'],
               hue=dados_2016['happy_quality'],
               style = dados_2016['happy_quality'])

