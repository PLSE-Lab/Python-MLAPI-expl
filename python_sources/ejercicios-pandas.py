#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('/kaggle/input/base-clase-3/basePqs.csv')

data.head(5)


# In[ ]:


data2 = pd.read_csv('/kaggle/input/base-clase-3/basePqs.csv', index_col = "id_cobis")
data2.head(5)


# In[ ]:


data.loc[0]


# In[ ]:


data2.loc[2960236]


# In[ ]:


dataIteraciones = data.head(5)
saldoTotal = 0
for index, row in dataIteraciones.iterrows():
    print("index: ", index)
    print(row)
    saldoTotal += row["saldoPromedio"]

print ("-----------------")    
print ("-----------------")    
print("saldo Ultimo Registro: ",row["saldoPromedio"])
print("suma de Saldos: ", saldoTotal)


# In[ ]:


dataGrouped = data.groupby("target")


# In[ ]:


dataGrouped.saldoPromedio.sum()


# In[ ]:


dataGrouped.saldoPromedio.describe()


# In[ ]:


columns = data.columns
for column in columns:
    print(column, data[column].dtype)
columns = list(columns)


# In[ ]:


for column in columns:
    print("--------------")
    print(column)
    print(dataGrouped[column].describe())
    


# In[ ]:


data.ratioSaldoProm3meses.hist()


# In[ ]:


data.boxplot(column = ["consumoTcCuotas"])


# In[ ]:


data.boxplot(column = ["ratioConsumos1p6meses", "ratioConsumosCuotas6meses"], by = ["target"])


# In[ ]:


import seaborn as sns
gapMiner = pd.read_csv('/kaggle/input/gapminer/gapMiner.csv')

gapMiner.boxplot(by='continent', 
                       column=['lifeExp'], 
                       grid=False)


# In[ ]:


bplot = sns.boxplot(y='lifeExp', x='continent', 
                 data=gapMiner, 
                 width=0.5,
                 palette="colorblind")

