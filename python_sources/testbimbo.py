#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


tp = pd.read_csv("../input/train.csv", chunksize=500000,
usecols=["Semana","Canal_ID","Ruta_SAK","Cliente_ID","Producto_ID","Demanda_uni_equil","Agencia_ID"]
,dtype  = {'Semana' : 'int8',
                              'Agencia_ID' :'int32',
                              'Canal_ID' : 'int8',
                              'Ruta_SAK' : 'int32',
                              'Cliente-ID' : 'int64',
                              'Producto_ID':'int32',
                              'Demanda_uni_equil':'int32'}
)

#tp.get_chunk(5)

df_train = pd.concat(tp, ignore_index=True)


# In[ ]:


df_train.head(3)
df_train.tail(3)


# In[ ]:


#https://www.kaggle.com/armalali/grupo-bimbo-inventory-demand/benchmark-medians/code
prod_median_tab = df_train.groupby('Producto_ID').agg({'Demanda_uni_equil': np.median})


# In[ ]:


prod_median_tab.head(3)


# In[ ]:


prod_median_tab2 = df_train.groupby(['Producto_ID', 'Cliente_ID']).agg({'Demanda_uni_equil': np.median})


# In[ ]:


prod_median_tab2.head(3)


# In[ ]:


global_median = np.median(df_train['Demanda_uni_equil'])


# In[ ]:


print global_median


# In[ ]:


prod_median_dict2 = prod_median_tab2.to_dict()


# In[ ]:


prod_median_dict = prod_median_tab.to_dict()


# In[ ]:



def gen_output(key):
    key = tuple(key)
    try:
        val = prod_median_dict2['AdjDemand'][key]
        try:
            val = prod_median_dict['AdjDemand'][key[0]]
        except:
            val = global_median
    except:
        val = global_median
    return val
    
df_test = pd.read_csv('../input/test.csv')
df_test.columns = ["id", "Semana","Canal_ID","Ruta_SAK","Cliente_ID","Producto_ID","Agencia_ID"]


# In[ ]:



#Generating the output
df_test['Demanda_uni_equil'] = df_test[['Producto_ID', 'Cliente_ID']].                apply(lambda x:gen_output(x), axis=1)
df_submit = df_test[['id', 'Demanda_uni_equil']]
df_submit = df_submit.set_index('id')
df_submit.to_csv('naive_product_client_median.csv')

