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


get_ipython().system('pip install statsmodels')


# In[ ]:


import statsmodels.imputation.mice as st


# In[ ]:


import pandas as pd
weather= pd.read_csv("../input/hourly-weather-surface-brazil-southeast-region/sudeste.csv")
path="../input/hourly-weather-surface-brazil-southeast-region/sudeste.csv"


# In[ ]:


df1 = weather.iloc[:int(len(weather)/2)]
df2 = weather.iloc[int(len(weather)/2):]


# In[ ]:


df1.info()


# In[ ]:


reduce_df1=df1.loc[:, df1.isna().mean() < .07]


# In[ ]:


reduce_df1.info()


# In[ ]:


imp=st.MICEData(reduce_df1)


# In[ ]:


#reduce_df1.columns[weather.isnull().mean() < 0.8]
#weather[weather.columns[weather.isnull().mean() < 0.8]]
#weather.info()


# In[ ]:


"""thresh = len(weather) * .02
print(weather)
weather_new = weather.dropna(thresh = thresh, axis = 1, inplace = True)
print(weather_new)"""


# In[ ]:


#print (weather.isin([' ','NULL',0]))


# In[ ]:


corr_matrix = reduce_df1.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))


# In[ ]:


cols = reduce_df1.columns
num_cols = reduce_df1._get_numeric_data().columns
cat_data=list(set(cols) - set(num_cols))
con_data=list(set(num_cols))


# In[ ]:


cat_data


# In[ ]:


from fastai.tabular import * 


# In[ ]:


procs = [FillMissing, Categorify, Normalize]


# In[ ]:


valid_idx = range(len(reduce_df1)-2000, len(reduce_df1))
dep_var = 'wsid'


# In[ ]:


data = TabularDataBunch.from_df(path,reduce_df1, dep_var, valid_idx=valid_idx, procs=procs, bs=64, cat_names=cat_data)


# In[ ]:


print(type(data))


# In[ ]:


print(data.train_ds.cont_names)
print(data.train_ds.cat_names)


# In[ ]:


learn = tabular_learner(data, layers=[1000,500], metrics=accuracy)


# In[ ]:


learn


# In[ ]:


learn.fit_one_cycle(1, 2.5e-2)

