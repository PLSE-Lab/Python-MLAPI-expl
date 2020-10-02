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


# 1st step is to load the required libraries

# In[ ]:


import pandas as pd #pandas is for importing files,data processing
import numpy as np # linear algebra, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import seaborn as sns # data visulaization
import matplotlib.pyplot as plt # library for visualization 
from statsmodels.graphics.tsaplots import plot_acf #visualization for auto-correlation
from statsmodels.graphics.tsaplots import plot_pacf # partial auto correlation


# reading the files, as they as csv thus it will be .csv. Path can be found from the workspace file links.

# In[ ]:


imp_d= pd.read_csv('../input/india-trade-data/2018-2010_import.csv')


# In[ ]:


exp_d=pd.read_csv('../input/india-trade-data/2018-2010_export.csv')


# Instead of wasting time in other thing, lets see if there are any missing values.

# In[ ]:


imp_d.isnull().sum()


# In[ ]:


exp_d.isnull().sum()


# In[ ]:


imp_d.describe()


# In[ ]:


exp_d.describe()


# lets see the data type

# In[ ]:


imp_d.info()


# In[ ]:


exp_d.info()


# We can see that there are 5 variables in each type i.e import and export.
# 
# Replacing the null values with appropriate ones.

# In[ ]:


imp_d=imp_d.dropna()
imp_d=imp_d.reset_index(drop=True)

exp_d=exp_d.dropna()
exp_d=exp_d.reset_index(drop=True)


# In[ ]:


imp_d.isnull().sum()


# In[ ]:


exp_d.isnull().sum()


# All missing values are being removed.

# In[ ]:


import_contri=imp_d['country'].nunique()
import_contri


# In[ ]:


export_contri=exp_d['country'].nunique()
export_contri


# It can be seen that India export to 248 countries whereas import from 241 countries. Though export countries are more, its figure out the total import & export transcation.

# In[ ]:


import_gr=imp_d.groupby(['country','year']).agg({'value':'sum'})
export_gr=exp_d.groupby(['country','year']).agg({'value':'sum'})
export_gr.groupby(['country'])
import_temp=import_gr.groupby(['country']).agg({'value':'sum'})
export_temp=export_gr.groupby(['country']).agg({'value':'sum'}).loc[import_temp.index.values]

data_1=import_gr.groupby(['country']).agg({'value':'sum'}).sort_values(by='value').tail(10)
data_2=export_temp
data_3=data_2-data_1
data_1.column=['Import']
data_2.column=['Export']
data_3.column=['spend/gain']


# In[ ]:


df=pd.DataFrame(index=data_1.index.values)
df['Import']=data_1
df['Export']=data_2
df['spend/gain']=data_3


# In[ ]:


df


# I used spend/gain because it cannot be considered as loss/profit. Spend means the amount spend in import and gain means the amount got during the export.

# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
df.plot(kind='bar',ax=ax)
ax.set_xlabel('Countries')
ax.set_ylabel('Value of transactions (in million US$)')


# In[ ]:


Deficit_=export_gr -import_gr
Time_Series=pd.DataFrame(index=import_gr.index.values)
Time_Series['Import']=import_gr
Time_Series['Export']=export_gr
Time_Series['spend / gain']=Deficit_

Time_Series

fig, ax = plt.subplots(figsize=(15,7))
Time_Series.plot(ax=ax,marker='o')
ax.set_xlabel('Years')
ax.set_ylabel('Value of transactions (in million US$)')


# In[ ]:


Time_Series


# In[ ]:


sns.barplot(x = 'year', y = 'spend / gain', data = Time_Series)
plt.show()


# As per current U.S and China Trade war, India has a chance to increase it export to both the cuntries and others as well. The India can surely be profited by the trade war provided avalibility of manufacturing. It will help to increase the GNP and GDP.  
