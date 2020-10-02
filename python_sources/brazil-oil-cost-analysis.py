#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_time=pd.read_csv('../input/2004-2019.tsv',sep='\t')
cols=list(data_time.columns)
print(data_time.shape)
for col in cols:
    print(data_time[col].count())

    


# In[ ]:


for col in cols:
    print(col)
print(data_time.head())


# In[ ]:


get_ipython().system('pip install googletrans')


# In[ ]:


import googletrans
from googletrans import Translator
translator=Translator()


# In[ ]:


cols_en=[]
for col in cols:
    translation=translator.translate(col)
    text_col=translation.text
    cols_en.append(text_col)
print(cols_en)    


# In[ ]:


data_time.columns=cols_en
data_time=data_time.drop('Unnamed: 0',axis=1)
print(data_time['INITIAL DATE'].unique().tolist())
print(data_time['DATA FINAL'].unique().tolist())


# observe that each row is a data for 7 days. It starts from 2014,9th may to 2019, 23th june. So, the data is weekly summary for 5 years, 1 month.

# In[ ]:


print(data_time['UNIT OF MEASUREMENT'].unique().tolist())


# In[ ]:


regions=data_time['REGION'].unique().tolist()
states=data_time['STATE'].unique().tolist()
print(len(regions))
print(len(states))


# In[ ]:


def columns_describer(column,data=data_time):
    col_list=data[column].unique().tolist()
    print('for column',column,'lengths of uniques are:',len(col_list))


# In[ ]:


for col in list(data_time.columns):
    columns_describer(col)


# In[ ]:


cols=list(data_time.columns)
for i in range(len(cols)-5):
    plt.figure(figsize=(6,9))
    plt.xlabel(cols[i])
    plt.hist(data_time[cols[i]].tolist())

