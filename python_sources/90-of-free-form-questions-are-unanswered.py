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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dfree = pd.read_csv('../input/freeFormResponses.csv')


# In[ ]:


def calculate_unanswered(df):
    ratio = df.isnull().sum()/df.shape[0]
    ratio.name = 'Unanswered question ratio'
    df_unanswered = ratio.index.to_series().str.slice(1).str.split('_').apply(pd.Series)
    df_unanswered['Question Number']=df_unanswered[0].astype(int)
    df_unanswered['Unanswered question ratio']=ratio
    df_unanswered=df_unanswered.drop([0,1,2,3], axis=1)
    df_unanswered = df_unanswered.sort_values('Question Number').filter(regex='^(?!.*(Part|34)).*$', axis=0)
    return df_unanswered


# In[ ]:


unanswered = calculate_unanswered(dfree)


# In[ ]:


unanswered.plot(kind='bar', x='Question Number', y='Unanswered question ratio', figsize=(10, 10), color=(0.1, 0.3, 0.7, 0.6));


# In[ ]:




