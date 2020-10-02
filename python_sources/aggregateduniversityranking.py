#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_shanghai = pd.read_csv('../input/shanghaiData.csv')
df_cwur = pd.read_csv('../input/cwurData.csv')
df_times = pd.read_csv('../input/timesData.csv')


# In[ ]:


df_shanghai=df_shanghai.rename(columns = {'total_score':'score_shanghai'})
df_cwur=df_cwur.rename(columns = {'institution':'university_name'})
df_cwur=df_cwur.rename(columns = {'score':'score_cwur'})
df_times=df_times.rename(columns = {'total_score':'score_times'})


# In[ ]:


df_cwur.head()


# In[ ]:


df_s = df_shanghai[['university_name', 'score_shanghai', 'year']]
df_c = df_cwur[['university_name', 'score_cwur', 'year']]
df_t = df_times[['university_name', 'score_times', 'year']]


# In[ ]:


test = df_s.groupby('university_name')['score_shanghai'].mean()


# In[ ]:


shanghai = df_s.loc[df_s['year'] == 2015]
cwur = df_c.loc[df_c['year'] == 2015]
times = df_t.loc[df_t['year'] == 2015]


# In[ ]:


mShanghai = shanghai['score_shanghai'].mean()


# In[ ]:


shanghai['score_shanghai'].fillna(mShanghai, inplace=True)


# In[ ]:


times.loc[times['score_times'] == '-', 'score_times'] = 75


# In[ ]:


tmp = pd.merge(shanghai, cwur, on='university_name')
data = pd.merge(tmp, times, on='university_name')
data = data[['university_name', 'score_shanghai', 'score_cwur', 'score_times']]


# In[ ]:


data['score_times'] = data['score_times'].apply(float)


# In[ ]:


data['total_score'] = (data['score_shanghai'] + data['score_cwur'] + data['score_times'])/3


# In[ ]:


data = data.sort('total_score', ascending=0)


# In[ ]:


data.head()


# In[ ]:


data.to_csv('data.csv', sep='\t')

