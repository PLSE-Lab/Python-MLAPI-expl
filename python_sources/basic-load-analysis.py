#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 100)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# #### Let's start with the candidates table. Let's load it & then find out raised contributions by party

# In[ ]:


candidates = pd.read_csv('/kaggle/input/campaign-contributions-19902016/candidates.csv')
candidates.head()


# In[ ]:


df = candidates.groupby('party').agg({'raised_from_pacs': 'sum', 'raised_from_individuals': 'sum', 'raised_total': 'sum'})
df.plot(kind='bar')


# #### Individual contributions is a large table & loading it all in memory will fail.
# 
# Here are 2 options to load & use this data :

# In[ ]:


# Just load the first X rows (this is useful at the start of the analysis)
indiv_contribs_chunk_reader = pd.read_csv('/kaggle/input/campaign-contributions-19902016/individual_contributions.csv', iterator=True)
indiv_contribs_chunk = indiv_contribs_chunk_reader.get_chunk(10000)
indiv_contribs_chunk.head()


# In[ ]:


# Load X rows 1 chunk at a time, run analysis 1 chunk at a time to combine the results into a final dataframe
indiv_contribs_chunk_iter = pd.read_csv('/kaggle/input/campaign-contributions-19902016/individual_contributions.csv', chunksize=500000)
total_amount = 0
for indiv_contribs_chunk in indiv_contribs_chunk_iter:
    print(indiv_contribs_chunk.shape)
    total_amount += np.sum(indiv_contribs_chunk['amount'])
total_amount


# #### Let's plot how the PAC contributions have aged for Republicans vs Democrats

# In[ ]:


pac_to_pacs = pd.read_csv('/kaggle/input/campaign-contributions-19902016/pac_to_pacs.csv')
pac_to_pacs.head()


# In[ ]:


pac_to_pacs_cycle_party = pac_to_pacs.groupby(['cycle', 'party']).agg({'amount': 'sum'}).reset_index().pivot_table(values='amount', columns='party', index='cycle')
pac_to_pacs_cycle_party[['D', 'R']].plot()


# #### Let's plot the evolution of PAC contributions over time

# In[ ]:


pacs = pd.read_csv('/kaggle/input/campaign-contributions-19902016/pacs.csv')
pacs.head()


# In[ ]:


pacs.groupby('cycle').agg({'amount': 'sum'}).plot()


# In[ ]:




