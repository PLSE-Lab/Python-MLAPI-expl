#!/usr/bin/env python
# coding: utf-8

# # Human Freedom Index

# In this kernel we will analyse the Human Freedom Index in different countries over many years

# ### Importing Data and Libraries

# In[ ]:


# Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Getting the dataset
df = pd.read_csv('../input/hfi_cc_2018.csv')


# In[ ]:


df.head()


# In[ ]:


df.describe()


# ### Visualizing Data

# Let's first analyse the countries on the basis of the region which they belong to. From the plot we can observe that most of the countries in the dataset are from Sub-Saharan Africa and Latin America & the Caribbean

# In[ ]:


p = sns.countplot(data=df, x="region")
_ = plt.setp(p.get_xticklabels(), rotation=90)


# Let's analyse data for nations who are the part of BRICS association.(Brazil, Russia, India, China and South Africa)

# In[ ]:


data_brz = df.loc[df.loc[:,'countries']=='Brazil',:]
data_rus = df.loc[df.loc[:,'countries']=='Russia',:]
data_ind = df.loc[df.loc[:,'countries']=='India',:]
data_chn = df.loc[df.loc[:,'countries']=='China',:]
data_sa = df.loc[df.loc[:,'countries']=='South Africa',:]


# The plot below shows the Human Freedom Score over the past years

# In[ ]:


_ = plt.plot('year', 'hf_score', data=data_brz)
_ = plt.plot('year', 'hf_score', data=data_rus)
_ = plt.plot('year', 'hf_score', data=data_ind)
_ = plt.plot('year', 'hf_score', data=data_chn)
_ = plt.plot('year', 'hf_score', data=data_sa)
_ = plt.legend(('Brazil', 'Russia', 'India', 'China', 'South Africa'))


# The plot below shows the Human Freedom Rank of BRICS nations over the past years

# In[ ]:


_ = plt.plot('year', 'hf_rank', data=data_brz)
_ = plt.plot('year', 'hf_rank', data=data_rus)
_ = plt.plot('year', 'hf_rank', data=data_ind)
_ = plt.plot('year', 'hf_rank', data=data_chn)
_ = plt.plot('year', 'hf_rank', data=data_sa)
_ = plt.legend(('Brazil', 'Russia', 'India', 'China', 'South Africa'),loc='upper right')


# The plot below shows the economic freedom score of BRICS nations in the past several years

# In[ ]:


_ = plt.plot('year', 'ef_score', data=data_brz)
_ = plt.plot('year', 'ef_score', data=data_rus)
_ = plt.plot('year', 'ef_score', data=data_ind)
_ = plt.plot('year', 'ef_score', data=data_chn)
_ = plt.plot('year', 'ef_score', data=data_sa)
_ = plt.legend(('Brazil', 'Russia', 'India', 'China', 'South Africa'),loc='upper right')


# The plot below shows the economic freedom score of BRICS nations in the past several years
# 

# In[ ]:


_ = plt.plot('year', 'ef_rank', data=data_brz)
_ = plt.plot('year', 'ef_rank', data=data_rus)
_ = plt.plot('year', 'ef_rank', data=data_ind)
_ = plt.plot('year', 'ef_rank', data=data_chn)
_ = plt.plot('year', 'ef_rank', data=data_sa)
_ = plt.legend(('Brazil', 'Russia', 'India', 'China', 'South Africa'),loc='upper right')

