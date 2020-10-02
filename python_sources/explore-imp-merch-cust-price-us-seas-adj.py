#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import seaborn as sns
# Any results you write to the current directory are saved as output.


# In[ ]:


pd.__version__


# In[ ]:


print(os.listdir('../input')[:2])


# In[ ]:


imcpusa0 = pd.read_excel('../input/Imports Merchandise Customs Price US seas. adj..xlsx', sheet_name=0)


# In[ ]:


imcpusa0.head()


# In[ ]:


imcpusa0.shape


# In[ ]:


imcpusa0 = imcpusa0.dropna(axis=0, how='all')


# In[ ]:


imcpusa0.head()


# In[ ]:


imcpusa0_corr = imcpusa0.corr()


# In[ ]:


imcpusa0_corr.head()


# In[ ]:


mask = np.zeros_like(imcpusa0_corr, dtype=np.bool)
mask[np.tril_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(imcpusa0_corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:


imcpusa0_corr[mask] = None


# In[ ]:


imcpusa0_corr.head()


# In[ ]:


imcpusa0_corr_melted = pd.melt(imcpusa0_corr.reset_index(), id_vars='index', value_vars=imcpusa0_corr.columns.tolist())


# In[ ]:


imcpusa0_corr_melted.head()


# In[ ]:


imcpusa0_corr_melted = imcpusa0_corr_melted.dropna()


# In[ ]:


imcpusa0_corr_melted.head()


# In[ ]:


imcpusa0_corr_melted = imcpusa0_corr_melted.sort_values('value')


# In[ ]:


imcpusa0_corr_melted.head()


# In[ ]:


imcpusa0_corr_melted.tail()


# In[ ]:


most_neg = ['Europe & Central Asia developing', 'High Income: Non-OECD']
sns.pairplot(imcpusa0.fillna(0).loc[:, most_neg]);


# In[ ]:


most_pos = ['Developing Countries', 'Middle Income Countries']
sns.pairplot(imcpusa0.fillna(0).loc[:, most_pos]);


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 8))
imcpusa0.loc[:, most_neg].plot(ax=ax);
ax.set(xlabel='Year', ylabel='Price')
ax.grid()


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 8))
imcpusa0.loc[:, most_pos].plot(ax=ax);
ax.set(xlabel='Year', ylabel='Price')
ax.grid()


# In[ ]:


imcpusa1 = pd.read_excel('../input/Imports Merchandise Customs Price US seas. adj..xlsx', sheet_name=1)


# In[ ]:


imcpusa1.head()


# In[ ]:


imcpusa1.shape


# In[ ]:


imcpusa1 = imcpusa1.dropna(axis=0, how='all')


# In[ ]:


imcpusa1.head()


# In[ ]:


imcpusa1.index.max()


# In[ ]:


months = pd.date_range(start='1991-01-01', end='2018-08-01', freq='MS')


# In[ ]:


months


# In[ ]:


imcpusa1.shape


# In[ ]:


imcpusa1.index = months


# In[ ]:


imcpusa1_corr = imcpusa1.corr()
mask = np.zeros_like(imcpusa1_corr, dtype=np.bool)
mask[np.tril_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(imcpusa1_corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});


# In[ ]:


imcpusa1_corr[mask] = None
imcpusa1_corr_melted = pd.melt(imcpusa1_corr.reset_index(), id_vars='index', value_vars=imcpusa1_corr.columns.tolist())
imcpusa1_corr_melted = imcpusa1_corr_melted.dropna()
imcpusa1_corr_melted = imcpusa1_corr_melted.sort_values('value')


# In[ ]:


imcpusa1_corr_melted.head()


# In[ ]:


most_neg = ['High Income: Non-OECD', 'Slovakia', 'Philippines', 'Turkey']


# In[ ]:


imcpusa1_corr_melted.tail()


# In[ ]:


most_pos = ['Europe & Central Asia developing', 'Turkey', 'Developing Countries', 'Middle Income Countries']


# In[ ]:


sns.pairplot(imcpusa1.fillna(0).loc[:, most_neg]);


# In[ ]:


imcpusa1.loc[:, most_neg].plot(figsize=(15, 10), grid=True, logy=True);


# In[ ]:


sns.pairplot(imcpusa1.fillna(0).loc[:, most_pos]);


# In[ ]:


imcpusa1.loc[:, most_pos].plot(figsize=(15, 10), grid=True, logy=True);


# In[ ]:


imcpusa2 = pd.read_excel('../input/Imports Merchandise Customs Price US seas. adj..xlsx', sheet_name=2)


# In[ ]:


imcpusa2.shape


# In[ ]:


imcpusa2.head()


# In[ ]:


imcpusa2 = imcpusa2.dropna(axis=0, how='all')
imcpusa2.head()


# In[ ]:


imcpusa2.index.max()


# In[ ]:


quarters = pd.date_range(start='1991-01-01', end='2018-09-30', freq='Q')
quarters


# In[ ]:


imcpusa2.shape


# In[ ]:


imcpusa2.index = quarters


# In[ ]:


imcpusa2_corr = imcpusa2.corr()
mask = np.zeros_like(imcpusa2_corr, dtype=np.bool)
mask[np.tril_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(imcpusa2_corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});


# In[ ]:


is_small_columns = imcpusa2_corr.where(imcpusa2_corr < -.6).notnull().any(axis=0)
is_small_rows = imcpusa2_corr.where(imcpusa2_corr < -0.6).notnull().any(axis=1)
neg_corrs2 = is_small_columns.loc[is_small_columns.values].index.tolist()
sns.pairplot(imcpusa2.fillna(0).loc[:, neg_corrs2]);


# In[ ]:


imcpusa2.loc[:, neg_corrs2].plot(figsize=(15, 10), grid=True);


# In[ ]:




