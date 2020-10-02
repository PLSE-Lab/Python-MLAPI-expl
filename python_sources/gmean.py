#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob

from scipy.stats import rankdata
from scipy.stats import describe
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Stacking Approach using GMEAN

# In[ ]:


ls '../input/'


# In[ ]:


LABELS = ["is_click"]
all_files = glob.glob("../input/wnssubmits/*.csv")
all_files = [i for i in all_files if 'XX' not in i and 'Enet' not in i]
all_files


# In[ ]:


scores = np.zeros(len(all_files))
for i in range(len(all_files)):
    print(i,all_files[i])
    scores[i] = float(all_files[i].replace('_','-').split("-")[1].replace('.csv',''))


# In[ ]:


top10 = scores.argsort()[-10:][::-1]
for i, f in enumerate(top10):
    print(i,all_files[f])


# In[ ]:


outs = [pd.read_csv(all_files[f], index_col=0) for f in top10]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "m" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
#concat_sub.reset_index(inplace=True)


# In[ ]:


concat_sub.index


# In[ ]:


# check correlation
corr = concat_sub.iloc[:,1:].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(len(cols)+2, len(cols)+2))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr,mask=mask,cmap='prism',center=0, linewidths=1,
            annot=True,fmt='.4f', cbar_kws={"shrink":.5})


# # Weighted GMEAN by inverse correlation

# In[ ]:


rank = np.tril(concat_sub.iloc[:,1:].corr().values,-1)
m = (rank>0).sum() - (rank>0.99).sum()
m_gmean, s = 0, 0
for n in range(min(rank.shape[0],m)):
    mx = np.unravel_index(rank.argmin(), rank.shape)
    w = (m-n)/(m+n/8)
    m_gmean += w*(np.log(concat_sub.iloc[:,mx[0]+1])+np.log(concat_sub.iloc[:,mx[1]+1]))/2
    s += w
    rank[mx] = 1
m_gmean = np.exp(m_gmean/s)


# In[ ]:


m_gmean = (m_gmean-m_gmean.min())/(m_gmean.max()-m_gmean.min())
describe(m_gmean)


# In[ ]:


m_gmean.values


# In[ ]:


concat_sub['is_click'] = m_gmean.values
concat_sub['impression_id'] = m_gmean.index
concat_sub[['impression_id','is_click']].to_csv('stack_gmean.csv',index=False)


# In[ ]:


# out = pd.DataFrame()
# out['is_click'] = m_gmean
# out['impression_id'] = concat_sub.index
# out[['impression_id','is_click']].to_csv('stack_gmean.csv',index=False)


# In[ ]:


from IPython.display import FileLink, FileLinks
FileLinks('.') #lists all downloadable files on server

