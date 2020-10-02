#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
os.listdir("../input")


# In[ ]:


all_files = glob.glob("../input/rsna-outputs/*.csv")
all_files


# In[ ]:


all_files_addit = glob.glob("../input/rsna-public-not-checked0/*.csv")
all_files_addit


# In[ ]:


all_files_addit_manoj = glob.glob("../input/inceptionresnetv2manoj/*.csv")
all_files_addit_manoj


# In[ ]:


get_ipython().system('mkdir ../input/alltogether')
get_ipython().system('ls ../input/')


# In[ ]:


get_ipython().system('cp ../input/rsna-public-not-checked0/*.csv ../input/alltogether/')
get_ipython().system('cp ../input/inceptionresnetv2manoj/*.csv ../input/alltogether/')
get_ipython().system('cp ../input/rsna-outputs/*.csv ../input/alltogether/')


# In[ ]:


all_together = glob.glob("../input/alltogether/*.csv")
all_together


# In[ ]:


#outs = [pd.read_csv(f, index_col=0) for f in all_files]
outs = [pd.read_csv(f, index_col=0) for f in all_together]
concat_sub = reduce(lambda left,right: pd.merge(left,right,on='ID'), outs)
#cols = list(map(lambda x: f"{all_files[x].split('/')[3]}", range(len(concat_sub.columns))))
cols = list(map(lambda x: f"{all_together[x].split('/')[3]}", range(len(concat_sub.columns))))
concat_sub.columns = cols
ncol = concat_sub.shape[1]


# In[ ]:


concat_sub.head()


# In[ ]:


# check correlation
corr = concat_sub.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(len(cols)+2, len(cols)+2))

# Draw the heatmap with the mask and correct aspect ratio
_ = sns.heatmap(corr,mask=mask,cmap='prism',center=0, linewidths=1,
                annot=True,fmt='.4f', cbar_kws={"shrink":.2})


# In[ ]:


concat_sub['m_median'] = concat_sub.iloc[:, 1:].median(axis=1)
concat_sub['m_max'] = concat_sub.iloc[:, 1:].max(axis=1)
concat_sub['m_min'] = concat_sub.iloc[:, 1:].min(axis=1)


# In[ ]:


cutoff_lo = 0.8
cutoff_hi = 0.2


# In[ ]:


rank = np.tril(corr.values,-1)
#rank[rank<0.94] = 1
rank[rank<0.92] = 1
m = (rank>0).sum() - (rank>0.97).sum()
m_gmean, s = 0, 0
for n in range(m):
    mx = np.unravel_index(rank.argmin(), rank.shape)
    w = (m-n)/m
    m_gmean += w*(np.log(concat_sub.iloc[:,mx[0]])+np.log(concat_sub.iloc[:,mx[1]]))/2
    s += w
    rank[mx] = 1
m_gmean = np.exp(m_gmean/s)


# In[ ]:


concat_sub['Label'] = m_gmean
concat_sub['m_mean'] = m_gmean
concat_sub['ID'] = concat_sub.index
concat_sub[['ID', 'Label']].to_csv('rsna_simple_gmean.csv', index=False)


# In[ ]:


concat_sub['Label'] = concat_sub['m_median']
concat_sub[['ID', 'Label']].to_csv('stack_median.csv', 
                                        index=False, float_format='%.6f')


# In[ ]:


concat_sub['Label'] = np.where(np.all(concat_sub.iloc[:,1:6] > cutoff_lo, axis=1), 1, 
                                    np.where(np.all(concat_sub.iloc[:,1:6] < cutoff_hi, axis=1),
                                             0, concat_sub['m_median']))
concat_sub[['ID', 'Label']].to_csv('stack_pushout_median.csv', 
                                        index=False, float_format='%.6f')


# In[ ]:


concat_sub['Label'] = np.where(np.all(concat_sub.iloc[:,1:6] > cutoff_lo, axis=1), 
                                    concat_sub['m_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:6] < cutoff_hi, axis=1),
                                             concat_sub['m_min'], 
                                             concat_sub['m_mean']))
concat_sub[['ID', 'Label']].to_csv('stack_minmax_mean.csv', 
                                        index=False, float_format='%.6f')


# In[ ]:


concat_sub['Label'] = np.where(np.all(concat_sub.iloc[:,1:6] > cutoff_lo, axis=1), 
                                    concat_sub['m_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:6] < cutoff_hi, axis=1),
                                             concat_sub['m_min'], 
                                             concat_sub['m_median']))
concat_sub[['ID', 'Label']].to_csv('stack_minmax_median.csv', 
                                        index=False, float_format='%.6f')


# ## Let's see average of all

# In[ ]:


get_ipython().system('ls ../input/alltogether/')


# In[ ]:


ll = 6 # number of files

df_suball = []
for i in range(0,ll):
    df_suball.append(0)
for i in range(0,ll):
    df_suball[i] = pd.read_csv(glob.glob("../input/alltogether/*.csv")[i])


# In[ ]:


print(df_suball[0])


# In[ ]:


df_suball


# In[ ]:


##submission (0.86).csv	submission (0.87).csv	submission (0.078).csv	submission (0.84).csv	RSNA Inception Resnet V2.csv	submission (0.89).csv
concat_sub.head()


# In[ ]:


df_suball[0].loc[df_suball[0]['ID'] == 'ID_000012eaf_any'] #
#df_suball[0].loc[df_suball[0]['ID'] == 'ID_000012eaf_epidural']


# In[ ]:


df_suball[1].loc[df_suball[1]['ID'] == 'ID_000012eaf_any'] # submission (0.87).csv	


# In[ ]:


df_suball[2].loc[df_suball[2]['ID'] == 'ID_000012eaf_any'] # submission (0.078).csv


# In[ ]:


df_suball[3].loc[df_suball[3]['ID'] == 'ID_000012eaf_any'] # submission (0.84).csv


# In[ ]:


df_suball[4].loc[df_suball[4]['ID'] == 'ID_000012eaf_any'] # RSNA Inception Resnet V2.csv 0.077


# In[ ]:


#df_suball[5]['ID'] # submission (0.89).csv
df_suball[5].loc[df_suball[5]['ID'] == 'ID_000012eaf_any']


# # Simple averaging

# In[ ]:


#for i in range(1,ll):
#    df_suball[0]['Label'] += (1/(ll-1))*df_suball[i]['Label']
#    
#df_suball[0]


# # Weights

# In[ ]:


1/(ll-1) # for average


# In[ ]:


# 1,3,5 > 0.85
# 2,4 < 0.80

## this gave same 0.77
#for i in range(1,ll):
#    df_suball[0]['Label'] += df_suball[0]['Label'] + 0.18*df_suball[1]['Label'] + 0.23*df_suball[2]['Label'] + 0.18*df_suball[3]['Label']+ 0.23*df_suball[4]['Label'] + 0.18*df_suball[5]['Label']
#    
#df_suball[0]


# In[ ]:


for i in range(1,ll):
    df_suball[0]['Label'] += 0.15*df_suball[0]['Label'] + 0.15*df_suball[1]['Label'] + 0.20*df_suball[2]['Label'] + 0.15*df_suball[3]['Label']+ 0.20*df_suball[4]['Label'] + 0.15*df_suball[5]['Label']
    
df_suball[0]


# In[ ]:


df_suball[0].to_csv('submission_average.csv', index=False)

