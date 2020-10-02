#!/usr/bin/env python
# coding: utf-8

# # THIS KERNAL IS BLEND OF So awesome kernels present Right now
# # Vote if you love blend. 
# # 214 forks and 55 votes !o_o!
# 
# ![](http://2.bp.blogspot.com/-rO_aqkor58M/XqZHiAN1v9I/AAAAAAAAG5M/VkEfBStZS90Q8nT2R9oPRT9mspMItX9WACK4BGAYYCw/s1600/fork.PNG)
# 
# ## Kernels used comming from these awesome people:
# 
# [[TPU-Inference] Super Fast XLMRoberta](https://www.kaggle.com/shonenkov/tpu-inference-super-fast-xlmroberta)
# 
# [Jigsaw TPU: BERT with Huggingface and Keras](https://www.kaggle.com/miklgr500/jigsaw-tpu-bert-with-huggingface-and-keras)
# 
# [inference of bert tpu model ml w/ validation](https://www.kaggle.com/abhishek/inference-of-bert-tpu-model-ml-w-validation)

# # phase 1 [Ensemble]

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os


# In[ ]:


#submission1 = pd.read_csv('/kaggle/input/009383/submission (17).csv')
#submission2 = pd.read_csv('/kaggle/input/009354/submission (25).csv')
submission1 = pd.read_csv('/kaggle/input/tfidf/submission (27).csv')
submission2 = pd.read_csv('../input/tpuinference-super-fast-xlmroberta/submission (47).csv')


# # Hist Graph of scores

# In[ ]:


sns.set()
plt.hist(submission1['toxic'],bins=100)
plt.show()


# In[ ]:


sns.set()
plt.hist(submission2['toxic'],bins=100)
plt.show()


# In[ ]:


s = []
s1 = submission1['toxic'].tolist()
s2 = submission2['toxic'].tolist()
for i in range(len(s1)):
    if s1[i]>0.5 and s2[i]<0.5:
        s.append(s1[i])
    else: s.append(s2[i])
submission1['toxic'] = s


# In[ ]:


#submission1['toxic'] = submission1['toxic']*0.05 + submission2['toxic']*0.95


# In[ ]:


submission1.to_csv('submission.csv', index=False)


# # phase 2 [Stacking]

# In[ ]:


sub_path = "../input/blending"
all_files = os.listdir(sub_path)
all_files


# In[ ]:


outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "jigsaw" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)
concat_sub.head()
ncol = concat_sub.shape[1]


# In[ ]:


# check correlation
concat_sub.iloc[:,1:ncol].corr()


# In[ ]:


corr = concat_sub.iloc[:,1:7].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:


# get the data fields ready for stacking
concat_sub['jigsaw_max'] = concat_sub.iloc[:, 1:ncol].max(axis=1)
concat_sub['jigsaw_min'] = concat_sub.iloc[:, 1:ncol].min(axis=1)
concat_sub['jigsaw_mean'] = concat_sub.iloc[:, 1:ncol].mean(axis=1)
concat_sub['jigsaw_median'] = concat_sub.iloc[:, 1:ncol].median(axis=1)


# In[ ]:


concat_sub.describe()


# In[ ]:


cutoff_lo = 0.7
cutoff_hi = 0.3


# In[ ]:


concat_sub['toxic'] = concat_sub['jigsaw_mean']
concat_sub[['toxic']].to_csv('submission2.csv', 
                                        index=False, float_format='%.6f')


# In[ ]:


concat_sub['toxic'] = concat_sub['jigsaw_median']
concat_sub[['toxic']].to_csv('submission1.csv', 
                                        index=False, float_format='%.6f')


# In[ ]:


concat_sub['toxic'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 1, 
                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),
                                             0, concat_sub['jigsaw_median']))
concat_sub[['toxic']].to_csv('submission3.csv', 
                                        index=False, float_format='%.6f')


# In[ ]:


concat_sub['toxic'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 
                                    concat_sub['jigsaw_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),
                                             concat_sub['jigsaw_min'], 
                                             concat_sub['jigsaw_mean']))
concat_sub[['toxic']].to_csv('submission4.csv', 
                                        index=False, float_format='%.6f')


# In[ ]:


concat_sub['toxic'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 
                                    concat_sub['jigsaw_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),
                                             concat_sub['jigsaw_min'], 
                                             concat_sub['jigsaw_median']))
concat_sub[['toxic']].to_csv('submission5.csv', 
                                        index=False, float_format='%.6f')

