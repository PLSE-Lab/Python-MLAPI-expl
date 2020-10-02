#!/usr/bin/env python
# coding: utf-8

# Human brain research is among the most complex areas of study for scientists. We know that age and other factors can affect its function and structure, but more research is needed into what specifically occurs within the brain. With much of the research using MRI scans, data scientists are well positioned to support future insights. In particular, neuroimaging specialists look for measurable markers of behavior, health, or disorder to help identify relevant brain regions and their contribution to typical or symptomatic effects.
# 
# In this competition, you will predict multiple assessments plus age from multimodal brain MRI features. You will be working from existing results from other data scientists, doing the important work of validating the utility of multimodal features in a normative population of unaffected subjects. Due to the complexity of the brain and differences between scanners, generalized approaches will be essential to effectively propel multimodal neuroimaging research forward.
# 
# The Tri-Institutional Georgia State University/Georgia Institute of Technology/Emory University Center for Translational Research in Neuroimaging and Data Science (TReNDS) leverages advanced brain imaging to promote research into brain health. The organization is focused on developing, applying and sharing advanced analytic approaches and neuroinformatics tools. Among its software projects are the GIFT and FIT neuroimaging toolboxes, the COINS data management system, and the COINSTAC toolkit for federated learning, all aimed at supporting data scientists and other neuroimaging researchers.

# Thanks to the contributors to this project
# 
# * Ashish Gupta (https://www.kaggle.com/roydatascience) - Created .1593 Model
# * Mukharbek Organokov (https://www.kaggle.com/muhakabartay) - Created .1590 kernel with high CV
# * Yassine Alouini (https://www.kaggle.com/yassinealouini) - Created .1595 kernel 
# * Vlad (https://www.kaggle.com/bonhart) - Created .1593 kernel

# We further ensembled our 4 models. The Final solution gave us a bronze medal on Kaggle.

# In[ ]:


import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from subprocess import check_output


# In[ ]:


sub_path = "../input/final-ensemble"
all_files = os.listdir(sub_path)
all_files


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "var" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)
concat_sub.head()
ncol = concat_sub.shape[1]


# In[ ]:


# check correlation
concat_sub.iloc[:,1:ncol].corr()


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


# get the data fields ready for stacking
concat_sub['m_max'] = concat_sub.iloc[:, 1:ncol].max(axis=1)
concat_sub['m_min'] = concat_sub.iloc[:, 1:ncol].min(axis=1)
concat_sub['m_median'] = concat_sub.iloc[:, 1:ncol].median(axis=1)
concat_sub['m_mean'] = concat_sub.iloc[:, 1:ncol].mean(axis=1)


# G-Mean with low average correlation

# In[ ]:


cutoff_lo = 0.8
cutoff_hi = 0.2


# # Mean Stacking

# In[ ]:


concat_sub['Predicted'] = concat_sub['m_mean']
concat_sub[['Id','Predicted']].to_csv('stack_mean.csv', 
                                        index=False, float_format='%.6f')


# # Median Stacking

# In[ ]:


concat_sub['Predicted']  = concat_sub['m_median']
concat_sub[['Id','Predicted']].to_csv('stack_median.csv', 
                                        index=False, float_format='%.6f')


# # Pushout + Median Stacking
# >* Pushout strategy is bit aggresive

# In[ ]:


concat_sub['Predicted']  = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 1, 
                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),
                                             0, concat_sub['m_median']))
concat_sub[['Id','Predicted']].to_csv('stack_pushout_median.csv', 
                                        index=False, float_format='%.6f')


# # MinMax + Mean Stacking
# >* MinMax seems more gentle and it outperforms the previous one

# In[ ]:


concat_sub['Predicted']  = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 
                                    concat_sub['m_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),
                                             concat_sub['m_min'], 
                                             concat_sub['m_mean']))
concat_sub[['Id','Predicted']].to_csv('stack_minmax_mean.csv', 
                                        index=False, float_format='%.6f')


# # MinMax + Median Stacking

# In[ ]:


concat_sub['Predicted'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 
                                    concat_sub['m_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),
                                             concat_sub['m_min'], 
                                             concat_sub['m_median']))
concat_sub[['Id','Predicted']].to_csv('stack_minmax_median.csv', 
                                        index=False, float_format='%.6f')

