#!/usr/bin/env python
# coding: utf-8

# 
# # More To Come. Stay Tuned. !!
# If there are any suggestions/changes you would like to see in the Kernel please let me know :). Appreciate every ounce of help!
# 
# **This notebook will always be a work in progress.** Please leave any comments about further improvements to the notebook! Any feedback or constructive criticism is greatly appreciated!. **If you like it or it helps you , you can upvote and/or leave a comment :).**
# 

# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1095143%2F47c74960e6540f11287e1e271438e029%2FTReNDS.png?generation=1587603283379241&alt=media)

# 
# - <a href='#1'>1. Introduction</a>  
# - <a href='#2'>2. Retrieving the Data</a>
#      - <a href='#2-1'>2.1 Load libraries</a>
#      - <a href='#2-2'>2.2 Read the Data</a>
# - <a href='#3'>3. Glimpse of Data</a>
#      - <a href='#3-1'>3.1 Overview of tables</a>
#      - <a href='#3-2'>3.2 Statistical overview of the Data</a>
# - <a href='#4'>4. Check for missing data</a>
# - <a href='#5'>5. Data Exploration</a>
#     - <a href='#5-1'>5.1 Distribution of input variables in loading_data</a>
#     - <a href='#5-2'>5.2 Distribution of target variables in train data</a>
# - <a href='#6'>6. Sample submissions</a>
#     - <a href='#6-1'>6.1 Baseline submission</a>
#     - <a href='#6-2'>6.2 Mean submission</a>

# # <a id='1'>1. Introduction</a>

# In this competition, you will predict multiple assessments plus age from multimodal brain MRI features. You will be working from existing results from other data scientists, doing the important work of validating the utility of multimodal features in a normative population of unaffected subjects. Due to the complexity of the brain and differences between scanners, generalized approaches will be essential to effectively propel multimodal neuroimaging research forward.

# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1537731%2Fa5fdbe17ca91e6713d2880887232c81a%2FScreen%20Shot%202019-12-09%20at%2011.25.31%20AM.png?generation=1575920121028151&alt=media)

#  # <a id='2'>2. Retrieving the Data</a>

#  ## <a id='2-1'>2.1 Load libraries</a>

# In[ ]:


import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
#import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()

# Venn diagram
from matplotlib_venn import venn2
import re
import nltk
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
eng_stopwords = stopwords.words('english')
import gc

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# In[ ]:


import os
bp = '/kaggle/input/trends-assessment-prediction'
print(os.listdir(bp))


# # <a id='2-2'>2.2 Reading Data</a>

# In[ ]:


print('Reading data...')
loading_data = pd.read_csv(bp+'/loading.csv')
train_data = pd.read_csv(bp+'/train_scores.csv')
sample_submission = pd.read_csv(bp+'/sample_submission.csv')
print('Reading data completed')


# In[ ]:


print('Size of loading_data', loading_data.shape)
print('Size of train_data', train_data.shape)
print('Size of sample_submission', sample_submission.shape)
print('test size:', len(sample_submission)/5)


# # <a id='3'>3. Glimpse of Data</a>

# ### There are 5877 training data and 5877 test data. 
# 
# ### For submission, we need to fill five rows for each entry (5877x5 rows):
# 
# > age	
# 
# > domain1_var1
# 
# > domain1_var2
# 
# > domain2_var1
# 
# > domain2_var2

# ## <a id='3-1'>3.1 Overview of tables</a>

# **loading_data**

# In[ ]:


display(loading_data.head())
display(loading_data.describe())


# **train_data**
# 
# 
# ##### This is actually only the target values for the training data set. The real training data is in loading data (partially). So, I will only focus on that file.

# In[ ]:


display(train_data.head())
display(train_data.describe())


# **sample_submission**

# In[ ]:


sample_submission.head()


# **Target variables**

# In[ ]:


targets = list(train_data.columns[1:])
targets


# # <a id='4'> 4 Check for missing data</a>

# **checking missing data in train_data **

# In[ ]:


# checking missing data
total = train_data.isnull().sum().sort_values(ascending = False)
percent = (train_data.isnull().sum()/train_data.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train_data.head()


# **checking missing data in loading_data **

# In[ ]:


# checking missing data
total = loading_data.isnull().sum().sort_values(ascending = False)
percent = (loading_data.isnull().sum()/loading_data.isnull().count()*100).sort_values(ascending = False)
missing_test_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_test_data.head()


# # <a id='5'>5. Data Exploration</a>

# # <a id='5-1'>5.1 Distribution of input variables in loading_data</a>

# In[ ]:


targets= loading_data.columns[1:]
fig, axes = plt.subplots(6, 5, figsize=(18, 15))
axes = axes.ravel()
bins = np.linspace(-0.05, 0.05, 20)

for i, col in enumerate(targets):
    ax = axes[i]
    sns.distplot(loading_data[col], label=col, kde=False, bins=bins, ax=ax)
    # ax.set_title(col)
#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 6079])
plt.tight_layout()
plt.show()
plt.close()


# # <a id='5-2'>5.2 Distribution of target variables in train data</a>

# In[ ]:


targets= train_data.columns[1:]
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
axes = axes.ravel()
bins = np.linspace(0, 100, 20)

for i, col in enumerate(targets):
    ax = axes[i]
    sns.distplot(train_data[col], label=col, kde=False, bins=bins, ax=ax)
    # ax.set_title(col)
#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 6079])
plt.tight_layout()
plt.show()
plt.close()


# # <a id='6'>6. Sample submissions</a>

# # <a id='6-1'>6.1 Baseline submission</a>

# In[ ]:


sample_submission.to_csv('sample_submission.csv', index=False)


# # <a id='6-2'>6.2 Mean submission</a>

# In[ ]:


sub1 = sample_submission.copy()
for i in range(int(len(sub1)/5)):
    sub1.at[i*5+0, 'Predicted']= 50.034068
    sub1.at[i*5+1, 'Predicted']= 51.474692
    sub1.at[i*5+2, 'Predicted']= 59.244132 
    sub1.at[i*5+3, 'Predicted']= 47.325130 
    sub1.at[i*5+4, 'Predicted']= 51.905658 
sub1.to_csv('mean_submission.csv', index=False)


# # More To Come. Stay Tuned. !!
