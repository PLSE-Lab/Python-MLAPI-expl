#!/usr/bin/env python
# coding: utf-8

# ## Data analysis of naukri.com

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


np.sum(df.isnull())


# ### Popular roles 

# In[ ]:


sns.set_style('white')
sns.set_context('notebook')
df_copy = df.copy()
df_copy = pd.DataFrame(df_copy['Role'].value_counts()[0:11])
sns.barplot(y = df_copy.index , x = df_copy['Role'] , color = sns.color_palette()[0])


# #### Software Developer clearly has highest number of jobs 

# ### City wise job distribution

# In[ ]:


df_copy = df.copy()
df_copy = pd.DataFrame(df_copy['Location'].value_counts()[0:10])
sns.barplot(y = df_copy.index , x = df_copy['Location'] , color = sns.color_palette()[0])


# #### Bengalura has higgest number of job opnings

# ### City wise role distributions 

# In[ ]:


j = 0
fig, axes = plt.subplots(5, 2 , figsize=[15,15])
for i in df_copy.index:
    df_new = df[df['Location'] == i]
    df_new= pd.DataFrame(df_new['Role'].value_counts()[0:11])
    sns.barplot(y = df_new.index , x = df_new['Role'] , color = sns.color_palette()[0] , ax = axes.flatten()[j])
    fig.subplots_adjust(hspace=0.5)    
    fig.subplots_adjust(wspace=1)
    axes.flatten()[j].set_title(i)
    j+=1
    
    
    


# #### Software developer is popular in every city except for kolkata.

# ### Popular industries

# In[ ]:


df_copy = df.copy()
df_copy = pd.DataFrame(df_copy['Industry'].value_counts()[0:10])
sns.barplot(y = df_copy.index , x = df_copy['Industry'] , color = sns.color_palette()[0])


# In[ ]:


df_copy = df.copy()
j = 0
fig, axes = plt.subplots(5, 2 , figsize=[15,15])
df_copy = pd.DataFrame(df_copy['Location'].value_counts()[0:10])
for i in df_copy.index:
    df_new = df[df['Location'] == i]
    df_new= pd.DataFrame(df_new['Industry'].value_counts()[0:11])
    sns.barplot(y = df_new.index , x = df_new['Industry'] , color = sns.color_palette()[0] , ax = axes.flatten()[j])
    fig.subplots_adjust(hspace=0.5)    
    fig.subplots_adjust(wspace=1)
    axes.flatten()[j].set_title(i)
    j+=1
    


# #### IT-Software and software services have higgest job opnings accross all the cities

# ### Top Skills 

# In[ ]:


df_copy = df.copy()
df_copy = pd.DataFrame(df_copy['Key Skills'].value_counts()[0:10])
sns.barplot(y = df_copy.index , x = df_copy['Key Skills'] , color = sns.color_palette()[0])


# ### Experiance requirements 

# In[ ]:


df['Job Experience Required'] = df['Job Experience Required'].str.replace('yrs' , '')
df['Job Experience Required'] = df['Job Experience Required'].str.replace('Years' , '')
df['min_year'] = df['Job Experience Required'].str.split('-').str[0]


# In[ ]:


df_copy = df.copy()
df_copy = pd.DataFrame(df_copy['min_year'].value_counts()[0:10])
sns.barplot(x = df_copy.index , y = df_copy['min_year'] , color = sns.color_palette()[0] , order = df_copy.index)


# #### So clearly comapines wants atleast 1-2 years of experience.
