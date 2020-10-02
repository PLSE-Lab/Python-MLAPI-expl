#!/usr/bin/env python
# coding: utf-8

# ## An EDAthon from a Fast Kaggler

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system(' cd ../input ;ls')


# In[ ]:


df = pd.read_csv('../input/charlotte-hot-chocolate-15k-2019/Hot Chocolate 15K Results.csv')


# In[ ]:


df.head()


# In[ ]:


df[df.duplicated(['BIB'],False)]


# ### Distribution within some Categorical Features

# In[ ]:


ax = df.AGE.value_counts().reindex(['0-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-100']).plot.bar()
ax.set_ylabel('Pariticpant Count');


# In[ ]:


df.GENDER.value_counts().plot.bar()


# ### Time Features (Pace and Finish Time)

# In[ ]:


df.PACE.dtype, df.TIME.dtype


# In[ ]:


df['PACE_SEC']= df.PACE.apply(lambda x: int(x.split(':')[0])*60+int(x.split(':')[1]))


# In[ ]:


df.head()


# In[ ]:


time_split = lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]) if int(x.split(':')[0])>2 else int(x.split(':')[0])*3600 + int(x.split(':')[1])*60 + int(x.split(':')[2])


# In[ ]:


df['TIME_SEC']= df.TIME.apply(time_split)


# In[ ]:


df.head()


# #### Fastest Cities

# In[ ]:


ax = plt.figure(figsize=(10,5))
ax = df.groupby(['HOMETOWN']).agg(np.mean)['PACE_SEC'].sort_values()[:15].plot.bar(color=plt.cm.Blues_r(np.arange(15)))
ax.set_ylabel('Mean Pace (Seconds)');
ax.set_title('Mean Pace of Participants Hometown');


# In[ ]:


ax = plt.figure(figsize=(10,5))
ax = df.groupby(['HOMETOWN']).agg(np.mean)['TIME_SEC'].sort_values()[:15].plot.bar(color=plt.cm.Blues_r(np.arange(15)))
ax.set_ylabel('Mean Time (Seconds)');


# #### Fastest Ages

# In[ ]:


ax = plt.figure(figsize=(10,5))
ax = df.groupby(['AGE']).agg(np.mean)['PACE_SEC'].sort_values()[:15].plot.bar(color=plt.cm.Blues_r(np.arange(15)))
ax.set_ylabel('Mean Pace (Seconds)');


# In[ ]:


ax = df.boxplot(column=['PACE_SEC'],by=['AGE'],figsize=(10,5));


# #### Fastest Gender (Accidentally Sexist :/)

# In[ ]:


ax = plt.figure(figsize=(10,5))
ax = df.groupby(['GENDER']).agg(np.mean)['PACE_SEC'].sort_values()[:15].plot.bar(color=['b','r'])
ax.set_ylabel('Mean Pace (Seconds)');


# ### Group Paces into a Categorical Variable in Terms of Speed

# In[ ]:


ax = df.PACE_SEC.plot.hist(bins=5)
ax.set_xlabel('Pace (Seconds)');
ax.set_ylabel('Count');


# In[ ]:


bins = list(np.histogram_bin_edges(df.PACE_SEC,bins=5,range=(df.PACE_SEC.min()-1,df.PACE_SEC.max()+1)))


# In[ ]:


bin_labels = ['UltraFast','Fast','Medium','Slow','Turtle']


# In[ ]:


df['PACE_GROUP'] = pd.cut(df['PACE_SEC'],bins=bins,labels=bin_labels)


# In[ ]:


df.head(5)


# In[ ]:


df['count'] = np.ones(len(df))


# In[ ]:


df.head()


# In[ ]:


gender = df.groupby(['PACE_GROUP','GENDER']).count()['count']


# In[ ]:


ax = plt.figure(figsize=(10,5));
ax = gender.unstack(1).plot.bar(rot=0,subplots=False)
ax.set_ylabel('Count');


# So except "UltraFast" all speed groups are dominated by women. 

# In[ ]:


age =  df.groupby(['PACE_GROUP','AGE']).count()['count']


# In[ ]:


ax = plt.figure(figsize=(10,5));
ax = age.unstack(0).plot.bar(rot=90,subplots=False,stacked=True)
ax.set_xlabel('Count');
ax.set_ylabel('Age Group');


# In[ ]:


rank = df.groupby(['AGE']).agg(min)['RANK']


# In[ ]:


ax = rank.reindex(['0-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-100']).plot.bar()
ax.set_ylabel('Min Rank')


# ## Visualising Correlation with Joint Distributions

# In[ ]:


pd.plotting.scatter_matrix(df[['RANK','PACE_SEC']],figsize=(10,10));


# In[ ]:


sns.jointplot(x=['RANK'], y=['PACE_SEC'], data=df,kind='kde');


# In[ ]:


pd.plotting.scatter_matrix(df[['RANK','BIB']],figsize=(10,10),diagonal='kde');


# In[ ]:


sns.jointplot(x=['RANK'], y=['BIB'], data=df,kind='kde');


# More will come with Discrete Joint Distributions of categorical columns...

# In[ ]:




