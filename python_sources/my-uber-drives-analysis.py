#!/usr/bin/env python
# coding: utf-8

# #Question1: How many miles was earned per category and purpose ?
# #Question2: What is percentage of business miles vs personal vs. Meals ?
# #Question3: How much time was spend for each drive per category and purpose ?

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


#add visualisation library
import seaborn as sns


# In[ ]:


#load the data from datasets
df = pd.DataFrame.from_csv("../input/My Uber Drives - 2016.csv", index_col=None)


# In[ ]:


#Take a look into the loaded dataframe
#The field PURPOSE* has nan values. Replace it by category "Other"
df.info()
df['PURPOSE*'].unique()
df['PURPOSE*'].replace(np.nan, 'Other', inplace=True)


# In[ ]:


# START DATE and END_DATE have string format. Convert it to datetime object
# You will discover that the last row contains string values. Just remove it from DataFrame
df[-5:]
df = df[:-1]
df.loc[:, 'START_DATE*'] = df['START_DATE*'].apply(lambda x: pd.datetime.strptime(x, '%m/%d/%Y %H:%M'))
df.loc[:, 'END_DATE*'] = df['END_DATE*'].apply(lambda x: pd.datetime.strptime(x, '%m/%d/%Y %H:%M'))


# In[ ]:


#Calculate the time for the rides and convert it to numbers
df['DIFF'] = df['END_DATE*'] - df['START_DATE*']
df.loc[:, 'DIFF'] = df['DIFF'].apply(lambda x: pd.Timedelta.to_pytimedelta(x).days/(24*60) + pd.Timedelta.to_pytimedelta(x).seconds/60)


# In[ ]:


df.head()


# In[ ]:


#Question 1
g = sns.factorplot(x="PURPOSE*", y="MILES*", hue="CATEGORY*", data=df,
                   size=10, kind="bar", palette="muted")
#from the graph is clearly seen that the main contributors for miles are:
#in Business category: meetings and customer; private: commute and charity 


# In[ ]:


#Question2:
df.groupby(['CATEGORY*'])['MILES*'].sum() / df['MILES*'].sum()
#94% of Miles was earned by business trips


# In[ ]:


#Question3:
df.groupby(['CATEGORY*', 'PURPOSE*'])['DIFF'].sum() / df.groupby(['CATEGORY*'])['DIFF'].sum()
#the most time spend in the cab for business - meeting/other, personal - commute/other


# In[ ]:




