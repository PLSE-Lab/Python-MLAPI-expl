#!/usr/bin/env python
# coding: utf-8

# **Learning as a part of ML_Course_AI ( Lecture I**)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt # for data visualization purposes

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv(filepath_or_buffer='../input/beauty.csv')
#for Jupyter Notebook
#df = pd.read_csv(filepath_or_buffer='../input/beauty.csv', sep=';')


# In[ ]:


type(df)


# In[ ]:


df.head()


# In[ ]:


df['wage']


# In[ ]:


type(df['wage'])


# In[ ]:


df['wage'].head()


# In[ ]:


#number of rows and columns
df.shape


# In[ ]:


#column names
df.columns


# In[ ]:


#column info
df.info()


# In[ ]:


#showing simple statistics of data columns
df.describe()


# In[ ]:


#plotting with dataframes
df['wage'].hist()


# In[ ]:


plt.figure(figsize=(20,10))
df.hist()


# In[ ]:


df['female'].unique()
#men & women


# In[ ]:


df['female'].nunique()
#male & female


# In[ ]:


df['female'].value_counts()
# 0 - men (count 824), 1 - women/female (count 436)


# In[ ]:


df['looks'].nunique()


# In[ ]:


df['looks'].value_counts()


# In[ ]:


df['goodhlth'].value_counts(normalize=True)


# **Indexing
# 
# **.iloc(~NumPy arrays)****

# In[ ]:


df.iloc[0,5]


# In[ ]:


#data slices
df.iloc[:6, 5:7]


# In[ ]:


# making a new dataframe

toy_df = pd.DataFrame({'age': [17,32,56],
                       'salary': [56,69,120]},
                        index=['Kate', 'Leo', 'Max'])


# In[ ]:


toy_df


# In[ ]:


toy_df.iloc[1,0]


# In[ ]:


# .loc

toy_df.loc[['Leo', 'Max'], 'age']


# In[ ]:


df[df['wage'] > 40]


# In[ ]:


df[df['wage'] > 40]


# In[ ]:


df[(df['wage'] > 10) & (df['female'] == 1)]


# **apply function/method**
# 
# **'female'/'male'**

# In[ ]:


def gender_id_to_str(gender_id):
    if gender_id == 1:
        return 'female'
    elif gender_id == 0:
        return 'male'
    else:
        return 'Wrong Input'


# In[ ]:


df['female'].apply(gender_id_to_str).head()


# In[ ]:


df['female'].apply(lambda gender_id :
                 'female' if gender_id == 1 
                 # elif gender_id == 0 'male'
                  else 'male').head()


# In[ ]:


df['female'].map({0: 'male', 1: 'female'}).head()


# **GroupBy**

# In[ ]:


df.loc[df['female'] == 0, 'wage'].median()


# In[ ]:


df.loc[df['female'] == 1, 'wage'].median()


# In[ ]:


for (gender_id, sub_dataframe) in df.groupby('female'):
    print(gender_id)
    print(sub_dataframe.shape)


# In[ ]:


for (gender_id, sub_dataframe) in df.groupby('female'):
    print('Median wages for {} are {}'.format('men' if gender_id == 0
                                                      else 'women',                                         
                                         sub_dataframe['wage'].median()))


# **More about GroupBy**

# In[ ]:


df.groupby('female')['wage'].median()


# In[ ]:


df.groupby(['female', 'married'])['wage'].median()


# **Crosstab**

# In[ ]:


pd.crosstab(df['female'],df['married'])


# **Data Visualization**

# In[ ]:


# pip install seaborn / conda install seaborn
import seaborn as sns


# **wage vs educ**

# In[ ]:


df['educ'].nunique()


# In[ ]:


df['educ'].value_counts()


# In[ ]:


sns.boxplot(x='wage', data=df)


# **IQR (Inter-Quartile Range) = perc_75 - perc_25**

# In[ ]:


sns.boxplot(x='wage', data=df[df['wage'] < 30]);
#restricting the outliers


# In[ ]:


sns.boxplot(x='educ', y='wage', data=df[df['wage'] < 30]);


# In[ ]:




