#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt


# In[ ]:


df = pd.read_csv(filepath_or_buffer='../input/beauty.csv')


# In[ ]:


type(df)


# In[ ]:


df.head()


# In[ ]:


df['wage'].head()


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df['wage'].hist();


# In[ ]:


plt.figure(figsize=(16, 10))
df.hist();


# In[ ]:


df['female'].value_counts()


# In[ ]:


df['looks'].value_counts()


# In[ ]:


df['female'].value_counts()


# In[ ]:


df['goodhlth'].value_counts(normalize=True)


# # Indexing

# ## .iloc (~NumPy arrays)

# In[ ]:


df.iloc[:6, 5:7]


# In[ ]:


toy_df = pd.DataFrame({'age':[17, 32, 56], 
                       'salary':[56,69, 120]}, 
                      index=['Kate', 'Leo', 'Max'])


# In[ ]:


toy_df


# In[ ]:


toy_df.iloc[1, 1]


# ## .loc

# In[ ]:


toy_df.loc[['Leo', 'Max'], 'age']


# ## Boolean indexing

# In[ ]:


df[df['wage'] > 40]


# In[ ]:


df[(df['wage'] > 10) & (df['female'] == 1)]


# ## apply
# ### 'female'/'male'

# In[ ]:


df['female'].apply(lambda gender_id : 'female' if gender_id == 1 else 'male').head()


#    ### map

# In[ ]:


df['female'].map({0: 'male', 1: 'female'}).head()


# ## GroupBy

# In[ ]:


df.loc[df['female'] == 0, 'wage'].median()


# In[ ]:


df.loc[df['female'] == 1, 'wage'].median()


# In[ ]:


for (gender_id, sub_dataframe) in df.groupby('female'):
    #print(gender_id)
    #print(sub_dataframe.shape)
    print('Median wages for {} are {}'.format('men' if gender_id == 0
                                             else 'women', 
                                             sub_dataframe['wage'].median()))


# In[ ]:


df.groupby('female')['wage'].median()


# In[ ]:


df.groupby(['female', 'married'])['wage'].median()


# ## Crosstab

# In[ ]:


pd.crosstab(df['female'], df['married'])


# In[ ]:


import seaborn as sns


# ### wage/educ

# In[ ]:


df['educ'].value_counts()


# ### IQR (Inter-Quartile Range) = perc_75 - perc_25

# In[ ]:


sns.boxplot(x='educ', y='wage', data=df[df['wage']<30]);


# In[ ]:




