#!/usr/bin/env python
# coding: utf-8

# **What is Pandas?**<br>
# The pandas package is the most important tool at the disposal of Data Scientists and Analysts working in Python today. The powerful machine learning and glamorous visualization tools may get all the attention, but pandas is the backbone of most data projects.

# **Cheat sheet**<br>
# http://http://pandas.pydata.org/Pandas_Cheat_Sheet.pdf

# **A set of lesson for new pandas users**<br>
# https://bitbucket.org/hrojas/learn-pandas/src/master

# <img src="https://media.mnn.com/assets/images/2017/03/Panda-Eating-Bamboo-Stalk.jpg" width="700px"/>

# **Import libraries**

# In[ ]:


import pandas as pd


# **Read dataset**

# In[ ]:


df = pd.read_csv('../input/pokemon/Pokemon.csv')


# **Print the dataset**

# In[ ]:


df


# **Count total rows**

# In[ ]:


print(str(len(df)))


# **Print first five row**

# In[ ]:


df.head(5)


# **Print last five rows**

# In[ ]:


df.tail(5)


# **Read data size**

# In[ ]:


df.shape


# **Read headers**

# In[ ]:


df.columns


# **Check the types of each column**

# In[ ]:


df.dtypes


# **Read each Columns**

# In[ ]:


df['Name']


# **Read first five names**

# In[ ]:


df['Name'][0:5]


# In[ ]:


df[['Name', 'Generation', 'Speed']]


# **Read each row**

# In[ ]:


df.iloc[1]


# In[ ]:


df.iloc[543]


# **Read multiple rows**

# In[ ]:


df.iloc[9:12]


# **Read a specific a value**

# In[ ]:


df.iloc[10,5]


# In[ ]:


df.iloc[9,2]


# **Read rows using condition**

# In[ ]:


df.loc[df['Type 1'] == 'Water']


# **Read rows using multiple condition**

# In[ ]:


df.loc[(df['Type 1'] == 'Water') & (df['Type 2'] == 'Dark')]


# In[ ]:


df.loc[(df['Type 1'] == 'Water') & (df['Type 2'] == 'Dark') & (df['HP'] > 60)]


# In[ ]:


array = ['Dark', 'Ghost', 'Ground']
df.loc[(df['Type 1'] == 'Fire') & df['Type 2'].isin(array)]


# In[ ]:


array = ['Fire', 'Water', 'Rain']
df.loc[df['Type 1'].isin(array)]


# **Describing data**

# In[ ]:


df.describe()


# **Sorting data**

# In[ ]:


df.sort_values('Name')


# **Sorting data using multiple columns**

# In[ ]:


df.sort_values(['Name', 'HP', 'Generation'])


# **Sorting data into dscending order**

# In[ ]:


df.sort_values('Name', ascending = False)


# In[ ]:


df.sort_values(['Name', 'Type 1', 'Type 2'], ascending = [1, 0, 1])


# **Making changes to the data**

# In[ ]:


df.head(5)


# In[ ]:


df['Sum'] = df['Total'] + df['HP'] + df['Attack']
df.head(5)


# In[ ]:


df['Mul'] = df['HP'] * df['Attack']
df.tail(5)


# **Drop a column**

# In[ ]:


df.columns


# In[ ]:


df = df.drop(columns=['Sum'])
df.head(3)


# **Alternative way to sum columns**

# In[ ]:


df['Sum2'] = df.iloc[:, 4:8].sum(axis=1)
df.head(3)


# **Drop multiple columns**

# In[ ]:


df = df.drop(columns=['Sum2', 'Mul'])
df.head(5)


# **Filtering data**

# **Sreach rows using valus**

# In[ ]:


df.loc[df['Type 1'] == 'Dark']


# In[ ]:


df.loc[df['Type 2'] == 'Fire']


# **Search rows using multiple values**

# In[ ]:


df.loc[(df['Type 1'] == 'Bug') & (df['Type 2'] == 'Fire')]


# In[ ]:


df.loc[(df['Type 1'] == 'Dark') & (df['Type 2'] == 'Fire') & (df['HP'] > 20)]


# In[ ]:


df.loc[((df['Type 1'] == 'Dark') | (df['Type 2'] == 'Dark')) & (df['Total'] > 500)]


# **Making new csv file**

# In[ ]:


df_new = df.loc[(df['Type 1'] == 'Dark') & (df['Type 2'] == 'Fire') & (df['HP'] > 10)]
df_new
#df_new.to_csv('New Pokemon')


# In[ ]:


df_new1 = df.loc[(df['Type 1'] == 'Dark') | (df['Defense'] >= 600)]
df_new1


# **Searchibg by string values**

# In[ ]:


df.loc[df['Name'].str.contains('Houndoom')]


# In[ ]:


df.loc[df['Name'].str.contains('ye')]


# In[ ]:


df.loc[df['Name'].str.contains('Sableye') & (df['Total'] > 390)]


# In[ ]:


df.loc[df['Name'].str.contains('Houndoom')]


# In[ ]:


df.loc[~df['Name'].str.contains('Houndoom')][0:5]


# In[ ]:


import re
df.loc[df['Type 1'].str.contains('Grass|Ghost', regex = True)]


# In[ ]:


df.loc[df['Type 1'].str.contains('poison|flying', flags = re.I, regex=True)]


# **Conditional changes**

# In[ ]:


df.head(5)


# In[ ]:


df.loc[df['Type 1'] == 'Grass', 'Type 1'] = 'Poison'
df.head(5)


# In[ ]:


df.loc[df['Type 1'] == 'Poison', 'Type 1'] = 'Grass'
df.head(5)


# In[ ]:


df.loc[df['HP'] >= 50, ['Total', 'Attack']] = 'Test0'
df.head(5)


# In[ ]:


df.loc[df['Total'] == 'Test0', ['Total', 'Attack']] = [12, 45]
df.head(5)


# In[ ]:


df.loc[df['Total'] == 12, ['Total', 'Attack']] = [2, 5]
df.head(5)


# **Group by - mean**

# In[ ]:


df.groupby('Type 1').mean()


# In[ ]:


df.groupby('Type 1').mean().sort_values('HP', ascending = False)


# **Group by - sum**

# In[ ]:


df.groupby('Type 1').sum()


# **Group by - count**

# In[ ]:


df.groupby('Type 1').count()


# In[ ]:


df['Count'] = 1
df.groupby(['Type 1']).count()['Count']


# In[ ]:


df['Count'] = 1
df.groupby(['Type 1', 'Type 2']).count()['Count']


# **To be countinued...!**
