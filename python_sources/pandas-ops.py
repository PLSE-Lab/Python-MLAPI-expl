#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing train and test data into train_df and test_df dataframes
import pandas as pd
train_df = pd.read_csv('/kaggle/input/sce-data-science-2020-course/train.csv')
test_df = pd.read_csv('/kaggle/input/sce-data-science-2020-course/test.csv')


# In[ ]:


# dataframe deep copy
data = train_df.copy(deep = True)


# In[ ]:


# dataframe column names
train_df.columns.values


# In[ ]:


# preview the n first rows of dataframe
train_df.head(n = 4)


# In[ ]:


# preview the n last rows of dataframe
train_df.tail(n=7)


# In[ ]:


# dataframe types
train_df.info()


# In[ ]:


# dataframe numerical features distribution
train_df.describe()


# In[ ]:


# dataframe categorical features distribution
train_df.describe(include=['O'])


# In[ ]:


# dataframe missing data information
train_df.isnull().sum()


# In[ ]:


# get table of frequency counts for items in a dataframe column
train_df['Pclass'].value_counts()


# In[ ]:


# get table of frequency counts for items in a dataframe column
train_df['Embarked'].value_counts()


# In[ ]:


# rename column in dataframe
d = train_df.copy(deep = True)
d.rename(columns = {'Survived':'S'}, inplace = True)
d.head(1)


# In[ ]:


# selecting a subset of dataframe
df_subset = train_df[(train_df.Age > 50) & (train_df.Pclass == 2)] 
df_subset.describe()


# In[ ]:


# selecting a subset of dataframe
df_subset = train_df[((train_df.Pclass == 1) | (train_df.Pclass == 2)) & (train_df.Survived == 'T')] 
df_subset.describe()


# In[ ]:


# selecting a subset of dataframe by range of indexes and column index
df_subset = train_df.iloc[0:100, 5]
df_subset.describe()


# In[ ]:


# group data by dataframe subset
grouped_data = train_df.groupby(['Sex', 'Pclass','Age'])
grouped_data['Survived'].describe()


# In[ ]:


# missing values replacement for numerical column
data = train_df.copy(deep = True)
c = 'Age'
m = data[c].dropna().mean()
data[c].fillna(m, inplace=True)
data.info()


# In[ ]:


# missing values replacement for categorical column
data = train_df.copy(deep = True)
c = 'Cabin'
m = data[c].dropna().mode()[0]
data[c].fillna(m, inplace=True)
data.info()


# In[ ]:


# columns transformation - way 1
data = train_df.copy(deep = True)
c = {'T':1,'F':0}
data.rename(columns = {'Survived':'S'}, inplace = True)
data['Survived'] = data['S'].map(c)
data['Died'] = 1 - data['Survived']
data.head(10)


# In[ ]:


# columns transformation - way 2
data = train_df.copy(deep = True)
data.rename(columns = {'Survived':'S'}, inplace = True)
data['Survived'] = data['S'].apply(lambda x: 1 if x == 'T' else 0)
data['Died'] = 1 - data['Survived']
data.head(10)


# In[ ]:


# columns transformation - way 3
# http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example
data = train_df.copy(deep = True)
cols = pd.get_dummies(data['Survived'],prefix='Survived_', drop_first=False)
data = pd.concat([data,cols],axis=1)
data.head(10)


# In[ ]:


# label encoder


# In[ ]:


# grouping data and aggregating columns
pd = train_df.groupby('Sex').agg('sum')[['Survived', 'Died']]
print(pd.head())


# In[ ]:


# plotting data
pd.plot(kind='bar', figsize=(25, 7), stacked=False, color=['g', 'r'])


# In[ ]:


# add a column with constant value
data = train_df.copy(deep = True)
data['Test'] = 'Test'
data.head(10)


# Manipulating dataframe
# https://gist.github.com/agalea91/545e2337b94d965be788f7db18b1f497

# In[ ]:


# add a column and bin to intervals
# https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.cut.html
data = train_df.copy(deep = True)
data['Age_Intervals'] = pd.cut(data['Age'], 4)
data.head(10)


# In[ ]:


#drop a column
data = train_df.copy(deep = True)
data = data.drop(['Age'], axis=1)
data.info()

