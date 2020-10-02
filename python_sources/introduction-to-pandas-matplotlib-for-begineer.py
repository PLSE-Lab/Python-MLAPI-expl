#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl # To plot data
import seaborn as sns # to plot data + some regression
import statsmodels.api as sm # Ordinary Least Squares (OLS) Regression and some other
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Importing Data

# In[ ]:


names = ['id', 'title', 'year', 'rating', 'votes', 'length', 'genres']
data = pd.read_csv('/kaggle/input/imbd-top-10000/imdb_top_10000.txt', sep="\t", names=names)
data


# # Exploring our Data

# In[ ]:


data.head()


# In[ ]:


data.head(3)


# In[ ]:


data.tail()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# # Exporting Data

# In[ ]:


data.to_csv('test.csv', header=True, index=True, sep=',')


# # Sorting Data

# In[ ]:


data.sort_values(by='rating')


# In[ ]:


data.sort_values(by='rating', ascending=True)


# # Creating Data Frames From Scratch

# In[ ]:


sample_data = {
    'tv': [230, 44, 17],
    'radio': [37, 39, 45],
    'news': [69, 45, 69],
    'sales': [22, 10, 9]
}


# In[ ]:


data2 = pd.DataFrame(sample_data)


# In[ ]:


data2


# In[ ]:


del data2


# In[ ]:


data2
# this code play some error due to it can't find variable data2 because i has delete to show you "del data2" works.


# # Selecting Data

# In[ ]:


data['title']


# In[ ]:


data[['title', 'year']]


# In[ ]:


data['rating'].mean()


# In[ ]:


data['rating'].max()


# In[ ]:


data['rating'].min()


# In[ ]:


data['genres'].unique()


# In[ ]:


data['rating'].value_counts()


# In[ ]:


data['rating'].value_counts().sort_index()


# In[ ]:


data['rating'].value_counts().sort_index(ascending=False)


# # Plotting

# In[ ]:


# We can't write code "import matplotlib as mpl" because i had written above.


# In[ ]:


data.plot()


# In[ ]:


data.plot(kind='scatter', x='rating', y='votes')


# In[ ]:



data.plot(kind='scatter', x='rating', y='votes', alpha=0.3)


# In[ ]:


data['rating'].plot(kind='hist')


# In[ ]:


# We can't write code "import seaborn as sns" because i had written above.


# In[ ]:


sns.lmplot(x='rating', y='votes', data=data)


# In[ ]:


sns.pairplot(data)


# # Ordinary Least Squares (OLS) Regression

# In[ ]:


# We can't write code "import statsmodels.api as sm" because i had written above.


# In[ ]:


results = sm.OLS(data['votes'], data['rating']).fit()


# In[ ]:


results.summary()


# # Advanced Data Selection

# In[ ]:


data[data['year'] > 1995]


# In[ ]:


data['year'] > 1995


# In[ ]:


data[data['year'] == 1966]


# In[ ]:


data[(data['year'] > 1995) & (data['year'] < 2000)]


# In[ ]:


data[(data['year'] > 1995) | (data['year'] < 2000)]


# In[ ]:


data[(data['year'] > 1995) & (data['year'] < 2000)].sort_values(by='rating', ascending=False).head(10)


# # Grouping

# In[ ]:


data.groupby(data['year'])['rating'].mean()


# In[ ]:


data.groupby(data['year'])['rating'].max()


# In[ ]:


data.groupby(data['year'])['rating'].min()


# # Challenges
# 1. What was the highest scoring movie in 1996?
# 2. In what year was the highest rated movie of all time made?
# 3. What five movies have the most votes ever?
# 4. What year in the 1960s had the highest average movie rating?

# > Expand below four lines to see Answers of Above Challege Question

# In[ ]:


# Answer of Question 1
data[data['year'] == 1996].sort_values(by='rating', ascending=False).head()


# In[ ]:


# Answer of Question 2
data[data['rating'] == data['rating'].max()]


# In[ ]:


# Answer of Question 3
data.sort_values(by='votes', ascending=False).head()


# In[ ]:


# Answer of Question 4
data[(data['year'] >= 1960) & (data['year'] < 1970)].groupby(data['year'])['rating'].mean()


# # Cleaning Data

# In[ ]:


data['formatted title'] = data['title'].str[:-7]


# In[ ]:


data.head()


# In[ ]:


data['formatted title'] = data['title'].str.split(' \(').str[0]


# In[ ]:


data.head()


# In[ ]:


data['formatted length'] = data['length'].str.replace(' mins.', '').astype('int')


# In[ ]:


data.head()


# In[ ]:


sns.pairplot(data)


# In[ ]:


data[data['formatted length'] == 0]


# # If you like my notebook so Follow Me And upvote my Notebook
