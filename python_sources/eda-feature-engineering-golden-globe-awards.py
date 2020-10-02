#!/usr/bin/env python
# coding: utf-8

# # Let's work together
# 
# ## Exploratory Data Analysis on EDA - Golden Globe Awards, 1944 - 2020 dataset.
# 
# Hi welcome to the notebook on Exploratory Data Analysis (EDA) for the Golden Globe awards.
# 
# I was inspired to undertake EDA on this dataset, as I was watching the highlights in the early Jan 2020 and being a fan of movies myself, I wanted to learn more.
# 
# In this notebook, we will be exploring the data, focusing on feature engineering and interesting insights from this dataset.
# 
# Please upvote if you found it interesting and would be much appreciated.
# 
# ![Golden_Globe](http://richiewong.co.uk/wp-content/uploads/2020/01/golden-globe-atmosphere-4.png)

# # Loading the data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv('/kaggle/input/golden-globe-awards/golden_globe_awards.csv')


# In[ ]:


#check the amount of rows within the dataset
len(df)


# In[ ]:


#Let's have a look at the dataset - let's choose sample at random to get a unbias view
df.sample(10)


# In[ ]:


#this is a great function to use to be able to identify the number of categories, number of different nomiees etc.
df.nunique()


# In[ ]:


# There are a lot of unique categories
df.category.unique()


# In[ ]:


#checking type of dataset within the column
df.dtypes


# In[ ]:


#checking for nulls values within the dataset
print('Null values in dataset: \n')
print(df.isnull().sum())
print('\n')
print('Percentage of null in film column:' )
print(round(df['film'].isnull().sum()/len(df)*100,2),"%")


# In[ ]:


# Let's see the column film and why there are NaN?
df[df.isna().any(axis=1)].sample(10)


# It seems like there are NaNs in the film category as they are 'Nominee' are film category.
# 
# In the next part...
# 
# Let's add a new column, Actor or not actor
# 
# Also replace all NaN with the film title of the nomiee.

# # Feature Engineering + Cleaning the data

# ## New column for actor/actress award

# In[ ]:


#this function adds in a new column and if film column is non na then it outputs true in a new column.
df["actor_actress_award"] = pd.notna(df["film"])
df.sample(10)


# Replace all NaN with the film Title.

# In[ ]:


#Now we replace all nan with in films with nominee as this is the film name.

df["film"] = pd.notna(df["film"])
df.sample(10)


# In[ ]:


df['film'] = df.apply(
    lambda row: row['nominee']*row['film'] if np.isnan(row['film']) else row['nominee'],
    axis=1
)
    
df.sample(10)
# Notice we have valuable information behind the dataset base on NaNs.


# Replacing film column with film and year release in the same column

# In[ ]:


df["film"] = df["film"] +' - '+ df["year_film"].astype(str)

# Let's check again how the data looks
df.sample(10)


# 
# 
# 

# In[ ]:


# Checking for nulls now

print(df.isnull().sum())


# ## New column when category of the award was first introduced
# 
# This was bit of a process - if there is more ideal way, keen to know.

# In[ ]:


# Let's see how many awards were presented since 1944 for each category

# Ensure to put filter for if award has been won
# How would you put filters via groupby?
df_winners=df[df.win == True]
df[df.win == True]


# In[ ]:


len(df_winners)


# In[ ]:


# Now we can filter out how many winners there were in each category historically

total_winners = df_winners.groupby(['category']).size().reset_index(name='counts')
total_winners.sample(10)

# We learn that the awards have been changed and taloired to be more specific in the future, for example it is split via television series as well as movies.
# We also learn that there are category that have not been awarded often.


# In[ ]:


total_winners.sample(10).sort_values(by='counts', ascending=False)


# In[ ]:


# We can also use the first function from pandas to be able to see when the awards were first introducted

first_film_award = df[['category','year_award']].groupby('category').first().reset_index()

# Add in true within the column

first_film_award.rename({"year_award": "award_first_awarded"}, axis=1, inplace=True)

#here is a overview of the first awards 
first_film_award.sample(10)


# In[ ]:


# Lets make use of this and merge data together.

df= pd.merge(df,first_film_award,left_on='category',right_on='category',how='left')


# In[ ]:


# Check to see if the column has been added.
df


# ## Column for the decade the year was awarded 
# 
# This will group the data into dates i.e. 40s, 50s, 60s etc.

# In[ ]:


# First we have to convert the column 'year_film' from int64 to string

df["decade_film_award"] = df.year_award.astype(str)

# Next we only want the first 3 chars and we add in '0s'

df["decade_film_award"] = df['decade_film_award'].str[:3]+'0s'


# In[ ]:


# Again check to see if the column has been added as expected.
df


# # Time to show some visuals!

# In[ ]:


# By grouping the year release
# We can see how many nominations there were throughout the years

plt.figure(figsize=(15,6))
sns.countplot(data=df, x='decade_film_award',   palette='hls')
plt.title('Counts of Nominations for the Golden Globe awards from 1944-2020', fontsize=15)
plt.xlabel('Decade')
plt.ylabel("Count")
plt.legend(frameon=False, fontsize=12)

# Notice that 1920's there is only 1 year.


# ## With the grouping by award decade we can

# In[ ]:


#Wins only
df[df.win == True]


# In[ ]:


# Total Nominees in each decade
df_nominees = df.groupby('decade_film_award')['win'].count().to_frame('Count').reset_index()
print(df_nominees)


# In[ ]:


# Total Winners in each decade
df_winners=df[df.win == True]
df_winners_decade = df_winners.groupby('decade_film_award')['win'].count().to_frame('Count').reset_index()
print(df_winners_decade)

df_winners_decade.dtypes


# Can see that within the 1970s to 2010s there has been consistent number of nomiees and winners.

# ## Want to find the percentage of winners from nomiees
# 

# In[ ]:


df_winners_decade['Per_nominated'] = df_winners_decade['Count'] / df_nominees['Count']
df_winners_decade


# We can see it's about 1 in 5 chance to win an award after the 1970s.

# # Let's explore the awards this year in 2020.
# 
# Let's focus on this era of the data to see the awards as it would of been a different era!

# In[ ]:


# The winners in 2020
df_winners[df_winners['decade_film_award'].str.contains("2020s")]


# # Further investigation
# * Number of nominees per category 
# * Films with the most nomiation
# * Films that won the most awards
# * Actors/Actress that won the most awards

# In[ ]:




