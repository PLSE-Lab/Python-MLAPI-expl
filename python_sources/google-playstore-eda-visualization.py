#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# Hai kagglers, this kernel is about EDA from google playstore app dataset. Enjoy.

# ## Import Modules

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')


# ## Quick Look

# In[ ]:


df.head()


# In[ ]:


df.info()


# So we got a total 10841 row with 13 columns, a few missing value. I'm gonna handle each of them on the run. let's start by splitting the dataset by category.

# ## EDA
# 
# Let's take a look on all the category

# In[ ]:


# Category

cat = df.Category.unique()
cat


# So we got 34 category on this dataset, let's see which one is the famous category

# In[ ]:


plt.figure(figsize=(12,12))

most_cat = df.Category.value_counts()
sns.barplot(x=most_cat, y=most_cat.index, data=df)


# Wow, there is around 2000 app with family category, followed by game category with 1200 app. And this '1.9' Category, i don't know what it is, but it only had 1 app so far, so its not visible on the graph. Let's look at the rating, and what kind of correlation share between category and rating.

# In[ ]:


# Rating

df.Rating.unique()


# There we had a null values, i'm going to leave it as it is. And a '19' for rating is not possible, so i assume it's a '1.9'. So let's change it and see the distribution value on rating column.

# In[ ]:


df['Rating'].replace(to_replace=[19.0], value=[1.9],inplace=True)
sns.distplot(df.Rating)


# Most of the rating is around 4. Let's see how rating is distributed by category column

# In[ ]:


g = sns.FacetGrid(df, col='Category', palette="Set1",  col_wrap=5, height=4)
g = (g.map(sns.distplot, "Rating", hist=False, rug=True, color="r"))


# By the horizontal is the rating value, and verticaly is quantity of the rating. 

# In[ ]:


# Mean Rating

plt.figure(figsize=(12,12))

mean_rat = df.groupby(['Category'])['Rating'].mean().sort_values(ascending=False)
sns.barplot(x=mean_rat, y=mean_rat.index, data=df)


# And this is the average of rating by category, family and game has a lot of quantity causing the low on average rating, on the other side event has the highest average rating by category.
# 
# Next is reviews, review sometime can measure the app popularity.
# The more reviews, the better.

# In[ ]:


# Reviews

df.Reviews.unique()


# In[ ]:


# inside review there is a value with 3.0M with M stand for million, lets change it so it can be measure as float

Reviews = []

for x in df.Reviews:
    x = x.replace('M','00')
    Reviews.append(x)

Reviews = list(map(float, Reviews))
df['reviews'] = Reviews
sns.distplot(Reviews)


# This graph is the distribution of total reviews on each app.

# In[ ]:


g = sns.FacetGrid(df, col='Category', palette="Set1",  col_wrap=5, height=4)
g = (g.map(plt.hist, "Reviews", color="g"))


# This graph is the correlation between category and reviews, Family and game category had a lot of reviews.
# Some app also almost had no review at all, like event, beauty, medical, parenting and more. it is interesting Event app has a high rating but rare review on it.

# In[ ]:


# Total reviews

plt.figure(figsize=(12,12))
sum_rew = df.groupby(['Category'])['reviews'].sum().sort_values(ascending=False)
sns.barplot(x=sum_rew, y=sum_rew.index, data=df)


# Showing the amount of total reviews.

# In[ ]:


# Mean reviews

plt.figure(figsize=(12,12))
mean_rew = df.groupby(['Category'])['reviews'].mean().sort_values(ascending=False)
sns.barplot(x=mean_rew, y=mean_rew.index, data=df)


# This is the average of reviews on each category.
# 
# Let's move on to next column, installs.

# In[ ]:


# Installs

df.Installs.unique()


# Now i'm going to transform this column into float as well like review. First we need to change the 0 and Free value to 0+. Next we need to replace the ',' value and discard the + sign form the value.

# In[ ]:


df['Installs'].replace(to_replace=['0', 'Free'], value=['0+','0+'],inplace=True)


# In[ ]:


Installs = []

for x in df.Installs:
    x = x.replace(',', '')
    Installs.append(x[:-1])

Installs = list(map(float, Installs))
df['installs'] = Installs
sns.distplot(Installs)


# DIstributed value of Install on each category.

# In[ ]:


g = sns.FacetGrid(df, col='Category', palette="Set1",  col_wrap=5, height=4)
g = (g.map(plt.hist, "installs", bins=5, color='c'))


# In[ ]:


# Total Installs

plt.figure(figsize=(12,12))
sum_inst = df.groupby(['Category'])['installs'].sum().sort_values(ascending=False)
sns.barplot(x=sum_inst, y=sum_inst.index, data=df)


# In[ ]:


# Mean Install

plt.figure(figsize=(12,12))
mean_ints = df.groupby(['Category'])['installs'].mean().sort_values(ascending=False)
sns.barplot(x=mean_ints, y=mean_ints.index, data=df)


# The mosy installed app is game, and the average of install is comunication app.

# Next, let's go for size.

# In[ ]:


df.Size.unique()


# That's a lot of value. let's deal with varies and change to 0 first. Then we will do the same thing like installs column.

# In[ ]:


df['Size'].replace(to_replace=['Varies with device'], value=['0'],inplace=True)


# In[ ]:


# i need to diiscard + and , value. amd change M for million. Then check the distibution.

Size = []

for x in df.Size:
    x = x.replace('+', '')
    x = x.replace(',', '')
    if 'M' in x:
        if '.' in x:
            x = x.replace('.', '')
            x = x.replace('M', '00')
        else:
            x = x.replace('M', '000')
    elif 'k' in x:
        x = x.replace('k', '')
    Size.append(x)

Size = list(map(float, Size))
df['size'] = Size
sns.distplot(Size)


# This is the distribution of size column in Kb.

# In[ ]:


g = sns.FacetGrid(df, col='Category',  col_wrap=5, height=4)
g = (g.map(plt.hist, "size", bins=5, color='y'))


# In[ ]:


# Mean Size

plt.figure(figsize=(12,12))
mean_size = df.groupby(['Category'])['size'].mean().sort_values(ascending=False)
sns.barplot(x=mean_size, y=mean_size.index, data=df)


# The average size of game app is the highest around 40 MB.

# The Type column, let's check if the app is free or paid.

# In[ ]:


# Type for category

df.Type.unique()


# There is 0 and null value, let's change them to free.

# In[ ]:


df['Type'].replace(to_replace=['0'], value=['Free'],inplace=True)
df['Type'].fillna('Free', inplace=True)


# In[ ]:


print(df.groupby('Category')['Type'].value_counts())
Type_cat = df.groupby('Category')['Type'].value_counts().unstack().plot.barh(figsize=(10,20), width=0.7)
plt.show()


# Then again, family category has the most free and paid app on the playstore. We can see social app is always free, like entertainment, event, education, comic, and more. The medical has a high amount of paid app considering quantity of medical app is not much.

# Last is the version of android you should have before accessing the app.

# In[ ]:


# And Ver

df['Android Ver'].unique()


# Now this is messy, i'm going to group it to 1 till 8 version of android. Change the null value to 1.0.

# In[ ]:


df['Android Ver'].replace(to_replace=['4.4W and up','Varies with device'], value=['4.4','1.0'],inplace=True)
df['Android Ver'].replace({k: '1.0' for k in ['1.0','1.0 and up','1.5 and up','1.6 and up']},inplace=True)
df['Android Ver'].replace({k: '2.0' for k in ['2.0 and up','2.0.1 and up','2.1 and up','2.2 and up','2.2 - 7.1.1','2.3 and up','2.3.3 and up']},inplace=True)
df['Android Ver'].replace({k: '3.0' for k in ['3.0 and up','3.1 and up','3.2 and up']},inplace=True)
df['Android Ver'].replace({k: '4.0' for k in ['4.0 and up','4.0.3 and up','4.0.3 - 7.1.1','4.1 and up','4.1 - 7.1.1','4.2 and up','4.3 and up','4.4','4.4 and up']},inplace=True)
df['Android Ver'].replace({k: '5.0' for k in ['5.0 - 6.0','5.0 - 7.1.1','5.0 - 8.0','5.0 and up','5.1 and up']},inplace=True)
df['Android Ver'].replace({k: '6.0' for k in ['6.0 and up']},inplace=True)
df['Android Ver'].replace({k: '7.0' for k in ['7.0 - 7.1.1','7.0 and up','7.1 and up']},inplace=True)
df['Android Ver'].replace({k: '8.0' for k in ['8.0 and up']},inplace=True)
df['Android Ver'].fillna('1.0', inplace=True)


# In[ ]:


print(df.groupby('Category')['Android Ver'].value_counts())
Type_cat = df.groupby('Category')['Android Ver'].value_counts().unstack().plot.barh(figsize=(10,18), width=1)
plt.show()


# Most of the app on playstore use android 4.0 as the standard minimum version on their app. There is also a few use version 2.0 like game and family.

# ## END
# 
# 
# That is all the EDA of this dataset, Thank you kagglers.
