#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Pandas is a very good library for analyzing our dataset. Pandas stores the data in a dataframe.

# In[ ]:


#reading the input CSV file and storing it in a dataframe
df = pd.read_csv("../input/youtube-new/USvideos.csv")


# In[ ]:


#shape of the dataframe (dataset)
df.shape


# This means that our dataset has 40949 rows (records) and 16 features (columns or attributes)

# In[ ]:


#statistical information about the data
df.describe()


# The describe method helps us to get an idea of the data.

# In[ ]:


#checking for null values. It says increase the count by 1 whenever you find a null value in each column
df.isnull().sum()


# It's **impressive** that there are 0 null values except for the description column. This rarely happens in real-world cases. Most of the times the columns will contain a lot of null (NaN: Not a number) values and there are different methods to replace this NaN values with the mean or median depending upon the problem at hand. You can also get rid of these specific rows or columns where data is NaN.

# In[ ]:


#number of unique values (no duplicates are counted)
df.nunique()


# Since the columns comments_disabled, ratings_disabled and video_error_or_removed take only 2 unique values, we can understand before hand that their datatypes are boolean. We can confirm the same by running df.dtypes which gives the datatype of each column.

# In[ ]:


#data type of each column
df.dtypes


# If you want to know the number of non-null values and data type of columns using a single command, you can do as below.

# In[ ]:


#datatype and number of non-null values in each column
df.info()


# Datasets usually have thousands of rows, someitmes millions and even billions of rows. To know how the data actually looks, we need not print all the rows but only few rows using the below command. In that way we also don't consume a lot of memory.

# In[ ]:


#prints first five rows of the dataframe
df.head()


# Let's plot this data to understand the distribution of the data. Since the most important columns are likes, dislikes, views and comments, we will plot them first. A univariate distribution (single variable) will be plotted.

# In[ ]:


#defining the size of the figure
plt.figure(figsize = (20,4))

#Important note: 1st digit in subplot indicates the number of rows in the plot, 2nd digit indicates the number of columns in the plot, 3rd digit indicates the index
#So, we use 14 as we want one row and 4 columns.
plt.subplot(141)
p1 = sns.distplot(df['likes'], color='green')
plt.xticks(rotation='vertical')
p1.set_title('Distribution of number of likes')

plt.subplot(142)
p2 = sns.distplot(df['dislikes'], color='red')
p2.set_title('Distribution of number of dislikes')

plt.subplot(143)
p3 = sns.distplot(df['views'], color='blue')
p3.set_title('Distribution of number of views')

plt.subplot(144)
p4 = sns.distplot(df['comment_count'], color='yellow')
p4.set_title('Distribution of number of comments')

plt.subplots_adjust(wspace = 0.4)
plt.xticks(rotation='vertical')
plt.show()


# As you can see in the above plots, the values are so small and are not visible clearly. This gives us an indication that we have to plot a log distribution.

# In[ ]:


#defining the size of the figure
plt.figure(figsize = (20,4))

df['likes_log'] = np.log(df['likes']+1)
df['dislikes_log'] = np.log(df['dislikes']+1)
df['views_log'] = np.log(df['views']+1)
df['comments_log'] = np.log(df['comment_count']+1)

plt.subplot(141)
p1 = sns.distplot(df['likes_log'], color='green')
p1.set_title('Log distribution of number of likes')

plt.subplot(142)
p2 = sns.distplot(df['dislikes_log'], color='red')
p2.set_title('Log distribution of number of dislikes')

plt.subplot(143)
p3 = sns.distplot(df['views_log'], color='blue')
p3.set_title('Log distribution of number of views')

plt.subplot(144)
p4 = sns.distplot(df['comments_log'], color='yellow')
p4.set_title('Log distribution of number of comments')

plt.show()


# It can be seen that these distributions closely follow normal distribution. The first one (number of likes) is slightly right-skewed.

# Let's see how to add and remove a column from a dataframe. To add a column, access the column like you normally do using [] and initialize it to a list or a specific value (None, NaN...). To remove a column, use df.drop(column_names, axis). Axis = 1 (removes the entire column), axis = 0 (removes the index)

# In[ ]:


#adding a column with all entries 'None'
df['likesl'] = 'None'
df['dislikesl'] = 'None'

#dropping one column
df.drop('likesl', axis=1)

#dropping multiple columns (columns keyword is optional). Both the below statements yield the same result
df.drop(['dislikesl', 'likesl'], axis=1)
df.drop(columns=['dislikesl', 'likesl'], axis=1)


# Doing the above doesn't permanently delete the column from the dataframe. It deletes only in that execution of the cell. You can check so by doing df.head(). You can see that 'likesl' and 'dislikesl' columns are still present.

# In[ ]:


df.head()


# The reason this is happening is that you are not assigning the obtained dataframe to the current dataframe (namely df). Only by doing this will you be able to update df.

# Now, all the 4 columns will be deleted. Check below

# In[ ]:


df.head()


# In[ ]:


#Percentage of likes, dslikes and comments
df['like_rate'] =  df['likes'] / df['views'] * 100
df['dislike_rate'] =  df['dislikes'] / df['views'] * 100
df['comment_rate'] =  df['comment_count'] / df['views'] * 100


# In[ ]:


#defining the size of the figure
plt.figure(figsize = (20,4))

plt.subplot(131)
p1 = sns.distplot(df['like_rate'], color='green')
plt.xticks(rotation='vertical')
p1.set_title('Like rate')

plt.subplot(132)
p2 = sns.distplot(df['dislike_rate'], color='red')
p2.set_title('Dislikes rate')

plt.subplot(133)
p3 = sns.distplot(df['comment_rate'], color='blue')
p3.set_title('Views rate')

plt.subplots_adjust(wspace = 0.4)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


#Countplots
# sns.set(style="darkgrid")
# ax = sns.countplot(x=df['like_rate'], data=df)


# Plotting scatter plots...

# In[ ]:


plt.figure(figsize = (20,6))
plt.subplot(121)
plt.scatter(df['like_rate'], df['dislike_rate'])
plt.subplot(122)
plt.scatter(df['like_rate'], df['comment_rate'])


# Correlation between views and likes can help us to understand:
# * How many viewers liked the video
# * What are the reasons or what influences a person watching the video like it

# In[ ]:


#Correlation between views and likes
plt.figure(figsize = (20,10))
plt.scatter(df['views'], df['likes'])
plt.xlabel('Views', fontsize=26)
plt.ylabel('Likes', fontsize=26)
plt.xticks(size = 15)
plt.yticks(size = 15)


# We can see that for few videos, likes increased faster than views (for a small increase in number of views, there is a large increase in number of likes). For videos, it's the other way around. It can be seen that, most of the data points are concentrated in the box where views are less than 50 million views and likes are less than a million likes.

# In[ ]:


df['likes'].corr(df['views'])


# 85% is a strong positive correlation. In other ways, it says as number of views increases, number of likes increases and it's valid for 85% of the data points. Few unusual outliers may have a huge impact on correlation.

# **Frankly speaking, this is not accurate. Because, Pearson assumes that variables are linearly propotional whereas here, they are not actually so.**

# In[ ]:


df.corr(method='pearson')


# The standard correlation used by Pandas is Pearson correlation. So, the values used by Pearson method and df.corr() method are so close.

# In[ ]:


sns.heatmap(df.corr())


# The above is called the heatmap and is a very good visualizaiton tool for understanding the correlation between all the attributes of the dataset. ***Personally, this is one of my favorites! ***

# In[ ]:


df.corr(method ='kendall') 


# In[ ]:


plt.figure(figsize = (20,4))

plt.subplot(121)
p1 = sns.distplot(df['likes_log'], color='green')
plt.xticks(rotation='vertical')
p1.set_title('Distribution of number of likes')

plt.subplot(122)
p3 = sns.distplot(df['views_log'], color='blue')
p3.set_title('Distribution of number of views')

# plt.subplots_adjust(wspace = 0.4)
# plt.xticks(rotation='vertical')
plt.show()


# In[ ]:




