#!/usr/bin/env python
# coding: utf-8

# As the dataset name suggests, it contains data about various books which were published over the years and some metadata for those books. 
# 
# I'll try to focus on **Data Cleaning** and **Exploratory Data Analysis** in this notebook. 
# 
# **Please upvote it if you find this useful.**

# In[ ]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# ### Understanding Dataset
# 
# Let's try to look at various fields, their datatypes and some statistics around them to understand our dataset in detail.

# In[ ]:


df = pd.read_csv('../input/goodreadsbooks/books.csv',error_bad_lines=False)
df.head()


# In[ ]:


df.describe(include = 'all')


# Based on above data, we have **11123 unique books** and **no NULL values in our dataset**. 
# 
# Lets look at datatypes of those columns.

# In[ ]:


print(df.dtypes)


# If you observe closely, **we have an extra ' ' before num_pages column name**. It'll be good to fix this right now to avoid any issues later on.

# In[ ]:


df.rename(columns={"  num_pages": "num_pages"}, inplace = True)
df.columns


# With help of visualizations, we will try to answer few common questions:
# 1. Number of Books published by Language (each year)
# 2. Number of published books based on their language
# 3. Top 10 Authors with most number of published books
# 4. Distribution of Average Rating received by books?
# 5. Which are the top-10 rated books?
# 6. Top 10 publishing houses based on number of books published by them
# 
# Similarly we will also try to see if 2 or more variables are related to each other, by performing bivariate analysis:
# 7. Are any of the fields in the dataset correlated?
# 8. Visualize relationships of various fields with each other using a pairplot
# 
# 

# ### Number of Books published each year by Language

# In[ ]:


# df['publish_date'] =
# pd.DatetimeIndex(df['publication_date']).year
# pd.to_datetime(df['publication_date'], format = '%Y')

df['publication_year'] = [i.split('/')[2] for i in df['publication_date']]
df['decade'] = [((int(i)//10)*10) for i in df['publication_year']]

df_lang_year = df.groupby(['decade','language_code']).count().reset_index()
df_lang_year


# In[ ]:


plt.figure(figsize=(20,10))
plt.xlabel('Year')
plt.ylabel('Number of Books')
    
ax1 = sns.lineplot(x="decade", y="bookID",
             hue="language_code", #style="event",
             data=df_lang_year)

ax1.set_ylabel('Number of Books')
ax1.set_xlabel('Decade')


# Surprisingly, most of the books (at least in our dataset) got published between **2000-2010.**

# ### Published books by Language

# In[ ]:


x = df.groupby('language_code')['bookID'].count().reset_index().sort_values(by = 'bookID',ascending=False)
plt.figure(figsize=(15,10))

ax1 = sns.barplot(x = 'language_code', 
            y = 'bookID',
           data = x)

ax1.set_xlabel('Language Code')
ax1.set_ylabel('Number of Books')
ax1.set_yscale("log")
# ax1.set_ticklabels(x['bookID'], minor=False)


# Here we have used the **<font color = 'red'>log-scale</font>** as most of the books are published in English (eng) and for many languages they have barely 1 book. Using log-scale makes it easy to visualize the scale in case of such disparity. 
# 
# Observation: **<font color = 'red'>English books are actually published in 4 language codes</font>**, *eng, en-US, en-GB, en-CA*.
# 
# FYI, I've included a chart without log-scale below to show the disparity in y-axis labels.

# In[ ]:


plt.figure(figsize=(15,15))
chart = sns.countplot(
    data=df,
    x='language_code'
)


ax1.set_xlabel('Language Code')
ax1.set_ylabel('Number of Books')


# Let's look at the distribution with all English language books combined.

# In[ ]:


df['updated_language'] = ['en' if i in ('eng','en-US', 'en-GB', 'en-CA') else i for i in df['language_code']]
x = df.groupby('updated_language')['bookID'].count().reset_index().sort_values(by = 'bookID',ascending=False)

plt.figure(figsize=(15,10))

ax1 = sns.barplot(x = 'updated_language', 
            y = 'bookID',
           data = x)

ax1.set_xlabel('Language Code')
ax1.set_ylabel('Number of Books')
ax1.set_yscale("log")
# ax1.set_ticklabels(x['bookID'], minor=False)


# ### Top 10 Authors by Number of Books Published

# In[ ]:


authors = df.groupby('authors')['bookID'].count().reset_index().sort_values(by = 'bookID', ascending = False).head(10)

plt.figure(figsize=(15,10))
au = sns.barplot(x = 'authors',
                 y = 'bookID',
                 data = authors)

au.set_xlabel('Authors')
au.set_ylabel('Number of Books')

# Other way to rotate labels
# au.set_xticklabels(au.get_xticklabels(), 
#                    rotation=45,
#                   fontweight='light',
#                   fontsize='x-large')

plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)


# Looks like a close call. Stephen King and P.G. Wodehouse both have published maximum of 40 books over the period.

# ### Average Rating received by Books - Distribution

# In[ ]:


df['average_rating_rounded'] = df['average_rating'].round(1)
plt.figure(figsize=(20,15))
ax1 = sns.countplot(
    data=df,
    x='average_rating_rounded'
)

ax1.set_xlabel('Average Rating')
ax1.set_ylabel('Number of Books')


# Looks like most of the books in our dataset have a rating of 4.0. 
# 
# > Note: We have rounded the ratings to 1 decimal point to make it easier to visualize the data.

# In[ ]:


fig, ax = plt.subplots(figsize=[14,8])
sns.distplot(df['average_rating'],ax=ax)
ax.set_title('Average rating distribution for all books',fontsize=20)
ax.set_xlabel('Average rating',fontsize=13)


# ### Top 10 Most Rated Books

# In[ ]:


most_rated = df.sort_values(by = 'ratings_count', ascending = False)[['title','ratings_count']]

plt.figure(figsize = (15,10))

ax1 = sns.barplot(x="ratings_count", 
            y="title", 
            data=most_rated.head(10)
           )

ax1.set_yticklabels(ax1.get_yticklabels(), 
                  fontweight='light',
                  fontsize='small')

ax1.set_ylabel('Title')
ax1.get_xaxis().get_major_formatter().set_scientific(False)


# Although Twilight is the most rated book, we have **4 Harry Potter books** in the top-10.
# 
# ![HarryPotter](https://wallpaperboat.com/wp-content/uploads/2019/06/harry-potter-12.jpg)

# ### Top 10 Publishers

# In[ ]:


pub_data = df.groupby('publisher')['bookID'].count().reset_index().sort_values(by = 'bookID', ascending = False)

ax1 = sns.barplot(x = 'publisher',
                 y = 'bookID',
                 data = pub_data.head(10))

ax1.set_xticklabels(ax1.get_xticklabels(),
                   rotation = 45,
                  fontweight='light',
                  fontsize='small')

ax1.set_ylabel('Number of Books')


# Vintage publishing house has published almost 300 books over this timeframe.

# ### Bivariate Analysis

# Are any of the columns correlated with each other?

# In[ ]:


plt.figure(figsize=(12, 8))

df_corr = df[['average_rating','num_pages','ratings_count', 'text_reviews_count']].corr()
sns.heatmap(df_corr, 
            xticklabels = df_corr.columns.values,
            yticklabels = df_corr.columns.values,
            annot = True);


# As expected, we see there's a strong correlation between ratings_count and average_rating.

# In[ ]:


plt.figure(figsize=(14, 14))

sns.pairplot(df, diag_kind='kde');


# Here we have just plotted relations among all variables with a pairplot (crossplot). This is an easy way to spot any discrepancies or any relations in our data. 

# **I hope this notebook helps you. Please let me know if you have any suggestions/feedback. Please upvote it if you liked it.**
