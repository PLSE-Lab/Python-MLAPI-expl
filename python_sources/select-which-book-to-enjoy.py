#!/usr/bin/env python
# coding: utf-8

# # Select Which Book to Enjoy?

# ## Intruduction 

# > This file contains the detailed information about the books primarily. Detailed description for each column can be found alongside.And this project will explore these data from different aspects.I wish after this,I could select the best book list for me to enjoy and have sufficient reasons to recommend to other people.  

# > The columns description:    
# **bookID:** A unique Identification number for each book.  
# **title:** The name under which the book was published.  
# **authors:** Names of the authors of the book. Multiple authors are delimited with.  
# **average_rating:** The average rating of the book received in total(full score is 5).  
# **isbn:** Another unique number to identify the book, the International Standard Book Number.  
# **isbn13A:** 13-digit ISBN to identify the book, instead of the standard 11-digit ISBN.  
# **language_code:** Helps understand what is the primary language of the book. For instance, eng is standard for English.  
# **num_pages:** Number of pages the book contains.  
# **ratings_count:** Total number of ratings the book received.  
# **text_reviews_count:** Total number of written text reviews the book received.

# ## Questions

# **About books:**  
# - Q1. The top 10 rating books.  
# - Q2. The top 10 popular book.  
# - Q3. The correlation between rating and other factors.  
# - Q4. Is there a special distrbution across average rating?  
# **About authors:**
# - Q1. The top 10 highest production authors.  
# - Q2. The top 10 authors whose books have the highest rating.  
# - Q3. The correlation between rating and productions number.  
# - Q4. Compare J.K. Rowling, J.R.R. Tolkien and George R.R. Martin.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Data Understanding

# In[ ]:


# load data file
df = pd.read_csv('../input/books.csv', error_bad_lines=False)
df.sample(5)


# In[ ]:


df.shape


# In[ ]:


# check out null data
df.info()


# In[ ]:


# check out duplicated data
df.duplicated().sum()


# In[ ]:


# Check out the unique data of some columns
df['authors'].unique()


# In[ ]:


# Returns valid descriptive statistics for each column of data
df.describe()


# - The column `"# num_pages"` should be renamed to standard form, `"num_pages"`.  
# - When the rating count number is small, the result may be less credible, we should drop these data.  
# - For 0 page book data, we should fill it with the right number (search in Amazon) or replace it by the average pages of all books if there are no exact information on the Internet.
# - The columns,`"isbn"` , `"isbn13"` & `"bookID"` , are uesless in exploratory process.
# - Some books have more than one author, we can choose the first author to be the main author.

# ## Data Cleaning

# In[ ]:


# copy the data
data = df.copy()


# In[ ]:


# change "# num_pages" to "num_pages"
data.rename(columns = {'# num_pages': 'num_pages'}, inplace=True)


# In[ ]:


# drop the data with less than 300 ratings counts
data = data[data['ratings_count'] > 299]


# In[ ]:


# find the book data with 0 page
data[data['num_pages'] == 0]


# In[ ]:


# fill the "num_pages" with right number
data.loc[data['bookID'] == 2874, 'num_pages'] = 209
data.loc[data['bookID'] == 17983, 'num_pages'] = 306
data.loc[data['bookID'] == 32974, 'num_pages'] = 1020 # the sum pages of Nancy Drew: #1-6


# In[ ]:


# replace the 0 page by the average page of all books
data.loc[data['num_pages'] == 0, 'num_pages'] = data['num_pages'].mean()


# In[ ]:


data['num_pages'] = data['num_pages'].astype(int)


# In[ ]:


# dorp the useless columns
data.drop(['bookID', 'isbn', 'isbn13'], axis=1, inplace=True)


# In[ ]:


# select the 1st author
data['authors'] = data['authors'].apply(lambda x: x.split("-")[0])


# ## Exploratory data analysis

# ### A. About books

# #### Q1. The top 10 rating books.

# In[ ]:


# select top 10 rating books
top10_rating = data.sort_values('average_rating', ascending=False)[:10]


# In[ ]:


top10_rating


# In[ ]:


sns.set(style="white", context="talk")
fig = plt.figure(figsize=(13,12))
x = top10_rating['average_rating']
y = top10_rating['title']
sns.barplot(x=x, y=y, palette="GnBu_d")
plt.title('The top 10 rating books', fontsize=24)
plt.show();


# - The highest rating book is **"The Complete Calvin and Hobbes"**.

# #### Q2. The top 10 popular book 

# In[ ]:


# select the top 10 books with highest text_reviews_count 
top10_pop = data.sort_values('text_reviews_count', ascending=False)[:10]
top10_pop


# In[ ]:


sns.set(style="white", context="talk")
fig = plt.figure(figsize=(13,12))
x = top10_pop['text_reviews_count']
y = top10_pop['title']
sns.barplot(x=x, y=y, palette="GnBu_d")
plt.title('The top 10 popular books', fontsize=24)
plt.show();


# - According to the number of text_reviews_count, we may speculate the most popular book is "Twilight".

# #### Q3. The correlation between rating and other factors.

# In[ ]:


# average_rating & text_reviews_count
sns.set(style="darkgrid")
sns.jointplot(y = "text_reviews_count", x = "average_rating", data=data, kind="reg", space=0.5)
plt.title('The Scatter of Average Rating and Text Reviews Count',fontsize=15)
plt.show();


# - There is no correlation between "Average Rating" and "Text Reviews Count".

# In[ ]:


# average_rating & ratings_count
sns.set(style="darkgrid")
sns.jointplot(y = "ratings_count", x = "average_rating", data=data, kind="reg", space=0.5)
plt.title('The Scatter of Average Rating and Ratings Count',fontsize=15)
plt.show();


# - There is no correlation between "Average Rating" and "Ratings Count".

# In[ ]:


# average_rating & num_pages
sns.set(style="darkgrid")
sns.jointplot(y = "num_pages", x = "average_rating", data=data, kind="reg", space=0.5)
plt.title('The Scatter of Average Rating and Num Pages',fontsize=15)
plt.show();


# - There is a weak positive correlation between "Average Rating" and "Num Pages".

# #### Q4. Is there a special distrbution across average rating?

# In[ ]:


fig = plt.figure(figsize=(9,8))
sns.distplot(data['average_rating'])
plt.title('The Distribution across Avrage Rating', fontsize=15)
plt.vlines(data['average_rating'].mean(),ymin = 0,ymax = 1.75,color = 'black')
plt.vlines(data['average_rating'].median(),ymin = 0,ymax = 1.75,color = 'red')
plt.vlines(data['average_rating'].mode(),ymin = 0,ymax = 1.75,color = 'yellow')
plt.legend()
plt.show()


# - The "mean < median < mode", so the rating distribution is a negative skew distribution.

# ### B. About authors 

# #### Q1. The top 10 highest production authors

# In[ ]:


# select data
top10_production = data['authors'].value_counts()[:10]
top10_production = pd.DataFrame(top10_production)
top10_production


# In[ ]:


sns.set(style="white", context="talk")
fig = plt.figure(figsize=(13,12))
x = top10_production['authors']
y = top10_production.index
sns.barplot(x=x, y=y, palette="GnBu_d")
plt.title('The top 10 highest production authors', fontsize=24)
plt.show();


# - The writer with the highest productions is **Stephen King**.

# #### Q2. The top 10 authors whose books have the highest rating.

# In[ ]:


# select data
top10_rating_authors = pd.DataFrame(data.groupby('authors')['average_rating'].mean().sort_values(ascending=False)[:10])
top10_rating_authors


# In[ ]:


sns.set(style="white", context="talk")
fig = plt.figure(figsize=(13,12))
x = top10_rating_authors['average_rating']
y = top10_rating_authors.index
sns.barplot(x=x, y=y, palette="GnBu_d")
plt.title('The top 10 authors whose books have the highest rating', fontsize=24)
plt.show();


# - The highest rating author is **Bill Watterson**, and the top10_rating_authors list is quite different from the top10_production authors list.

# #### Q3.  The correlation between rating and productions number.

# In[ ]:


production = pd.DataFrame(data['authors'].value_counts())
production.reset_index(inplace=True)
production.rename(columns={'index': 'authors', 'authors': 'production_counts'}, inplace=True)
production.head()


# In[ ]:


rating = pd.DataFrame(data.groupby('authors')['average_rating'].mean())
rating.reset_index(inplace=True)


# In[ ]:


data_1 = rating.merge(production, how='inner')
data_1.head()


# In[ ]:


sns.set(style="darkgrid")
sns.jointplot(y = "production_counts", x = 'average_rating', data=data_1, kind="reg", space=0.5)
plt.title('The Scatter of Average Rating(authors) and Productions',fontsize=15)
plt.show();


# - There is no correlation between average rating(authors) and productions. Authors with a high quantity of works doesn't mean that his/her books also have a high average rating. 

# #### Q4. Compare J.K. Rowling, J.R.R. Tolkien and George R.R. Martin

# In[ ]:


# select data
data_2 = data[data['authors'].isin(['George R.R. Martin', 'J.R.R. Tolkien', 'J.K. Rowling'])]
data_2 = data_2.query('language_code == "eng"')
data_2.head()


# In[ ]:


data_2.describe()


# In[ ]:


# classify the rating level
bin_edges = [3.5, 4.0, 4.5, 5.0]
bin_names = ['low', 'medium', 'high']
data_2['rating_levels'] = pd.cut(data_2['average_rating'], bin_edges, labels=bin_names)


# In[ ]:


data_3 = pd.DataFrame(data_2.groupby('authors')['rating_levels'].value_counts().unstack().fillna(0))
data_3


# In[ ]:


# Pie chart
labels = 'high', 'low', 'medium'
explode = (0.1, 0, 0)
fig1, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize = (12,4))

# George R.R. Martin
ax1.pie(data_3.iloc[0], explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.set_title('George R.R. Martin')
ax1.axis('equal')
# J.K. Rowling
ax2.pie(data_3.iloc[1], explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax2.set_title('J.K. Rowling')
ax2.axis('equal')
# J.R.R. Tolkien
ax3.pie(data_3.iloc[2], explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax3.set_title('J.R.R. Tolkien')
ax3.axis('equal')

plt.show()


# In[ ]:


# compare "Harry Potter", "The Lord of the Rings" and "A Song of Ice and Fire"
data_4 = data_2[data_2['title'].str.contains('Harry Potter|The Lord of the Rings|A Song of Ice and Fire')]


# In[ ]:


data_4.groupby('authors')['average_rating'].mean().sort_values(ascending=False)


# - Compare to others, J.K. Rowling's works have higher average rating, and her book "Harry Potter" may be the better choice if I'd like to read a fantasy novel. However, these three authors are all my favorite and all have high rating, if I have enough time, I'll read the three books and recommond all to you.

# - The analysis aspacts are very limited, if possible, we can explore more information about the readers age distribution, books categories,  released time sequence and so on. 

# In[ ]:




