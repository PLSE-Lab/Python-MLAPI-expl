#!/usr/bin/env python
# coding: utf-8

# **Good-Reads**
# > **Many a times before buying a book we all refer to goodreads to know about the reviews,average rating of the books.
# It also happens sometimes that we come across books of a famous writer and it becomes really diifcult to figure out which book of that author to give a try first.
# Thus I have used some basic exploratory data analysis techniques to know about the 
# 1)highest rated books.
# 2)Books published most number of times by multiple publishers
# 3)The author with the highest rated books
# 4)The average ratings of some of the oldest books **

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Importing the csv file**

# In[ ]:


df = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv', error_bad_lines = False)


# **Exploring the dataset**

# In[ ]:


df.head()


# In[ ]:


print(df.shape)
df.dtypes


# **Number of books published by each author**

# In[ ]:


authorsdf=df["authors"].value_counts()


# In[ ]:


authorsdf


# **Top 10 authors with the highest number of books**

# In[ ]:


xd=df['authors'].value_counts().iloc[:10]
xd


# **But some of the books are published by multiple publishers which in turn increases the count of number of books per author**

# In[ ]:


yd=df['title'].value_counts().iloc[:10]
yd


# **A plot showing the number of published books of top 5 authors**

# In[ ]:


plt.figure(figsize=(20,10))
plt.bar(xd.keys().tolist(),xd.tolist(),color="orange")


# ****A barplot representing the number of books published by top 10 author****

# 

# In[ ]:


plt.figure(figsize=(25,10))
plt.bar(yd.keys().tolist(),yd.tolist(),color="orange")


# ****A barplot representing the top 10 books that are published by maximum number of times****

# In[ ]:


#ad=xd.keys().tolist()
#ab=df[(df['authors']==ad[0]) & (df['average_rating']>3)]
#for i in range(1,10):
 #   ab=(df[(df['authors']==ad[i]) & (df['average_rating']>3)])
#ab.sort_values('average_rating',ascending=False).iloc[:20]
topauthor = df[df['average_rating']>=4]
topauthor = topauthor.groupby('authors')['title'].count().reset_index().sort_values('title', ascending = False).head(10).set_index('authors')
topauthor


# **top 10 authors with the highest number of highest rated books**

# In[ ]:


#zd=df.groupby(['authors'])['title'].value_counts().sort_values(ascending=False).iloc[:10]
#zd
plt.figure(figsize=(10,10))
#plot = sns.countplot(x = topauthor.index, data = topauthor,palette="Set2")
#plt.hist(topauthor['title'],bins=10,color="orange")
ax = sns.barplot(topauthor['title'], topauthor.index, palette='Set2')


# **A barplot representing the authors with the highest number of highest rated books **

# In[ ]:


plt.scatter(x=topauthor["title"],y=topauthor.index)
plt.ylabel('author')
plt.xlabel('Number of books')


# **A scatterplot representing the authors with the highest number of highest rated books**

# In[ ]:


oldestbooks=df.sort_values('publication_date',ascending=True)
oldestbooks=oldestbooks.sort_values('publication_date', ascending = True).head(10).set_index('title')
oldestbooks


# **This table represents the average rating of some of the oldest books reviewed on good reads.
# we can see that though these books have a great average rating.None of these have a 5.0 rating and the review count is comparatively low as well. **

# In[ ]:


plt.figure(figsize=(15,10))
bx = sns.barplot(oldestbooks['average_rating'],oldestbooks.index, palette='Set2')
#plt.hist(oldestbooks['average_rating'],bins=10,color="orange")


# **A barplot representing the average rating of some of the oldest books**

# In[ ]:


ratedbooks=df.sort_values('average_rating',ascending=False)
rb=ratedbooks.groupby('average_rating')
plt.figure(figsize=(15,10))
plt.hist(ratedbooks["average_rating"],bins=35,color="orange")
#plt.scatter(x=ratedbooks["average_rating"])
#rb.head()


# **A histogram showing the average rating that most of the books on goodreads receive**

# **Through all the different plots we can notice that there can be several parameters to analyse the popularity of a book.
# -->A book that is published by multiple publishers may not be the highest rated book.
# -->The author with the highest number of books published may have the average rating for his/her books comparatively low.
# -->Some of the oldest books may have the lowest review count.
# -->Most of the books reviewed on goodreads have an average rating between 4 and 5.
# Thus the popularity of a book varies according to the parameter we focus on.**

# In[ ]:




