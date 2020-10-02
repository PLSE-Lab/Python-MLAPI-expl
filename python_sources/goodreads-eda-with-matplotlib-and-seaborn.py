#!/usr/bin/env python
# coding: utf-8

# No EDA is complete without visualization. The intent of this kernel is to get started on matplotlib and seaborn using the Goodreads dataset. These two libraries offer pretty charts and graphs - which are just about enough to get anyone started on creating pretty charts and graphs.
# 
# I started my visualization journey using matplotlib, and then moved to seaborn and pyplot and altair. I'm hoping this kernel helps anyone who's struggling to get the initial vivisualization done.
# 
# For each visualization, I will also be using a different style sheet for just exploratory reasons. We need pretty pictures!!  :)

# ### Importing Relevant Libraries

# In[ ]:


import numpy as np 
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ### Reading Data

# A simple, pd.read_csv on the input file fails as there are some discrepancies in the file. To ignore this, we will be using books =  pd.read_csv('goodreads_books.csv',error_bad_lines=False)

# In[ ]:


books = pd.read_csv('../input/books.csv', error_bad_lines=False)


# ### Data Exploration

# In[ ]:


books.head()


# In[ ]:


books.info()


# ### Data Visualization

# **Top Ten Books : Based on Average Rating**
# 
# To get this, let's imagine of how we'd want to presernt this data. I think, I'd want the book names to be stacked horizontally, i.e. the book names will be on the Y axis. The books should be bar shaped where each bar represents a book.
# The ratings should be on the X axis. Since the maximum average rating our book can have is 5, we will limit out values on X axis using this information. 
# 
# Great! We have a picture of our graph in our heads. Let's get it in our notebooks. 

# In[ ]:


### create the df
avg_rating_df = books.sort_values(['average_rating'], ascending=False).head(10)

### set figure size and axes
fig, ax = plt.subplots(figsize = (15,15))

### declare our x and y components
y_books = np.arange(len(avg_rating_df))
rating = avg_rating_df.average_rating.max()

### plot them and give necessary texts 
plt.style.use(['fivethirtyeight'])
ax.barh(y_books, rating, align='center', tick_label = avg_rating_df.title)
ax.invert_yaxis()  
ax.set_xlabel('Average Rating')
ax.set_title('Top 10 Books Based on Average Ratings')


# Ah! All the books have an average rating of 5. The Complete Theory Fun Factory is a children's book filled with music theory puzzles and games, with helpful hints from a motley crew of cartoon characters. No wonder it's got an average of 5. Who wouldnt want to learn from cartoon characters?
# 
# But to understand what people really liked, we'd have to choose a different column and not average rating. Let's take the ratings count column and see what are the top ten books most preferred by readers. 
# 
# I'd want to visualize this like the above, horizontally stacked as the book names can be read. But this time, I will use the seaborn bar plot

# **Top 10 Books: Most Rated **

# In[ ]:


### get df
most_rated = books.sort_values('ratings_count', ascending = False).head(10).set_index('title')

### set style
sns.set_style("darkgrid")

### initialize the matplotlib figure
fig, ax = plt.subplots(figsize=(10, 15))

### plot
sns.set_color_codes("pastel")
sns.barplot(most_rated['ratings_count'], most_rated.index, label="Total", color="b")
ax.set(ylabel="Books",xlabel="Rating Counts")
ax.set_title('Top 10 Books Based on User Ratings')


# This chart helps! It also has three of my favourite reads - LOTR1, Catcher in the Rye, and Animal Farm. 
# 
# Now that we're comfortable with this, how about we look at authors? Let's look at the top ten rated authors we have. And let's make the bar chart a bit colourful. For this, I will include the palette in parameters. 
# 
# **Top Ten Authors: Most Rated (Avg Rating > 4)**

# In[ ]:


### get df
top_10_authors = books[books['average_rating']>=4]
top_10_authors = top_10_authors.groupby('authors')['title'].count().reset_index().sort_values('title', ascending = False).head(10).set_index('authors')

### set style
sns.set_style("ticks")

### set figure size
plt.figure(figsize=(15,10))

### plot
ax = sns.barplot(top_10_authors['title'], top_10_authors.index, palette='muted')
ax.set_xlabel("# Books")
ax.set_ylabel("Authors")
ax.set_title('Top 10 Most Rated Authors')
for i in ax.patches:
    ax.text(i.get_width()+.2, i.get_y()+0.3, str(round(i.get_width())), fontsize = 10, color = 'k')


# Moving on to our last one. This time, lets make a pie chart. We've had enough of bars. What can we show in our pie charts? The data shouldnt be have some finite unique values for our pie to capture them. How about looking at the distinct spread of books across the top 5 languages? We'll chose 5 as if we do a book.language_code.unique() we get a long list of counts. Let's just stick to top 5. 
# 
# 
# **Books Available : Top 5 Languages**

# In[ ]:


### get df
lang_counts = pd.DataFrame(books.language_code.value_counts())
lang_counts = lang_counts.reset_index()
lang_counts = lang_counts.rename(columns={"index": "lang_code", "language_code": "counts"})
top_5_lang = lang_counts.sort_values(['counts'], ascending=False).head(5)

### set values
labels = top_5_lang.lang_code
counts = top_5_lang.counts
explode = (.05, 0, 0, 0,0) ### I want the most prominent language to be a little set apart when i view the pie.

### set figure and plot
fig, ax = plt.subplots(figsize = (20,20))
ax.pie(counts,explode=explode, labels=labels, autopct='%1.f%%', shadow=False, startangle=90)
ax.axis('equal')
ax.set_title('Top 5 Languages')

plt.show()


# Yay!!! We have our pie. It shows 80% of the books are in English, followed  by en-US and Spanish.
# 
# I hope by the end of this, we've worked on a dataset and also learnt some basics of visualization. 
# 
# This completes this kernel.

# In[ ]:




