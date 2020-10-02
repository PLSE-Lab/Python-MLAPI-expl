#!/usr/bin/env python
# coding: utf-8

# # Goodreads Books Exploration
# 
# This notebook intends to explore the **goodreadsbooks** dataset in order to determine which books are on top, both in terms of rating and number of reviews
# 
# ![](https://images.pexels.com/photos/159866/books-book-pages-read-literature-159866.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940)

# ## Preprocessing

# In order to be able to retrieve data from the CSV files, an initial cleaning process is necessary: as there are no quotes for lists of values, the file's contents will be formatted by quoting lists of values.

# In[ ]:


import re
import os

try:
    os.remove('/kaggle/working/books_clean.csv')
except OSError:
    pass

pattern = re.compile(r'^(?P<bookID>\d+),(?P<title>.+?(?=,)),(?P<authors>.+(?=,\d+\.))')

wrote_headers = False

with open('../input/goodreadsbooks/books.csv', 'r') as books_file:
  for line in books_file:
    if (not wrote_headers):
      with open('/kaggle/working/books_clean.csv', 'a') as clean_file:
        clean_file.write(line)
      wrote_headers = True

    found = re.match(pattern, line)

    if (found):
      quotified = pattern.sub(r'\g<bookID>,\g<title>,"\g<authors>"', line)
      with open('/kaggle/working/books_clean.csv', 'a') as clean_file:
        clean_file.write(quotified)

print("OK -- Generated /kaggle/working/books_clean.csv")


# ## Extraction

# Reading the CSV file from the dataset

# In[ ]:


import pandas as pd

df = pd.read_csv('/kaggle/working/books_clean.csv', encoding = 'utf-8')

df.head()


# ## Transformation

# In[ ]:


len(df[df['title'].isna()].index)


# In[ ]:


len(df[df['text_reviews_count'].isna()].index)


# No null values found in the columns of interest. Transformation stage skipped

# ## Analysis

# ### Book with most reviews

# In[ ]:


max_reviews_count = df['text_reviews_count'].max()
most_reviewed_book = df[df['text_reviews_count'] == max_reviews_count].iloc[0]
print(f'The book with the highest reviews count is\n\t{most_reviewed_book["title"]}\nwith {max_reviews_count} reviews')


# ### Book with highest rating
# 
# 

# In[ ]:


max_rating = df['average_rating'].max()
best_rated_book = df[df['average_rating'] == max_rating].iloc[0]
print(f'The book with the highest rating is\n\t{best_rated_book["title"]}\n with a score of {max_rating}')


# ### Book with highest rating and number of reviews

# In[ ]:


rating_sorted_df = df.sort_values('average_rating', ascending = False)
rating_grouping = rating_sorted_df.groupby('average_rating', sort = False)
best_group = list(rating_grouping)[0]
best_rating, group_df = best_group
best_book = group_df.sort_values('text_reviews_count', ascending = False).iloc[0]

print(f'The book with the highest rating and number of text_reviews is\n\t{best_book["title"]}\n with a score of {max_rating} and {best_book["text_reviews_count"]} reviews')


# ## Visualization
# 
# The following diagrams give evidence to the previous statements
# 

# ### Books by rating

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

rating_classes = np.array(range(0, 11)) * 0.5

df['rating_ranges'] = pd.cut(df['average_rating'], bins = rating_classes)

diagram = df['rating_ranges'].value_counts().sort_index().plot(kind = 'bar')
diagram.set_xlabel('Rating Groups')
diagram.set_ylabel('Number of Books')
diagram.set_title('Frequency of books by rating groups')
plt.show()

df['rating_ranges'].value_counts()


# ### Books by review counts of the group with highest rating (5.0)

# In[ ]:


counts_df = group_df['text_reviews_count'].value_counts()
diagram = counts_df.sort_index().plot(kind = 'bar', title = 'Number of books with a 5.0 rating, by review count')
diagram.set_xlabel('Number of Reviews')
diagram.set_ylabel('Number of Books')
diagram.set_xticklabels(counts_df.index, rotation = 0)
plt.show()

group_df['text_reviews_count'].value_counts()

