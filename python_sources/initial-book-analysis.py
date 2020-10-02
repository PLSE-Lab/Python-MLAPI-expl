#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
INPUT_DIR = "../input"
DATA_LOC = f"{INPUT_DIR}/books.csv"


# In[ ]:


books = pd.read_csv(DATA_LOC, index_col=0, error_bad_lines=False)
books.columns


# # Handling Erroneous Rows
# As indicated by the output above, there are 5 rows that don't fit the structure of the rest of the data. For now, we can ignore these since it is so small.

# In[ ]:


books.head()


# # Seeing the errors
# We can see one of the errors by having pandas read just one of the erroneous rows as indicated in the error message above

# In[ ]:


error_row = pd.read_csv(DATA_LOC, skiprows=4011, nrows=1, header=None)


# In[ ]:


error_row


# In[ ]:


books.shape


# In[ ]:


print('Before:\n', books.language_code.value_counts())
books['language_code'] = np.where(books.language_code.str.contains('en'), 'eng', books.language_code)
print('After:\n', books.language_code.value_counts())


# In[ ]:


import matplotlib.pyplot as plt

average_rating_by_language = books[['language_code', 'average_rating']].groupby(['language_code']).mean().sort_values(by='average_rating', ascending=False)
counts_by_language = books.language_code.value_counts()
languages_gt = counts_by_language[counts_by_language > 20]
books_language_gt = books[books.language_code.isin(languages_gt.index)]

fig = plt.figure()
books_language_gt.boxplot(column='average_rating', by='language_code', figsize=(15, 8), rot=60)
plt.show()


# In[ ]:


# select all as Series
author_counts = books_language_gt.authors.value_counts().sort_values(axis='index').reset_index().rename(columns={'authors': 'counts', 'index': 'authors'})
author_sums = books_language_gt[['authors', 'average_rating']].groupby(['authors']).sum()['average_rating'].sort_values(axis='index').reset_index()
all_values = author_counts.merge(author_sums).sort_values(by='average_rating', ascending=False)
all_values['author_average_rating'] = all_values['average_rating'] / all_values['counts']
all_values.head()


# In[ ]:


plot_df = all_values.head(10)

plt.figure()
x_lab = plot_df.authors
x_ax = range(len(x_lab))
plt.barh(x_ax, plot_df.author_average_rating)
plt.yticks(x_ax, x_lab)
plt.show()


# In[ ]:


# https://matplotlib.org/3.1.0/gallery/color/named_colors.html
import matplotlib.colors as mcolors
import random

random.seed(42)
language_codes = books_language_gt.language_code.unique()
unique_language_codes = len(language_codes)
color_list = list(mcolors.CSS4_COLORS.keys())
shuffled_list = list(range(len(color_list)))
random.shuffle(shuffled_list)
fig = plt.figure(figsize=(10, 7))
for lang_code, color_idx in zip(language_codes, shuffled_list):
    subset = books_language_gt.loc[books_language_gt.language_code==lang_code]
    plt.scatter(subset['# num_pages'], subset.average_rating, c=color_list[color_idx], alpha=0.5)
plt.legend(books_language_gt.language_code.unique())
plt.show()


# In[ ]:


random.seed(42)
language_codes = books_language_gt.language_code.unique()
unique_language_codes = len(language_codes)
color_list = list(mcolors.CSS4_COLORS.keys())
shuffled_list = list(range(len(color_list)))
random.shuffle(shuffled_list)
fig = plt.figure(figsize=(15, 7))
for lang_code, color_idx in zip(language_codes, shuffled_list):
    subset = books_language_gt.loc[books_language_gt.language_code==lang_code]
    sizes = subset.average_rating.apply(lambda x: (x ** x) / 2)
    plt.scatter(subset['# num_pages'], subset.text_reviews_count, c=color_list[color_idx], alpha=0.2, s=sizes.values)

plt.legend(books_language_gt.language_code.unique())
plt.xlabel('Number of Pages')
plt.ylabel('Number of Text Reviews')
plt.title('Book Reviews by Language:\nPages vs. Text Reviews\nSized by Rating Score')
plt.show()


# The difficult part with the chart above is that we don't get a good sense of book ratings by language code because they are too clustered and the datapoint sizes don't look that different. The only thing we can gather is that if a book has a high number of reviews, it is always a book written in English.

# In[ ]:


books_language_gt.groupby('language_code').describe()


# In[ ]:




