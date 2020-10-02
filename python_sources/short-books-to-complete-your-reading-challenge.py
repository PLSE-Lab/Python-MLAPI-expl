#!/usr/bin/env python
# coding: utf-8

# ### It's the end of August and I'm pretty far from completing my annual [Goodreads book reading challenge](https://www.goodreads.com/challenges/8863-2019-reading-challenge). 
# ### I've only finished 7 out of the total 29 books I pledged to read this year.
# ### In this kernel, I'm trying to find the best short books to read to help me finish my challenge.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


books = pd.read_csv("/kaggle/input/goodreadsbooks/books.csv", error_bad_lines=False)
books.head().T


# ### Let's filter books less than or equal to 100 pages with avg ratings of 3 or above

# In[ ]:


short_books = books[books['# num_pages'] <= 100]
short_books = short_books[short_books['average_rating'] > 3.0]
short_books.describe(include = 'all')


# ### Let's further filter this list by the no of ratings and language

# In[ ]:


short_books = short_books.query('ratings_count > 50 and language_code=="eng"')
short_books.shape


# In[ ]:


short_books.sort_values('average_rating', ascending=False)


# In[ ]:


short_books[short_books['# num_pages'] > 0].sort_values('# num_pages')


# In[ ]:




