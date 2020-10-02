#!/usr/bin/env python
# coding: utf-8

# # 1. Importing the required modules:
# * Pandas is used for data analysis
# * seaborn and matplotlib for visualizing

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# # 2. Reading the CSV file
# Note that the file has some corrupted values, thus I'm turning off `error_bad_lines`.

# In[ ]:


df = pd.read_csv('../input/goodreadsbooks/books.csv', error_bad_lines = False)


# # 3. Solving the Task: Book with the highest text reviews

# As it can be seen in the following cell, there are multiple entries for a single book:

# In[ ]:


print("Number of total books: {}".format(df['title'].count()))
print("Number of unique books: {}".format(df['title'].value_counts().count()))


# In the following cell, I'm grouping the records by `title` and then aggregating the sum of `text_reviews_count`. the top 5 books are the shown:

# In[ ]:


most_rated = df.groupby('title')['text_reviews_count'].sum().sort_values(ascending=False).head(5)
print(most_rated)


# Let's visualize the output:

# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot(most_rated, most_rated.index, palette='rocket')

