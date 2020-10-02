#!/usr/bin/env python
# coding: utf-8

# # Goodreads-books
# 
# In this notebook we will try to answer some simple questions about the [Goodreads-books dataset](https://www.kaggle.com/jealousleopard/goodreadsbooks). 
# 
# 
# 1. Show which book has better Rating.
# 2. Show which book has the highest number of reviews with the best Rating.
# 3. Clearly display your solutions in graphics.
# 
# let's do it.

# ## Packages
# 
# The libraries we are using in this case will be pandas for dataset management and plotly, matplotlib and seaborn for visualization.

# In[ ]:


import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Read Dataset
# To avoid the error "Error tokenizing data" we must put the argument error_bad_lines = False to skip the rows that cause us problems (They have more columns than most rows).
# 

# In[ ]:


df = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv', error_bad_lines=False)


# Let's take a look at the structure of the dataset

# In[ ]:


df.head()


# Let's review the size of the dataset.

# In[ ]:


df.shape


# It is important to always check if we have NA values

# In[ ]:


df.isna().sum(axis=0)


# All right! we don't have null values
# 

# Let's review the data types we have to see if they have the correct type. In this case we will focus on title, average_rating and text_reviews_count

# In[ ]:


df.dtypes


# We see that there are repeated books. Then we will have to add to process the data.

# In[ ]:


df.where(df['title'] == 'The Known World').count()


# Let's check if there are books that do not have a review count, but a rating.

# In[ ]:


books_with_reviews = df[df['text_reviews_count'] > 0]
books_with_no_reviews = df[df['text_reviews_count'] == 0]


# In[ ]:


books_with_no_reviews.shape[0]


# In[ ]:


ax = sns.jointplot(data= books_with_no_reviews, x='text_reviews_count', y='average_rating')
ax.set_axis_labels('Reviews Count', 'Averange Rating')
plt.title('Books with no reviews')


# We can see that there are ratings even when there are no reviews. Then we will keep the books that have reviews.

# Let's start answering our questions.
# 1. Show which book has better punctuation.
# 

# In[ ]:


best_rating = books_with_reviews.groupby('title')['average_rating'].mean().sort_values(ascending=False)
best_rating = pd.DataFrame(best_rating)
best_rating


# We can see that the highest rated books have a 5.0 rating. We will keep the books that have a rating of 5

# In[ ]:


best_rating = best_rating.loc[best_rating['average_rating'] == 5.0]
best_rating = best_rating.reset_index()
print(f'total: {best_rating.shape[0]}')
best_rating


# We have 6 books with this characteristic (rating of 5).

# Now let's graph our results.

# In[ ]:


fig = px.bar(best_rating, title='Highest rated books', x='title', y='average_rating', text='average_rating', labels={'title':'Books Title', 'average_rating': 'Rating'})
fig.show()


# Great! Now let's continue with our second question:
# 
# 
# 2. Show which book has the highest number of reviews with the best score.

# We get our books, the rating and the number of reviews.
# 
# We are left with the first 10 to graph them.

# In[ ]:


best_reviews = books_with_reviews.groupby('title').agg({'average_rating': ['mean'], 'text_reviews_count': ['sum']})
best_reviews.columns = ['average_rating', 'text_reviews_count']
best_reviews = best_reviews.reset_index()
best_reviews = best_reviews.sort_values(by=['text_reviews_count'], ascending=False).head(10)
best_reviews


# Let's graph the results

# In[ ]:


fig = px.scatter(best_reviews, x="text_reviews_count", y="average_rating", hover_data=['title'], labels={'text_reviews_count': 'Reviews Count', 'average_rating': 'Average Rating'})
fig.update_layout(title='Reviews Count vs Rating')
fig.show()


# With this we answer the questions.
# 
# A thank you to [deenafrancis](https://www.kaggle.com/deenafrancis) with [his work](https://www.kaggle.com/deenafrancis/books-ratings-and-insights#kln-65) I based myself to make this notebook.
