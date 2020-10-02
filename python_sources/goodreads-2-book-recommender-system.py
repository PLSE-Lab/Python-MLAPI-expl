#!/usr/bin/env python
# coding: utf-8

# # Books Recommender System

# ![](http://labs.criteo.com/wp-content/uploads/2017/08/CustomersWhoBought3.jpg)

# This is the second part of my project on Book Data Analysis and Recommendation Systems. 
# 
# In my first notebook ([The Story of Book](https://www.kaggle.com/omarzaghlol/goodreads-1-the-story-of-book/)), I attempted at narrating the story of book by performing an extensive exploratory data analysis on Books Metadata collected from Goodreads.
# 
# In this notebook, I will attempt at implementing a few recommendation algorithms (Basic Recommender, Content-based and Collaborative Filtering) and try to build an ensemble of these models to come up with our final recommendation system.

# # What's in this kernel?

# - [Importing Libraries and Loading Our Data](#1)
# - [Clean the dataset](#2)
# - [Simple Recommender](#3)
#     - [Top Books](#4)
#     - [Top "Genres" Books](#5)
# - [Content Based Recommender](#6)
#     - [Cosine Similarity](#7)
#     - [Popularity and Ratings](#8)
# - [Collaborative Filtering](#9)
#     - [User Based](#10)
#     - [Item Based](#11)
# - [Hybrid Recommender](#12)
# - [Conclusion](#13)
# - [Save Model](#14)

# # Importing Libraries and Loading Our Data <a id="1"></a> <br>

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


books = pd.read_csv('../input/goodbooks-10k//books.csv')
ratings = pd.read_csv('../input/goodbooks-10k//ratings.csv')
book_tags = pd.read_csv('../input/goodbooks-10k//book_tags.csv')
tags = pd.read_csv('../input/goodbooks-10k//tags.csv')


# # Clean the dataset <a id="2"></a> <br>
# 
# As with nearly any real-life dataset, we need to do some cleaning first. When exploring the data I noticed that for some combinations of user and book there are multiple ratings, while in theory there should only be one (unless users can rate a book several times). Furthermore, for the collaborative filtering it is better to have more ratings per user. So I decided to remove users who have rated fewer than 3 books.

# In[ ]:


books['original_publication_year'] = books['original_publication_year'].fillna(-1).apply(lambda x: int(x) if x != -1 else -1)


# In[ ]:


ratings_rmv_duplicates = ratings.drop_duplicates()
unwanted_users = ratings_rmv_duplicates.groupby('user_id')['user_id'].count()
unwanted_users = unwanted_users[unwanted_users < 3]
unwanted_ratings = ratings_rmv_duplicates[ratings_rmv_duplicates.user_id.isin(unwanted_users.index)]
new_ratings = ratings_rmv_duplicates.drop(unwanted_ratings.index)


# In[ ]:


new_ratings['title'] = books.set_index('id').title.loc[new_ratings.book_id].values


# In[ ]:


new_ratings.head(10)


# # Simple Recommender <a id="3"></a> <br>
# 
# The Simple Recommender offers generalized recommnendations to every user based on book popularity and (sometimes) genre. The basic idea behind this recommender is that books that are more popular and more critically acclaimed will have a higher probability of being liked by the average audience. This model does not give personalized recommendations based on the user.
# 
# The implementation of this model is extremely trivial. All we have to do is sort our books based on ratings and popularity and display the top books of our list. As an added step, we can pass in a genre argument to get the top books of a particular genre.
# 

# I will use IMDB's *weighted rating* formula to construct my chart. Mathematically, it is represented as follows:
# 
# Weighted Rating (WR) = $(\frac{v}{v + m} . R) + (\frac{m}{v + m} . C)$
# 
# where,
# * *v* is the number of ratings for the book
# * *m* is the minimum ratings required to be listed in the chart
# * *R* is the average rating of the book
# * *C* is the mean rating across the whole report
# 
# The next step is to determine an appropriate value for *m*, the minimum ratings required to be listed in the chart. We will use **95th percentile** as our cutoff. In other words, for a book to feature in the charts, it must have more ratings than at least 95% of the books in the list.
# 
# I will build our overall Top 250 Chart and will define a function to build charts for a particular genre. Let's begin!

# In[ ]:


v = books['ratings_count']
m = books['ratings_count'].quantile(0.95)
R = books['average_rating']
C = books['average_rating'].mean()
W = (R*v + C*m) / (v + m)


# In[ ]:


books['weighted_rating'] = W


# In[ ]:


qualified  = books.sort_values('weighted_rating', ascending=False).head(250)


# ## Top Books <a id="4"></a> <br>

# In[ ]:


qualified[['title', 'authors', 'average_rating', 'weighted_rating']].head(15)


# We see that J.K. Rowling's **Harry Potter** Books occur at the very top of our chart. The chart also indicates a strong bias of Goodreads Users towards particular genres and authors. 
# 
# Let us now construct our function that builds charts for particular genres. For this, we will use relax our default conditions to the **85th** percentile instead of 95. 

# ## Top "Genres" Books <a id="5"></a> <br>

# In[ ]:


book_tags.head()


# In[ ]:


tags.head()


# In[ ]:


genres = ["Art", "Biography", "Business", "Chick Lit", "Children's", "Christian", "Classics",
          "Comics", "Contemporary", "Cookbooks", "Crime", "Ebooks", "Fantasy", "Fiction",
          "Gay and Lesbian", "Graphic Novels", "Historical Fiction", "History", "Horror",
          "Humor and Comedy", "Manga", "Memoir", "Music", "Mystery", "Nonfiction", "Paranormal",
          "Philosophy", "Poetry", "Psychology", "Religion", "Romance", "Science", "Science Fiction", 
          "Self Help", "Suspense", "Spirituality", "Sports", "Thriller", "Travel", "Young Adult"]


# In[ ]:


genres = list(map(str.lower, genres))
genres[:4]


# In[ ]:


available_genres = tags.loc[tags.tag_name.str.lower().isin(genres)]


# In[ ]:


available_genres.head()


# In[ ]:


available_genres_books = book_tags[book_tags.tag_id.isin(available_genres.tag_id)]


# In[ ]:


print('There are {} books that are tagged with above genres'.format(available_genres_books.shape[0]))


# In[ ]:


available_genres_books.head()


# In[ ]:


available_genres_books['genre'] = available_genres.tag_name.loc[available_genres_books.tag_id].values
available_genres_books.head()


# In[ ]:


def build_chart(genre, percentile=0.85):
    df = available_genres_books[available_genres_books['genre'] == genre.lower()]
    qualified = books.set_index('book_id').loc[df.goodreads_book_id]

    v = qualified['ratings_count']
    m = qualified['ratings_count'].quantile(percentile)
    R = qualified['average_rating']
    C = qualified['average_rating'].mean()
    qualified['weighted_rating'] = (R*v + C*m) / (v + m)

    qualified.sort_values('weighted_rating', ascending=False, inplace=True)
    return qualified


# Let us see our method in action by displaying the Top 15 Fiction Books (Fiction almost didn't feature at all in our Generic Top Chart despite being one of the most popular movie genres).

# In[ ]:


cols = ['title','authors','original_publication_year','average_rating','ratings_count','work_text_reviews_count','weighted_rating']


# In[ ]:


genre = 'Fiction'
build_chart(genre)[cols].head(15)


# For simplicity, you can just pass the index of the wanted genre from below. 

# In[ ]:


list(enumerate(available_genres.tag_name))


# In[ ]:


idx = 24  # romance
build_chart(list(available_genres.tag_name)[idx])[cols].head(15)


# # Content Based Recommender <a id="6"></a> <br>
# 
# ![](https://miro.medium.com/max/828/1*1b-yMSGZ1HfxvHiJCiPV7Q.png)
# 
# The recommender we built in the previous section suffers some severe limitations. For one, it gives the same recommendation to everyone, regardless of the user's personal taste. If a person who loves business books (and hates fiction) were to look at our Top 15 Chart, s/he wouldn't probably like most of the books. If s/he were to go one step further and look at our charts by genre, s/he wouldn't still be getting the best recommendations.
# 
# For instance, consider a person who loves *The Fault in Our Stars*, *Twilight*. One inference we can obtain is that the person loves the romaintic books. Even if s/he were to access the romance chart, s/he wouldn't find these as the top recommendations.
# 
# To personalise our recommendations more, I am going to build an engine that computes similarity between movies based on certain metrics and suggests books that are most similar to a particular book that a user liked. Since we will be using book metadata (or content) to build this engine, this also known as **Content Based Filtering.**
# 
# I will build this recommender based on book's *Title*, *Authors* and *Genres*.

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


# My approach to building the recommender is going to be extremely *hacky*. These are steps I plan to do:
# 1. **Strip Spaces and Convert to Lowercase** from authors. This way, our engine will not confuse between **Stephen Covey** and **Stephen King**.
# 2. Combining books with their corresponding **genres** .
# 2. I then use a **Count Vectorizer** to create our count matrix.
# 
# Finally, we calculate the cosine similarities and return books that are most similar.

# In[ ]:


books['authors'] = books['authors'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x.split(', ')])


# In[ ]:


def get_genres(x):
    t = book_tags[book_tags.goodreads_book_id==x]
    return [i.lower().replace(" ", "") for i in tags.tag_name.loc[t.tag_id].values]


# In[ ]:


books['genres'] = books.book_id.apply(get_genres)


# In[ ]:


books['soup'] = books.apply(lambda x: ' '.join([x['title']] + x['authors'] + x['genres']), axis=1)


# In[ ]:


books.soup.head()


# In[ ]:


count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(books['soup'])


# ## Cosine Similarity <a id="7"></a> <br>
# 
# I will be using the Cosine Similarity to calculate a numeric quantity that denotes the similarity between two books. Mathematically, it is defined as follows:
# 
# $cosine(x,y) = \frac{x. y^\intercal}{||x||.||y||} $
# 
# 

# In[ ]:


cosine_sim = cosine_similarity(count_matrix, count_matrix)


# In[ ]:


indices = pd.Series(books.index, index=books['title'])
titles = books['title']


# In[ ]:


def get_recommendations(title, n=10):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    book_indices = [i[0] for i in sim_scores]
    return list(titles.iloc[book_indices].values)[:n]


# In[ ]:


get_recommendations("The One Minute Manager")


# What if I want a specific book but I can't remember it's full name!!
# 
# So I created the following *method* to get book titles from a **partial** title.

# In[ ]:


def get_name_from_partial(title):
    return list(books.title[books.title.str.lower().str.contains(title) == True].values)


# In[ ]:


title = "business"
l = get_name_from_partial(title)
list(enumerate(l))


# In[ ]:


get_recommendations(l[1])


# ## Popularity and Ratings <a id="8"></a> <br>
# 
# One thing that we notice about our recommendation system is that it recommends books regardless of ratings and popularity. It is true that ***Across the River and Into the Trees*** and ***The Old Man and the Sea*** were written by **Ernest Hemingway**, but the former one was cnosidered a bad (not the worst) book that shouldn't be recommended to anyone, since that most people hated the book for it's static plot and overwrought emotion.
# 
# Therefore, we will add a mechanism to remove bad books and return books which are popular and have had a good critical response.
# 
# I will take the top 30 movies based on similarity scores and calculate the vote of the 60th percentile book. Then, using this as the value of $m$, we will calculate the weighted rating of each book using IMDB's formula like we did in the Simple Recommender section.

# In[ ]:


def improved_recommendations(title, n=10):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    book_indices = [i[0] for i in sim_scores]
    df = books.iloc[book_indices][['title', 'ratings_count', 'average_rating', 'weighted_rating']]

    v = df['ratings_count']
    m = df['ratings_count'].quantile(0.60)
    R = df['average_rating']
    C = df['average_rating'].mean()
    df['weighted_rating'] = (R*v + C*m) / (v + m)
    
    qualified = df[df['ratings_count'] >= m]
    qualified = qualified.sort_values('weighted_rating', ascending=False)
    return qualified.head(n)


# In[ ]:


improved_recommendations("The One Minute Manager")


# In[ ]:


improved_recommendations(l[1])


# I think the sorting of similar is more better now than before.
# Therefore, we will conclude our Content Based Recommender section here and come back to it when we build a hybrid engine.
# 

# # Collaborative Filtering <a id="9"></a> <br>
# 
# ![](https://miro.medium.com/max/706/1*DYJ-HQnOVvmm5suNtqV3Jw.png)
# 
# Our content based engine suffers from some severe limitations. It is only capable of suggesting books which are *close* to a certain book. That is, it is not capable of capturing tastes and providing recommendations across genres.
# 
# Also, the engine that we built is not really personal in that it doesn't capture the personal tastes and biases of a user. Anyone querying our engine for recommendations based on a book will receive the same recommendations for that book, regardless of who s/he is.
# 
# Therefore, in this section, we will use a technique called **Collaborative Filtering** to make recommendations to Book Readers. Collaborative Filtering is based on the idea that users similar to a me can be used to predict how much I will like a particular product or service those users have used/experienced but I have not.
# 
# I will not be implementing Collaborative Filtering from scratch. Instead, I will use the **Surprise** library that used extremely powerful algorithms like **Singular Value Decomposition (SVD)** to minimise RMSE (Root Mean Square Error) and give great recommendations.

# There are two classes of Collaborative Filtering:
# ![](https://miro.medium.com/max/1280/1*QvhetbRjCr1vryTch_2HZQ.jpeg)
# - **User-based**, which measures the similarity between target users and other users.
# - **Item-based**, which measures the similarity between the items that target users rate or interact with and other items.

# ## - User Based <a id="10"></a> <br>

# In[ ]:


# ! pip install surprise


# In[ ]:


from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate


# In[ ]:


reader = Reader()
data = Dataset.load_from_df(new_ratings[['user_id', 'book_id', 'rating']], reader)


# In[ ]:


svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'])


# We get a mean **Root Mean Sqaure Error** of about 0.8419 which is more than good enough for our case. Let us now train on our dataset and arrive at predictions.

# In[ ]:


trainset = data.build_full_trainset()
svd.fit(trainset);


# Let us pick users 10 and check the ratings s/he has given.

# In[ ]:


new_ratings[new_ratings['user_id'] == 10]


# In[ ]:


svd.predict(10, 1506)


# For book with ID 1506, we get an estimated prediction of **3.393**. One startling feature of this recommender system is that it doesn't care what the book is (or what it contains). It works purely on the basis of an assigned book ID and tries to predict ratings based on how the other users have predicted the book.

# ## - Item Based <a id="11"></a> <br>

# Here we will build a table for users with their corresponding ratings for each book. 

# In[ ]:


# bookmat = new_ratings.groupby(['user_id', 'title'])['rating'].mean().unstack()
bookmat = new_ratings.pivot_table(index='user_id', columns='title', values='rating')
bookmat.head()


# In[ ]:


def get_similar(title, mat):
    title_user_ratings = mat[title]
    similar_to_title = mat.corrwith(title_user_ratings)
    corr_title = pd.DataFrame(similar_to_title, columns=['correlation'])
    corr_title.dropna(inplace=True)
    corr_title.sort_values('correlation', ascending=False, inplace=True)
    return corr_title


# In[ ]:


title = "Twilight (Twilight, #1)"
smlr = get_similar(title, bookmat)


# In[ ]:


smlr.head(10)


# Ok, we got similar books, but we need to filter them by their *ratings_count*.

# In[ ]:


smlr = smlr.join(books.set_index('title')['ratings_count'])
smlr.head()


# Get similar books with at least 500k ratings.

# In[ ]:


smlr[smlr.ratings_count > 5e5].sort_values('correlation', ascending=False).head(10)


# That's more interesting and reasonable result, since we could get *Twilight* book series in our top results. 

# # Hybrid Recommender <a id="12"></a> <br>
# 
# ![](https://www.toonpool.com/user/250/files/hybrid_20095.jpg)
# 
# In this section, I will try to build a simple hybrid recommender that brings together techniques we have implemented in the content based and collaborative filter based engines. This is how it will work:
# 
# * **Input:** User ID and the Title of a Book
# * **Output:** Similar books sorted on the basis of expected ratings by that particular user.

# In[ ]:


def hybrid(user_id, title, n=10):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:51]
    book_indices = [i[0] for i in sim_scores]
    
    df = books.iloc[book_indices][['book_id', 'title', 'original_publication_year', 'ratings_count', 'average_rating']]
    df['est'] = df['book_id'].apply(lambda x: svd.predict(user_id, x).est)
    df = df.sort_values('est', ascending=False)
    return df.head(n)


# In[ ]:


hybrid(4, 'Eat, Pray, Love')


# In[ ]:


hybrid(10, 'Eat, Pray, Love')


# We see that for our hybrid recommender, we get (almost) different recommendations for different users although the book is the same. But maybe we can make it better through following steps:
# 1. Use our *improved_recommendations* technique , that we used in the **Content Based** seciton above
# 2. Combine it with the user *estimations*, by dividing their summation by 2
# 3. Finally, put the result into a new feature ***score***

# In[ ]:


def improved_hybrid(user_id, title, n=10):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:51]
    book_indices = [i[0] for i in sim_scores]
    
    df = books.iloc[book_indices][['book_id', 'title', 'ratings_count', 'average_rating', 'original_publication_year']]
    v = df['ratings_count']
    m = df['ratings_count'].quantile(0.60)
    R = df['average_rating']
    C = df['average_rating'].mean()
    df['weighted_rating'] = (R*v + C*m) / (v + m)
    
    df['est'] = df['book_id'].apply(lambda x: svd.predict(user_id, x).est)
    
    df['score'] = (df['est'] + df['weighted_rating']) / 2
    df = df.sort_values('score', ascending=False)
    return df[['book_id', 'title', 'original_publication_year', 'ratings_count', 'average_rating', 'score']].head(n)


# In[ ]:


improved_hybrid(4, 'Eat, Pray, Love')


# In[ ]:


improved_hybrid(10, 'Eat, Pray, Love')


# Ok, we see that the new results make more sense, besides to, the recommendations are more personalized and tailored towards particular users.

# # Conclusion <a id="13"></a> <br>
# 
# In this notebook, I have built 4 different recommendation engines based on different ideas and algorithms. They are as follows:
# 
# 1. **Simple Recommender:** This system used overall Goodreads Ratings Count and Rating Averages to build Top Books Charts, in general and for a specific genre. The IMDB Weighted Rating System was used to calculate ratings on which the sorting was finally performed.
# 2. **Content Based Recommender:** We built content based engines that took book title, authors and genres as input to come up with predictions. We also deviced a simple filter to give greater preference to books with more votes and higher ratings.
# 3. **Collaborative Filtering:** We built two Collaborative Filters; 
#   - one that uses the powerful Surprise Library to build an **user-based** filter based on single value decomposition, since the RMSE obtained was less than 1, and the engine gave estimated ratings for a given user and book.
#   - And the other (**item-based**) which built a pivot table for users ratings corresponding to each book, and the engine gave similar books for a given book.
# 4. **Hybrid Engine:** We brought together ideas from content and collaborative filterting to build an engine that gave book suggestions to a particular user based on the estimated ratings that it had internally calculated for that user.
# 
# Previous -> [The Story of Book](https://www.kaggle.com/omarzaghlol/goodreads-1-the-story-of-book/)

# In[ ]:




