#!/usr/bin/env python
# coding: utf-8

# # Building a Movie Recommendation Model 

# Using fast.ai I'm going to use collaborative filtering to train a neural network to recommend films based on theirs and others likes and dislikes.

# ## Preparing The Data

# In[ ]:


import numpy as np
import pandas as pd
from fastai.tabular import *
from fastai.collab import *


# In[ ]:


movies = pd.read_csv('/kaggle/input/movietweetings/movies.dat', delimiter='::', engine='python', header=None, names = ['Movie ID', 'Movie Title', 'Genre'])
users = pd.read_csv('/kaggle/input/movietweetings/users.dat', delimiter='::', engine='python', header=None, names = ['User ID', 'Twitter ID'])
ratings = pd.read_csv('/kaggle/input/movietweetings/ratings.dat', delimiter='::', engine='python', header=None, names = ['User ID', 'Movie ID', 'Rating', 'Rating Timestamp'])
movies.head()


# In[ ]:


users.head()


# In[ ]:


ratings.head()


# In[ ]:


for i in [movies, users, ratings]:
    print(i.shape)


# In[ ]:


df = ratings.merge(movies[['Movie ID','Movie Title']], on='Movie ID')


# In[ ]:


df = df.rename(columns={'User ID':'userID','Movie ID':'movieID','Rating':'rating','Rating Timestamp':'timestamp', 'Movie Title': 'title'})
df.head()


# Our dataframe is just about ready to use. All we need to do is normalise the ratings so they are between 0 and 5 so they can work with the fast.ai learner.

# In[ ]:


df.rating = df.rating/2.0


# ## Training The Model

# In[ ]:


data = CollabDataBunch.from_df(df, seed=42, valid_pct=0.1, item_name='title')


# Create the DataBunch, with 10% used for validation.

# In[ ]:


data.show_batch()


# show_batch lets you peak at the data. 
# This just allows you to make sure your data is being interpreted correctly by CollabDataBunch.

# In[ ]:


y_range = [0,5.5]


# Since this learning model is using a sigmoid function which asymptotes at the end meaning if you plotted this function on a graph from 0 to 5, it never actually reaches 5 so we need to aim a little higher then that.

# In[ ]:


learn = collab_learner(data, n_factors=40, y_range=y_range, wd=1e-1)


# n_factors is just the embedding size. I had no idea what size would be best so I looped through 20,30,40,60,80 and found 40 to give the smallest loss.

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, 1e-3)


# Running for five epochs gave a RMSE of 0.66 which is a MSE of 0.43. That's to say for a rating out of ten, the model can predict on average within 0.87 points, which is really quite impressive.

# In[ ]:


learn.save('dotprod')


# Saving and reloading the model so we won't have to retrain each time we comeback to the notebook.

# ## Interpreting The Model

# In[ ]:


learn.load('dotprod');


# In[ ]:


g = df.groupby('title')['rating'].count()
top_movies = g.sort_values(ascending=False).index.values[:1000]
top_movies[:10]


# Movies that have been rated the highest number of times.
# Gravity being the highest with 3066 votes.

# In[ ]:


movie_bias = learn.bias(top_movies, is_item=True)
movie_bias.shape


# All film ratings have some form of bias attached to them. This is because no film is inherently good or bad, and are based on people's own preferances. I may like action films more than most, therefore films with action in have my bias attached to it's rating. We need a way to unbias them, by adding a bias to our model we can effectively try to cancel this out.

# In[ ]:


mean_ratings = df.groupby('title')['rating'].mean()*2
movie_ratings = [(b, i, mean_ratings.loc[i]) for i,b in zip(top_movies,movie_bias)]


# movie_ratings provides a list of each of the 1000 films in top_movies, with their bias, title, and what there mean rating is.

# In[ ]:


item0 = lambda o:o[0]


# In[ ]:


sorted(movie_ratings, key=item0)[:15]


# These films have the lowest bias, meaning they are some of the least liked. That means when it looks at recommending one of these films to someone because it's very familier to a bunch of other films that person liked it'll not rate it as high as it would without the attached bias because it understands that this is a very unlikable film.
# 
# What's interesting is the film The Thin Red Line. It's rated 7.6 on IMBD but Twitter users are voting it a measley 3.5. Let's look into this.

# In[ ]:


df2 = df.copy()
df2.rating = df2.rating*2


# In[ ]:


df2[df2.title == 'The Thin Red Line (1998)'].groupby('rating')['title'].count()


# Looks like a 168 people rated this film 1/10 for some reason.

# In[ ]:


sorted(movie_ratings, key=lambda o: o[0], reverse=True)[:15]


# Be Somebody, Joker, and Shawshank Redemption have quite a high bias. This means we can expect to see these films suggested the most often.
# 
# Be Somebody is rated 5.6 on imdb though so lets also look into movie.

# In[ ]:


df2[df2.title == 'Be Somebody (2016)'].groupby('rating')['title'].count()


# Again, all except one person who rated 'Be Somebody' rated it a 10. Why though?

# In[ ]:


movie_w = learn.weight(top_movies, is_item=True)
movie_w.shape


# In[ ]:


movie_pca = movie_w.pca(3)
movie_pca.shape


# To try and find some patterns in taste we can look at the weights, of which there are 40x1000. That's beacuse we used 40 factors for the training model and there's 1000 movies.
# We are going to cram these 40 factors into just 3 using Principal Components Analysis.
# I'm told this is done with a simple linear transformation that takes an input matrix and tries to find a smaller number of columns that cover a lot of the space in that original matrix.

# In[ ]:


fac0,fac1,fac2 = movie_pca.t()
movie_comp = [(f, i) for f,i in zip(fac0, top_movies)]


# In[ ]:


sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]


# In[ ]:


sorted(movie_comp, key=itemgetter(0))[:10]


# The first factor appears to group darker hero adventures. People who are fans of these types of films are least likely to want to watch films like Scare Movie 5 and Movie 43.

# In[ ]:


movie_comp = [(f, i) for f,i in zip(fac1, top_movies)]


# In[ ]:


sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]


# In[ ]:


sorted(movie_comp, key=itemgetter(0))[:10]


# The next factor is definitely looking at people who like thriller-horrors, and these fans seem to not be huge lovers of big action films.

# In[ ]:


movie_comp = [(f, i) for f,i in zip(fac2, top_movies)]
sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]


# In[ ]:


sorted(movie_comp, key=itemgetter(0))[:10]


# Lastly it looks like superhero films, where these fans are least interested in more serious slower films.

# In[ ]:


idxs = np.random.choice(len(top_movies), 75, replace=False)
idxs = list(range(75))
X = fac0[idxs]
Y = fac1[idxs]
plt.figure(figsize=(15,15))
plt.scatter(X, Y)
for i, x, y in zip(top_movies[idxs], X, Y):
    plt.text(x,y,i, color=np.random.rand(3)*0.7, fontsize=11)
plt.xlabel("<------- Movies like Movie43 and Scary Movie 5                     [fac0]                            Gritty Heros------>")
plt.ylabel("<------- Big Action                                [fac1]                                         Thriller/Horrors------>")
plt.title("75 Top Movies")
plt.show()


# This is just a random sampling of 75 of the most popular movies, with fac0 and fac1 on the X and Y axis. 

# ## Testing The Model

# We're now going to use the model to suggest some films to one of the users.

# In[ ]:


df.userID.value_counts()[:10]


# This is a list of the top ten users who voted on the most films, and would get the most accurate recommendations when testing the model.

# In[ ]:


learn.export()


# In[ ]:


learn = load_learner('/kaggle/input/export/')


# In[ ]:


h = df.groupby('movieID')['rating'].count()
all_films = h.sort_values(ascending=False).index.values[:10000]


# In[ ]:


def get_top_suggested(user_id):
    user_films = df[df.userID == user_id].movieID.sort_values().values
    unseen_films = [i for i in all_films if i not in user_films]
    user_df = pd.Series(unseen_films, name='movieID').to_frame()
    user_df['userID'] = user_id
    user_df = user_df.reindex(columns=['userID','movieID'])
    user_df = user_df.merge(df[['movieID','title']], on='movieID')
    user_df = user_df.groupby('title').mean().reset_index()
    learn = load_learner('/kaggle/input/export/', test=CollabList.from_df(user_df, cat_names=['userID', 'movieID'], path='/kaggle/input/export/'))
    preds = learn.get_preds(ds_type=DatasetType.Test)
    user_df['rating'] = preds[0]*2
    user_df['rating'] = round(user_df['rating'],1)
    return user_df.sort_values(by='rating', ascending=False).reset_index()[['title','rating']].head(20)


# In[ ]:


get_top_suggested(24249)


# This function finds all the movies in the top 1000 most voted for, which a user has not yet rated on, and lists the top 20 movies it believes that user would rate the highest.

# In[ ]:


def rating_comparison(user_id):
    learn = load_learner('/kaggle/input/export/', test=CollabList.from_df(df[df.userID == 24249], cat_names=['userID', 'movieID'], path='/kaggle/input/export/'))
    preds = learn.get_preds(ds_type=DatasetType.Test)
    df2 = df[df.userID == 24249].copy()
    df2['prediction'] = preds[0]*2
    df2['rating'] = df2['rating']*2
    df2['diff'] = abs(df2['prediction'] - df2['rating'])
    return df2.sort_values(by='prediction', ascending=False).reset_index()[['title','rating','prediction','diff']]


# I'd like to see just how well these predictions are, so this function lists the top 20 movies it thinks a user will like, and compares the difference between the user's actual predicitions.

# In[ ]:


rating_comparison(24249).head(20)


# The largest error in this list of the top 20 looks to be Back to the Future, which the user voted a 7 but our algorithm predicts 8.8. Overall this is extremely good and we can be quite confident in our recommendation model.

# In[ ]:


learn = load_learner('/kaggle/input/export/', test=CollabList.from_df(df, cat_names=['userID', 'movieID'], path='/kaggle/input/export/'))
preds = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


df['prediction'] = preds[0]*2
df['difference'] = abs(df['prediction'] - df['rating']*2)
df2 = df.groupby('userID')[['difference']].agg(['count','mean'])
df2.difference.plot(x='mean',y='count',kind='scatter',figsize=(20,10))


# Now to compare how many films a user has rated with their mean error
