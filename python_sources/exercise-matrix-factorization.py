#!/usr/bin/env python
# coding: utf-8

# # Exercise (Matrix Factorization)
# 
# In this lesson, we'll reuse the model we trained in [the tutorial](https://www.kaggle.com/colinmorris/matrix-factorization). To get started, run the setup cell below to import the libraries we'll be using, load our data into Dataframes, and load a serialized version of the model we trained earlier.

# In[ ]:


import os
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

from learntools.core import binder; binder.bind(globals())
from learntools.embeddings.ex2_factorization import *

input_dir = '../input/movielens-preprocessing'
df = pd.read_csv(os.path.join(input_dir, 'rating.csv'), usecols=['userId', 'movieId', 'rating', 'y'])
movies = pd.read_csv(os.path.join(input_dir, 'movie.csv'), index_col=0)

model_dir = '../input/matrix-factorization'
model_fname = 'factorization_model.h5'
model_path = os.path.join(model_dir, model_fname)
model = keras.models.load_model(model_path)

print("Setup complete!")


# ## Part 1: Generating Recommendations
# 
# At the end of [the first lesson where we built an embedding model](https://www.kaggle.com/colinmorris/embedding-layers), I showed how we could use our model to predict the ratings a particular user would give to some set of movies.
# 
# For reference, here's a (slightly modified) copy of that code where we calculated predicted ratings for 5 specific movies:

# In[ ]:


# Id of the user for whom we're predicting ratings
uid = 26556
candidate_movies = movies[
    movies.title.str.contains('Naked Gun')
    | (movies.title == 'The Sisterhood of the Traveling Pants')
    | (movies.title == 'Lilo & Stitch')
].copy()

preds = model.predict([
    [uid] * len(candidate_movies), # User ids 
    candidate_movies.index, # Movie ids
])
# Because our model was trained on a 'centered' version of rating (subtracting the mean, so that
# the target variable had mean 0), to get the predicted star rating on the original scale, we need
# to add the mean back in.
row0 = df.iloc[0]
offset = row0.rating - row0.y
candidate_movies['predicted_rating'] = preds + offset
candidate_movies.head()[ ['movieId', 'title', 'predicted_rating'] ]


# Suppose we're interested in the somewhat more open-ended problem of **generating recommendations**. i.e. given some user ID and some number `k`, we need to generate a list of `k` movies we think the user will enjoy.
# 
# The most straightforward way to do this would be to calculate the predicted rating this user would assign for *every movie in the dataset*, then take the movies with the `k` highest predictions.
# 
# In the code cell below, fill in the body of the `recommend` function to do this. (Hint: you may want to use the cell above as a reference)

# In[ ]:


def recommend(model, user_id, n=5):
    """Return a DataFrame with the n most highly recommended movies for the user with the
    given id. (Where most highly recommended means having the highest predicted ratings 
    according to the given model).
    The returned DataFrame should have a column for movieId and predicted_rating (it may also have
    other columns).
    """
    all_movie_ids = movies.index 
    preds = model.predict([
        np.repeat(uid, len(all_movie_ids)),
        all_movie_ids,
    ])
    # Add back the offset calculated earlier, to 'uncenter' the ratings, and get back to a [0.5, 5] scale.
    movies.loc[all_movie_ids, 'predicted_rating'] = preds + offset
    reccs = movies.sort_values(by='predicted_rating', ascending=False).head(n)
    return reccs


# In[ ]:


#part1.hint()


# In[ ]:


#part1.solution()


# ## Part 2: Sanity check
# 
# Run the code cell below to get our most highly recommended movies for user #26556.

# In[ ]:


recommend(model, 26556)


# Do these recommendations seem sensible? If you'd like a reminder of user 26556's tastes, run the cell below to see all their ratings (in descending order).

# In[ ]:


uid = 26556
user_ratings = df[df.userId==uid]
movie_cols = ['movieId', 'title', 'genres', 'year', 'n_ratings', 'mean_rating']
user_ratings.sort_values(by='rating', ascending=False).merge(movies[movie_cols], on='movieId')


# Review our top-recommended movies. Are they reasonable? If not, where did we go wrong? You may also find it interesting to look at:
# - The metadata associated with the top-recommended movies
# - The 'least-recommended' movies (the ones with the lowest predicted scores)
# - The actual predicted rating values.
# 
# Once you have an opinion, uncomment the cell below to see if we're in agreement.

# In[ ]:


part2.solution()


# ## Part 3: How are we going to fix this mess?
# 
# How can we improve the problem with our recommendations that we identified in Part 2? This could involve changing our model's structure, our training procedure, or our procedure for generating recommendations given a model.
# 
# Give it some thought, then uncomment the cell below to compare notes with me. (If you have no idea, that's totally fine!)

# In[ ]:


part3.solution()


# ## Part 4: Fixing our obscure recommendation problem (thresholding)
# 
# Fill in the code cell below to implement the `recommend_nonobscure` function, which will recommend the best movies which have at least some minimum number of ratings. (You may wish to modify the code you wrote in `recommend`, or even call `recommend` as a subroutine).

# In[ ]:


def recommend_nonobscure(model, user_id, n=5, min_ratings=1000):
    # Add predicted_rating column if we haven't already done so.
    if 'predicted_rating' not in movies.columns:
        all_movie_ids = df.movieId.unique()
        preds = model.predict([
            np.repeat(uid, len(all_movie_ids)),
            all_movie_ids,
        ])
        # Add back the offset calculated earlier, to 'uncenter' the ratings, and get back to a [0.5, 5] scale.
        movies.loc[all_movie_ids, 'predicted_rating'] = preds + offset

    nonobscure_movie_ids = movies.index[movies.n_ratings >= min_ratings]
    return movies.loc[nonobscure_movie_ids].sort_values(by='predicted_rating', ascending=False).head(n)


# In[ ]:


#part4.hint()


# In[ ]:


#part4.solution()


# Run the cell below to take a look at our new recommended movies. Did this fix our problem? Do we get better results with a different threshold?

# In[ ]:


recommend_nonobscure(model, uid)


# ## Part 5: A whirlwind introduction to L2 regularization
# 
# > If you're already familiar with L2 regularization, feel free to skip this part.
# 
# We train our model by minimizing a loss function. In this case, that's the squared difference between our model's predicted rating and the actual rating. L2 regularization adds another term to our model's loss function - a "weight penalty". Now our model must balance making accurate predictions while keeping embedding weights not too big.
# 
# We call this a form of regularization, meaning it's expected to reduce overfitting to the training set. How? And what does this have to do with our obscure recommendation problem?
# 
# Even if a movie has only a single rating in the dataset, our model will, in the absence of regularization, try to move its embedding around to match that one rating. However, if the model has a budget for movie weights, it's not very efficient to spend it on improving the accuracy of one rating out of 20,000,000. Popular movies will be worth assigning large weights. Obscure movies should have weights close to 0.
# 
# Does this seem sensible? Test your understanding: What can we say about our model's output/predicted rating for a movie whose embedding vector is all zeros? i.e. `[0, 0, 0, 0, 0, 0, 0, 0]`.

# In[ ]:


part5.solution()


# ## Part 6: Fixing our obscure recommendation problem (regularization)
# 
# The code below is identical to the code used to create the model we've been using in this exercise, except we've added L2 regularization to our embeddings (by specifying a value for the keyword argument `embeddings_regularizer` when creating our Embedding layers).

# In[ ]:


movie_embedding_size = user_embedding_size = 8
user_id_input = keras.Input(shape=(1,), name='user_id')
movie_id_input = keras.Input(shape=(1,), name='movie_id')

movie_r12n = keras.regularizers.l2(1e-6)
user_r12n = keras.regularizers.l2(1e-7)
user_embedded = keras.layers.Embedding(df.userId.max()+1, user_embedding_size,
                                       embeddings_initializer='glorot_uniform',
                                       embeddings_regularizer=user_r12n,
                                       input_length=1, name='user_embedding')(user_id_input)
movie_embedded = keras.layers.Embedding(df.movieId.max()+1, movie_embedding_size, 
                                        embeddings_initializer='glorot_uniform',
                                        embeddings_regularizer=movie_r12n,
                                        input_length=1, name='movie_embedding')(movie_id_input)

dotted = keras.layers.Dot(2)([user_embedded, movie_embedded])
out = keras.layers.Flatten()(dotted)

l2_model = keras.Model(
    inputs = [user_id_input, movie_id_input],
    outputs = out,
)


# Training this model for a decent number of iterations takes around 15 minutes, so to save some time, I have an already trained model you can load from disk by running the cell below.

# In[ ]:


model_dir = '../input/regularized-movielens-factorization-model'
model_fname = 'movie_svd_model_8_r12n.h5'
model_path = os.path.join(model_dir, model_fname)
l2_model = keras.models.load_model(model_path)


# (If you're curious, you can check out the kernel where I trained this model [here](https://www.kaggle.com/colinmorris/regularized-movielens-factorization-model). You may notice that, aside from whether the addition of regularization improves the subjective quality of our recommendations, it already has the benefit of improving our validation error, by reducing overfitting.)
# 
# Try using the code you wrote in part 1 to generate recommendations using this model. How do they compare?

# In[ ]:


# Use the recommend() function you wrote earlier to get the 5 best recommended movies
# for user 26556, and assign them to the variable l2_reccs.
l2_reccs = []
l2_reccs = recommend(l2_model, 26556)


# In[ ]:


#part6.solution()


# What do you think this model's predicted scores will look like for the 'obscure' movies that our earlier model highly recommended? 
# 
# Think about it, then run the cell below to see if you're right.

# In[ ]:


uid = 26556
obscure_reccs = recommend(model, uid)
obscure_mids = obscure_reccs.index
preds = l2_model.predict([
    np.repeat(uid, len(obscure_mids)),
    obscure_mids,
])
recc_df = movies.loc[obscure_mids].copy()
recc_df['l2_predicted_rating'] = preds + offset
recc_df


# 
# ---
# That's the end of this exercise. How'd it go? If you have any questions, be sure to post them on the [forums](https://www.kaggle.com/learn-forum).
# 
# **P.S.** This course is still in beta, so I'd love to get your feedback. If you have a moment to [fill out a super-short survey about this exercise](https://form.jotform.com/82826936984274), I'd greatly appreciate it.
# 
# # Keep going
# 
# When you're ready to continue, [click here](https://www.kaggle.com/colinmorris/exploring-embeddings-with-gensim) to continue on to the next tutorial on exploring embeddings with gensim.
# 

# In[ ]:





# In[ ]:




