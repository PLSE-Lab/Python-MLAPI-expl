#!/usr/bin/env python
# coding: utf-8

# Earlier we trained a model to predict the ratings users would give to movies using a network with embeddings learned for each movie and user. Embeddings are powerful! But how do they actually work? 
# 
# Previously, I claimed that embeddings capture the 'meaning' of the objects they represent, and discover useful latent structure. Let's put that to the test!
# 
# # Looking up embeddings
# 
# Let's load a model we trained earlier so we can investigate the embedding weights that it learned.

# In[ ]:


import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

input_dir = '../input/movielens-preprocessing'
model_dir = '../input/movielens-spiffy-model'
model_path = os.path.join(model_dir, 'movie_svd_model_32.h5')
model = keras.models.load_model(model_path)


# The embedding weights are part of the model's internals, so we'll have to do a bit of digging around to access them. We'll grab the layer responsible for embedding movies, and use the `get_weights()` method to get its learned weights.

# In[ ]:


emb_layer = model.get_layer('movie_embedding')
(w,) = emb_layer.get_weights()
w.shape


# Our weight matrix has 26,744 rows for that many movies. Each row is 32 numbers - the size of our movie embeddings.
# 
# Let's look at an example movie vector:

# In[ ]:


w[0]


# What movie is this the embedding of? Let's load up our dataframe of movie metadata.

# In[ ]:


movies_path = os.path.join(input_dir, 'movie.csv')
movies_df = pd.read_csv(movies_path, index_col=0)
movies_df.head()


# Of course, it's *Toy Story*! I should have recognized that vector anywhere.
# 
# Okay, I'm being facetious. It's hard to make anything of these vectors at this point. We never directed the model about how to use any particular embedding dimension. We left it alone to learn whatever representation it found useful.
# 
# So how do we check whether these representations are sane and coherent?
# 
# ## Vector similarity
# 
# A simple way to test this is to look at how close or distant pairs of movies are in the embedding space. Embeddings can be thought of as a smart distance metric. If our embedding matrix is any good, it should map similar movies (like *Toy Story* and *Shrek*) to similar vectors.

# In[ ]:


i_toy_story = 0
i_shrek = movies_df.loc[
    movies_df.title == 'Shrek',
    'movieId'
].iloc[0]

toy_story_vec = w[i_toy_story]
shrek_vec = w[i_shrek]

print(
    toy_story_vec,
    shrek_vec,
    sep='\n',
)


# Comparing dimension-by-dimension, these look vaguely similar. If we wanted to assign a single number to their similarity, we could calculate the euclidean distance between these two vectors. (This is our conventional 'as the crow flies' notion of distance between two points. Easy to grok in 1, 2, or 3 dimensions. Mathematically, we can also extend it to 32 dimensions, though good luck visualizing it.)

# In[ ]:


from scipy.spatial import distance

distance.euclidean(toy_story_vec, shrek_vec)


# How does this compare to a pair of movies that we would think of as very different?

# In[ ]:


i_exorcist = movies_df.loc[
    movies_df.title == 'The Exorcist',
    'movieId'
].iloc[0]

exorcist_vec = w[i_exorcist]

distance.euclidean(toy_story_vec, exorcist_vec)


# As expected, much further apart.
# 
# ## Cosine Distance
# 
# If you check out [the docs for the `scipy.spatial` module](https://docs.scipy.org/doc/scipy-0.14.0/reference/spatial.distance.html), you'll see there are actually a *lot* of different measures of distance that people use for different tasks.
# 
# When judging the similarity of embeddings, it's more common to use [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity).
# 
# In brief, the cosine similarity of two vectors ranges from -1 to 1, and is a function of the *angle* between the vectors. If two vectors point in the same direction, their cosine similarity is 1. If they point in opposite directions, it's -1. If they're orthogonal (i.e. at right angles), their cosine similarity is 0.
# 
# Cosine distance is just defined as 1 minus the cosine similarity (and therefore ranges from 0 to 2).
# 
# Let's calculate a couple cosine distances between movie vectors:

# In[ ]:


print(
    distance.cosine(toy_story_vec, shrek_vec),
    distance.cosine(toy_story_vec, exorcist_vec),
    sep='\n'
)


# > **Aside:** *Why* is cosine distance commonly used when working with embeddings? The short answer, as with so many deep learning techniques, is "empirically, it works well". In the exercise coming up, you'll get to do a little hands-on investigation that digs into this question more deeply.
# 
# Which movies are most similar to *Toy Story*? Which movies fall right between *Psycho* and *Scream* in the embedding space? We could write a bunch of code to work out questions like this, but it'd be pretty tedious. Fortunately, there's already a library for exactly this sort of work: **Gensim**.
# 
# # Exploring embeddings with Gensim
# 
# I'll instantiate an instance of [`WordEmbeddingsKeyedVectors`](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.WordEmbeddingsKeyedVectors) with our model's movie embeddings and the titles of the corresponding movies.
# 
# > Aside: You may notice that Gensim's docs and many of its class and method names refer to *word* embeddings. While the library is most frequently used in the text domain, we can use it to explore embeddings of any sort.

# In[ ]:


from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors

# Limit to movies with at least this many ratings in the dataset
threshold = 100
mainstream_movies = movies_df[movies_df.n_ratings >= threshold].reset_index(drop=True)

movie_embedding_size = w.shape[1]
kv = WordEmbeddingsKeyedVectors(movie_embedding_size)
kv.add(
    mainstream_movies['key'].values,
    w[mainstream_movies.movieId]
)


# Okay, so which movies are most similar to *Toy Story*?

# In[ ]:


kv.most_similar('Toy Story')


# Wow, these are pretty great! It makes perfect sense that *Toy Story 2* is the most similar movie to *Toy Story*. And most of the rest are animated kids movies with a similar computer-animated style.

# So it's learned something about 3-d animated kids flick, but maybe that was just a fluke. Let's look at the closest neighbours for a few more movies from a variety of genres:

# In[ ]:



import textwrap
movies = ['Eyes Wide Shut', 'American Pie', 'Iron Man 3', 'West Side Story',
          'Battleship Potemkin', 'Clueless'
]

def plot_most_similar(movie, ax, topn=5):
    sim = kv.most_similar(movie, topn=topn)[::-1]
    y = np.arange(len(sim))
    w = [t[1] for t in sim]
    ax.barh(y, w)
    left = min(.6, min(w))
    ax.set_xlim(right=1.0, left=left)
    # Split long titles over multiple lines
    labels = [textwrap.fill(t[0] , width=24)
              for t in sim]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title(movie)    

fig, axes = plt.subplots(3, 2, figsize=(15, 9))

for movie, ax in zip(movies, axes.flatten()):
    plot_most_similar(movie, ax)
    
fig.tight_layout()


# Artsy erotic dramas, raunchy sophomoric comedies, old-school musicals, superhero movies... our embeddings manage to nail a wide variety of cinematic niches!

# # Semantic vector math
# 
# The [`most_similar`](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.most_similar) method optionally takes a second argument, `negative`. If we call `kv.most_similar(a, b)`, then instead of finding the vector closest to `a`, it will find the closest vector to `a - b`.
# 
# Why would you want to do that? It turns out that doing addition and subtraction of embedding vectors often gives surprisingly meaningful results. For example, how would you fill in the following equation?
# 
#     Scream = Psycho + ________
# 
# *Scream* and *Psycho* are similar in that they're violent, scary movies somewhere on the border between Horror and Thriller. The biggest difference is that *Scream* has elements of comedy. So I'd say *Scream* is what you'd get if you combined *Psycho* with a comedy.
# 
# But we can actually ask Gensim to fill in the blank for us via vector math (after some rearranging):
# 
#     ________ = Scream - Psycho

# In[ ]:


kv.most_similar(
    positive = ['Scream'],
    negative = ['Psycho (1960)']
)


# If you are familiar with these movies, you'll see that the missing ingredient that takes us from *Psycho* to *Scream* is comedy (and also late-90's-teen-movie-ness).

# ## Analogy solving
# 
# The SAT test which is used to get into American colleges and universities poses analogy questions like:
# 
#     shower : deluge :: _____ : stare
#     
# (Read "shower is to deluge as ___ is to stare")
# 
# To solve this, we find the relationship between deluge and shower, and apply it to stare. A shower is a milder form of a deluge. What's a milder form of stare? A good answer here would be "glance", or "look". 
# 
# It's kind of astounding that this works, but people have found that these can often be effectively solved by simple vector math on word embeddings. Can we solve movie analogies with our embeddings? Let's try. What about:
# 
#     Brave : Cars 2 :: Pocahontas : _____
#     
# The answer is not clear. One interpretation would be that *Brave* is like *Cars 2*, except that the latter is aimed primarily at boys, and the former might be more appealing to girls, given its female protagonist. So maybe the answer should be, like *Pocahontas*, a mid-90's conventional animation kids movie, but more of a 'boy movie'. *Hercules*? *The Lion King*? 
# 
# Let's ask our embeddings what they think.
# 
# In terms of vector math, we can frame this as...
# 
#     Cars 2 = Brave + X
#     _____  = Pocahontas + X
#     
# Rearranging, we get:
# 
#     ____ = Pocahontas + (Cars 2 - Brave)
# 
# We can solve this by passing in two movies (*Pocahontas* and *Cars 2*) for the positive argument to `most_similar`, with *Brave* as the negative argument:

# In[ ]:


kv.most_similar(
    ['Pocahontas', 'Cars 2'],
    negative = ['Brave']
)


# This weakly fits our prediction: the 4 closest movies are indeed kids animated movies from the 90s. After that, the results are a bit more perplexing.
# 
# Is our model wrong, or were we? Another difference we failed to account for between *Cars 2* and *Brave* is that the former is a sequel, and the latter is not. 7/10 of our results are also sequels. This tells us something interesting about our learned embeddings (and, ultimately, about the problem of predicting movie preferences). "Sequelness" is an important property to our model - which suggests that some of the variance in our data is accounted for the fact that some people tend to like sequels more than others.

# # Your turn!
# 
# Head over to [the Exercises notebook](https://www.kaggle.com/kernels/fork/1598893) to get some hands-on practice working with exploring embeddings with gensim.
# ### P.S...
# 
# This course is still in beta, so I'd love to get your feedback. If you have a moment to [fill out a super-short survey about this lesson](https://form.jotform.com/82826473884269), I'd greatly appreciate it. You can also leave public feedback in the comments below, or on the [Learn Forum](https://www.kaggle.com/learn-forum).
# 
