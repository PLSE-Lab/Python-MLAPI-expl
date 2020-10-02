#!/usr/bin/env python
# coding: utf-8

# # Matrix factorization for recommendation problems
# 
# In the previous lesson, we trained a model to predict the ratings assigned to movies by users in the [MovieLens dataset](https://www.kaggle.com/grouplens/movielens-20m-dataset/home). As a reminder the model looked something like this:
# 
# ![Imgur](https://i.imgur.com/Z1eVQu9.png)
# 
# We look up an embedding vector for the movie and user, concatenate them together. Then we add some hidden layers. Finally these come together at a single output node to predict a rating.
# 
# In this lesson, I'll show a simpler architecture for solving the same problem: **matrix factorization**. And simpler can be a very good thing! Sometimes a simple model will converge quickly to an adequate solution, where a more complicated model might overfit or fail to converge.
# 
# Here's what our matrix factorization model will look like:
# 
# ![Imgur](https://i.imgur.com/lUzvCHj.png)

# # Dot Products
# 
# Let's review a bit of math. If you're a linear algebra pro, feel free to skip this section.
# 
# The dot product of two length-$n$ vectors $\mathbf{a}$ and $\mathbf{b}$ is defined as:
# 
# $$\mathbf{a}\cdot\mathbf{b}=\sum_{i=1}^n a_ib_i=a_1b_1+a_2b_2+\cdots+a_nb_n$$
# 
# The result is a single scalar number (not a vector).
# 
# The dot product is only defined for vectors *of the same length*. This means we need to use the same size for movie embeddings and user embeddings.
# 
# As an example, suppose we've trained embeddings of size 4, and the movie *Twister* is represented by the vector:
# 
# $$\mathbf{m_{Twister}} = \begin{bmatrix} 1.0 & -0.5 & 0.3 & -0.1 \end{bmatrix} $$
# 
# And the user Stanley is represented by:
# 
# $$\mathbf{u_{Stanley}} = \begin{bmatrix} -0.2 & 1.5 & -0.1 & 0.9 \end{bmatrix} $$
# 
# What rating do we think Stanley will give to *Twister*? We can calculate our model's output as:
# 
# \begin{align}
# \ \mathbf{m_{Twister}} \cdot \mathbf{u_{Stanley}} &= (1.0 \cdot -0.2) + (-0.5 \cdot 1.5) + (0.3 \cdot -0.1) + (-0.1 \cdot 0.9) \\
# &= -1.07
# \end{align}
# 
# Because we're training on a a centered version of the rating column, our model's output is on a scale where 0 = the overall average rating in the training set (about 3.5). So we predict that Stanley will give *Twister* $3.5 + (-1.07) = 2.43$ stars.

# # Why?
# 
# There's an intuitive interpretation that supports the decision to combine our embedding vectors in this way. Suppose the dimensions of our movie embedding space correspond to the following axes of variation:
# 
# - Dimension 1: How action-packed?
# - Dimension 2: How romantic?
# - Dimension 3: How mature is the intended audience?
# - Dimension 4: How funny is it?
# 
# Hence, *Twister*, an action-packed disaster movie, has a positive value of 1.0 for $m_1$.
# 
# What does this imply about the meaning of our user vectors? Remember that $m_1u_1$ is one of the terms we add up to get our predicted rating. So if $u_1$ is 1.0, it will increase our predicted rating by 1 star (vs. $u_1 = 0$). If $u_1 = .5$, our predicted rating goes up half a star. If $u_1$ is -1, our predicted rating goes down a star.
# 
# In plain terms $u_1$ tells us 'how does this user feel about action?'. Do they love it? Hate it? Or are they indifferent?
# 
# Stanley's vector tells us he's a big fan of romance and comedy, and slightly dislikes action and mature content. What if we give him a movie that's similar to the last one except that it has lots of romance?
# 
# $$\mathbf{m_{Titanic}} = \begin{bmatrix} 1.0 & 1.1 & 0.3 & -0.1 \end{bmatrix} $$
# 
# It's not hard to predict how this affects our rating output. We're giving Stanley more of what he likes, so his predicted rating increases.
# 
# \begin{align}
# \ \mathrm{predicted\_rating(Stanley, Titanic)} &= \mathbf{m_{Titanic}} \cdot \mathbf{u_{Stanley}} + 3.5 \\
# &= (1.0 \cdot -0.2) + (1.1 \cdot 1.5) + (0.3 \cdot -0.1) + (-0.1 \cdot 0.9) + 3.5 \\
# &= 4.83 \text{ stars}
# \end{align}
# 
# > **Aside:** In practice, the meaning of the dimensions of our movie embeddings will not be quite so clear-cut, but it remains true that the meaning of our movie embedding space and user embedding space are fundamentally tied together: $u_i$ will always represent "how much does this user like movies that have the quality represented by $m_i$?". (Hopefully this also gives some more intuition for why the movie embedding space and user embedding space have to be the same size for this technique.)
# 
# 
# # Implementing it

# In[ ]:



# Setup. Import libraries and load dataframes for Movielens data.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
import random

tf.set_random_seed(1); np.random.seed(1); random.seed(1) # Set random seeds for reproducibility

input_dir = '../input/movielens-preprocessing'
ratings_path = os.path.join(input_dir, 'rating.csv')

ratings_df = pd.read_csv(ratings_path, usecols=['userId', 'movieId', 'rating', 'y'])
df = ratings_df

movies_df = pd.read_csv(os.path.join(input_dir, 'movie.csv'), usecols=['movieId', 'title'])


# The code to create this model is similar to the code we wrote in the previous lesson, except I combine the outputs of the user and movie embedding layers using a `Dot` layer (instead of concatenating them, and piling on dense layers).

# In[ ]:


movie_embedding_size = user_embedding_size = 8

# Each instance consists of two inputs: a single user id, and a single movie id
user_id_input = keras.Input(shape=(1,), name='user_id')
movie_id_input = keras.Input(shape=(1,), name='movie_id')
user_embedded = keras.layers.Embedding(df.userId.max()+1, user_embedding_size, 
                                       input_length=1, name='user_embedding')(user_id_input)
movie_embedded = keras.layers.Embedding(df.movieId.max()+1, movie_embedding_size, 
                                        input_length=1, name='movie_embedding')(movie_id_input)

dotted = keras.layers.Dot(2)([user_embedded, movie_embedded])
out = keras.layers.Flatten()(dotted)

model = keras.Model(
    inputs = [user_id_input, movie_id_input],
    outputs = out,
)
model.compile(
    tf.train.AdamOptimizer(0.001),
    loss='MSE',
    metrics=['MAE'],
)
model.summary(line_length=88)


# Let's train it.

# In[ ]:


history = model.fit(
    [df.userId, df.movieId],
    df.y,
    batch_size=5000,
    epochs=20,
    verbose=0,
    validation_split=.05,
);


# In[ ]:



# Save the model to disk. (We'll be reusing it in a later exercise)
model.save('factorization_model.h5')


# Let's compare the error over time for this model to the deep neural net we trained in the previous lesson:

# In[ ]:



# Load up the training stats we saved to disk in the previous tutorial
history_dir = '../input/embedding-layers'
path = os.path.join(history_dir, 'history-1.csv')
hdf = pd.read_csv(path)

fig, ax = plt.subplots(figsize=(15, 8))
c1 = 'blue'
ax.plot(history.epoch, history.history['val_mean_absolute_error'], '--', label='Validation MAE', color=c1)
ax.plot(history.epoch, history.history['mean_absolute_error'], label='Training MAE', color=c1)

c2 = 'orange'
ax.plot(hdf.epoch, hdf.val_mae, '--', label='Validation MAE (DNN)', color=c2)
ax.plot(hdf.epoch, hdf.train_mae, label='Training MAE (DNN)', color=c2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Mean Absolute Error')
ax.set_xlim(left=0)
baseline_mae = 0.73
ax.axhline(baseline_mae, ls='-.', label='Baseline', color='#002255', alpha=.5)
ax.grid()
fig.legend();


# Our new, simpler model (in blue) is looking pretty good.
# 
# However, even though our embeddings are fairly small, both models suffer from some obvious overfitting. That is,  the error on the training set - the solid lines - is significantly better than on the unseen data. We'll work on addressing that very soon in the exercise.

# # Your turn!
# 
# Head over to [the Exercises notebook](https://www.kaggle.com/kernels/fork/1598589) to get some hands-on practice working with matrix factorization.
# ### P.S...
# 
# This course is still in beta, so I'd love to get your feedback. If you have a moment to [fill out a super-short survey about this lesson](https://form.jotform.com/82826168584267), I'd greatly appreciate it. You can also leave public feedback in the comments below, or on the [Learn Forum](https://www.kaggle.com/learn-forum).
# 
