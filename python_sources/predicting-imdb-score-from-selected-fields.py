#!/usr/bin/env python
# coding: utf-8

# This is an attempt to predict **IMDB score** based on the selected fields:
# 
#  * duration (float64)
#  * director_facebook_likes (float64)
#  * actor_1_facebook_likes (float64)
#  * actor_2_facebook_likes (float64)
#  * actor_3_facebook_likes (float64)
#  * genres (26 multi-hot)
#  * cast_total_facebook_likes (float64)
#  * budget (float64)
#  * title_year (float64)
#  * content_rating (18 one-hot)
#  * language (48 one-hot)
# 
# The **RMSE** for each method (score error range +-):
#  
#  * LinearRegressor (something's wrong, as the error on cross-validation is unreal)
#  * DecisionTreeRegressor -  1.27 (~13%)
#  * RandomForestRegressor - 0.94 (~9%)
#  * NN (101x512x1024x1) - 1.24 (~12%)
#  * NN (101x512x1024x1) - 100k steps - 1.15 (~11%)
#  * NN (101x512x1) - 1.20 (~12%)

# In[ ]:


import pandas as pd

MOVIE_DATASET_PATH = '../input/movie_metadata.csv'

def load_movie_data(path=MOVIE_DATASET_PATH):
    return pd.read_csv(path)

movies = load_movie_data()
movies.head()


# The data has null values, we'll fill those:

# In[ ]:


obj_cols = ['color', 'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name', 'genres', 'movie_title', 'plot_keywords', 
                          'movie_imdb_link', 'country', 'language', 'content_rating']
num_cols = [x for x in list(movies.columns.values) if x not in obj_cols]

movies_num = movies[num_cols]
movies_obj = movies[obj_cols]


# In[ ]:


for col in obj_cols:
    movies[col].fillna('', inplace=True)
    
for col in num_cols:
    median = movies_num[col].median()
    movies[col].fillna(median, inplace=True)


# Let's transform the useful string columns to numerical values.

# In[ ]:


genres = ['History', 'Reality-TV', 'Family', 'Adventure', 'Romance', 'Film-Noir', 'Music', 'War', 'Crime', 'Thriller', 'Drama', 'Sport', 'Game-Show', 'Documentary', 'News', 'Biography', 'Comedy', 'Short', 'Animation', 'Horror', 'Action', 'Fantasy', 'Mystery', 'Sci-Fi', 'Western', 'Musical']
content_ratings = movies['content_rating'].unique().tolist()
languages = movies['language'].unique().tolist()
movies['genres'] = movies['genres'].apply(lambda x: [genres.index(o) for o in x.split('|')])
movies['content_rating'] = movies['content_rating'].apply(lambda x: content_ratings.index(x))
movies['language'] = movies['language'].apply(lambda x: languages.index(x))
movies.head()


# I will extract the data that will be used in training. I will try to predict the IMDB score based on the following inputs:
# 
#  * duration (float64)
#  * director_facebook_likes (float64)
#  * actor_1_facebook_likes (float64)
#  * actor_2_facebook_likes (float64)
#  * actor_3_facebook_likes (float64)
#  * genres (26 multi-hot)
#  * cast_total_facebook_likes (float64)
#  * budget (float64)
#  * title_year (float64)
#  * content_rating (18 one-hot)
#  * language (48 one-hot)

# In[ ]:


num_attribs = ['duration', 'director_facebook_likes', 'actor_1_facebook_likes', 'actor_2_facebook_likes',
               'actor_3_facebook_likes', 'cast_total_facebook_likes', 'budget', 'title_year']
cat_attribs = ['content_rating', 'language']
multi_cat_attribs = ['genres']


# We have 3 non-numerical fields that need to be transformed. For each one-hot encoded I'll create N columns filled with 0 (not matching) and 1 (matching). 

# In[ ]:


data = movies[num_attribs + ['genres']].copy(deep=True)

for x in cat_attribs:
    num = len(movies[x].unique())
    for i in range(num):
        data[x + '_' + str(i)] = (movies[x] == i).astype(int)

def calculate(s):
    row = dict()
    for x in range(len(genres)):
        row['genre_' + str(x)] = 1 if x in s['genres'] else 0
    return pd.Series(row)

data = data.merge(data.apply(calculate, axis=1), left_index=True, right_index=True)
data.drop('genres', axis=1, inplace=True)
    
data.head()


# In[ ]:


scores = movies['imdb_score'].copy(deep=True)


# In[ ]:


# Scaling inputs
for x in data:
    m = data[x].max() * 1.0
    data[x] = data[x].apply(lambda x: x / m)

data.head()


# In[ ]:


# Scaling outputs
max_score = 10.0
scores = scores.apply(lambda x: x / max_score)


# In[ ]:


combined = data.copy(deep=True)
combined['score'] = scores

corr_matrix = combined.corr()
corr_matrix["score"].sort_values(ascending=False)


# Let's try different fitters :)

# In[ ]:


from sklearn.model_selection import cross_val_score
import numpy as np

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[ ]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(data, scores)

lin_scores = cross_val_score(lin_reg, data, scores,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(data, scores)

tree_scores = cross_val_score(tree_reg, data, scores,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores(tree_rmse_scores)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(data, scores)

forest_scores = cross_val_score(forest_reg, data, scores,
                             scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# Let's try some NN with tensorflow

# In[ ]:


import tensorflow as tf
import math


# In[ ]:


npinputs = data.as_matrix()
npoutputs = np.asarray([[x] for x in scores.as_matrix()])

print(npinputs)
print(npoutputs)

split = 4600
train_dataset = npinputs[:split, :]
train_labels = npoutputs[:split, :]
valid_dataset = npinputs[split:, :]
valid_labels = npoutputs[split:, :]


# In[ ]:


BATCH_SIZE = 64

INPUT_SIZE = len(npinputs[0])
OUTPUT_SIZE = 1
HIDDEN_LAYERS = [512, 1024]

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            predictions.shape[0])

def model(inputs):
    # First hidden
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal(
            [INPUT_SIZE, HIDDEN_LAYERS[0]], stddev=1.0 /
            math.sqrt(float(HIDDEN_LAYERS[0]))))
        biases = tf.Variable(tf.zeros([HIDDEN_LAYERS[0]]), dtype=tf.float32)
        hidden = tf.nn.relu(tf.matmul(inputs, weights) + biases)
        
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal(
            [HIDDEN_LAYERS[0], HIDDEN_LAYERS[1]], stddev=1.0 /
            math.sqrt(float(HIDDEN_LAYERS[1]))))
        biases = tf.Variable(tf.zeros([HIDDEN_LAYERS[1]]), dtype=tf.float32)
        hidden = tf.nn.relu(tf.matmul(hidden, weights) + biases)

    with tf.name_scope('output'):
        weights = tf.Variable(tf.truncated_normal(
            [HIDDEN_LAYERS[1], OUTPUT_SIZE], stddev=1.0 /
            math.sqrt(float(HIDDEN_LAYERS[1]))), dtype=tf.float32)
        biases = tf.Variable(tf.zeros([OUTPUT_SIZE]), dtype=tf.float32)
        output = tf.matmul(hidden, weights) + biases

    logits = output

    return logits

def floss(logits, outputs):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.sub(logits, outputs))))


def foptimizer(loss):
    return tf.train.GradientDescentOptimizer(0.001).minimize(loss)


# In[ ]:


graph = tf.Graph()
with graph.as_default():
    # Placeholders
    inputs = tf.placeholder(tf.float32, shape=[
        None, INPUT_SIZE])
    outputs = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])

    logits = model(inputs)

    loss = floss(logits, outputs)
    optimizer = foptimizer(loss)
    preds = tf.nn.softmax(logits)

num_steps = 5001
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
        batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
        feed_dict = {inputs: batch_data, outputs: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, preds], feed_dict=feed_dict)
        if (step % 500 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            feed_dict = {inputs: valid_dataset, outputs: valid_labels}
            _, l, predictions = session.run(
                [optimizer, loss, preds], feed_dict=feed_dict)
            print('Validation loss at step %d: %f' % (step, l))


# In[ ]:


for i, genre in enumerate(genres):
    print (i, genre)

