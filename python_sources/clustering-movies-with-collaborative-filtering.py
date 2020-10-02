#!/usr/bin/env python
# coding: utf-8

# # Collaborative Filtering Algorithm
# ## This is an implementation of Movie Recommemder System using Collaborative Filtering Algorithm from scratch using only Python. In this kernel, I made an attempt to cluster movies into 2 genres based on ratings from users, using Collaborative Filtering algorithm.
# <p align="center"><img src="https://data.whicdn.com/images/325570784/original.jpg" height="300px" width="400px"></p>
# ### Dataset source: https://www.kaggle.com/grouplens/movielens-20m-dataset

# In[ ]:


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import os
from textwrap import wrap

# Set default fontsize and colors for graphs
SMALL_SIZE, MEDIUM_SIZE, BIG_SIZE = 8, 12, 16
plt.rc('font', size=MEDIUM_SIZE)       
plt.rc('axes', titlesize=BIG_SIZE)     
plt.rc('axes', labelsize=MEDIUM_SIZE)  
plt.rc('xtick', labelsize=MEDIUM_SIZE) 
plt.rc('ytick', labelsize=MEDIUM_SIZE) 
plt.rc('legend', fontsize=SMALL_SIZE)  
plt.rc('figure', titlesize=BIG_SIZE)
my_colors = 'rgbkymc'

# Disable scrolling for long output
from IPython.display import display, Javascript
disable_js = """
IPython.OutputArea.prototype._should_scroll = function(lines) {
    return false;
}
"""
display(Javascript(disable_js))

plt.style.use('ggplot')


# ## (1) Read and Prepare Data

# ### Read "movie" and "rating" dataset

# In[ ]:


# Read the input training data
input_data_file_movie = "../input/movie.csv"
input_data_file_rating = "../input/rating.csv"

movie_data_all = pd.read_csv(input_data_file_movie)
rating_data_all = pd.read_csv(input_data_file_rating)


# In[ ]:


movie_data_all.head(5)


# In[ ]:


rating_data_all.head(5)


# In[ ]:


print("Total number of movies =", movie_data_all.shape[0])
print("Total number of unique movies =", len(movie_data_all.movieId.unique()))
print("")
print("Total number of user ratings =", rating_data_all.shape[0])
print("Total number of unique users =", len(rating_data_all.userId.unique()))


# In[ ]:


# Keep only required columns
movie_data_all = movie_data_all.drop(['genres'], axis=1)
rating_data_all = rating_data_all.drop(['timestamp'], axis=1)


# ### Select few most popular movies, from two distinct genres. In this particular example, we considered movies of genres "Action" and "Romance".
# ### The objective is to find if collborative filtering algorithm can successfully learn the features of these movies based on user ratings, such that we can clearly distinguish their genres and recommend accordingly.

# In[ ]:


# Pick top movies
top_action_movies = ['Dark Knight, The', 'Lord of the Rings: The Return of the King', 
                     'Inception', 'Star Wars: Episode V - The Empire Strikes Back',
                     'Matrix, The']
top_romantic_movies = ['Notting Hill', 'Love Story \(1970\)', 'When Harry Met Sally',
                       'Titanic \(1997\)', 'Pretty Woman']
top_movies = top_action_movies + top_romantic_movies

movie_data = movie_data_all[movie_data_all.title.str.contains('|'.join(top_movies))]
movie_data


# In[ ]:


# Pick all ratings
#num_ratings = 2000000
rating_data = rating_data_all.iloc[:, :]


# ### Merge movie and rating dataset based on movieId column

# In[ ]:


movie_rating_merged_data = movie_data.merge(rating_data, on='movieId', how='inner')
movie_rating_merged_data.head()


# In[ ]:


# Mean rating of a movie
movie_rating_merged_data[movie_rating_merged_data.title == 'Pretty Woman (1990)']['rating'].mean()


# In[ ]:


# Top 10 movies by mean rating
movie_rating_merged_data.groupby(['title'], sort=False)['rating'].mean().sort_values(ascending=False).head(10)


# ## (2) Build Collaborative Filtering Model

# ### Create a pivot table of movies (on rows) and corresponsing user ratings (on columns). The pivot table will contain the ratings of only selected movies.
# ### Thus, rows = movies and columns = users

# In[ ]:


movie_rating_merged_pivot = pd.pivot_table(movie_rating_merged_data,
                                           index=['title'],
                                           columns=['userId'],
                                           values=['rating'],
                                           dropna=False,
                                           fill_value=0
                                          )
movie_rating_merged_pivot.shape


# In[ ]:


Y = movie_rating_merged_pivot


# ### Create a matrix R, such that, R(i,j) = 1 iff User j has selected a rating for Movie i. R(i,j) = 0 otherwise.

# In[ ]:


R = np.ones(Y.shape)
no_rating_idx = np.where(Y == 0.0)
R[no_rating_idx] = 0
R


# ### Assign n_m (number of movies), n_u (number of users) and n_f (number of features)

# In[ ]:


n_u = Y.shape[1]
n_m = Y.shape[0]
n_f = 2  # Because we want to cluster movies into 2 genres


# ### Assign random initial values to movie and user parameters.
# ### X = parameters of movies (each row represent a movie)
# ### Theta = parameters of users (each row represent a user)

# In[ ]:


# Setting random seed to reproduce results later
np.random.seed(7)
Initial_X = np.random.rand(n_m, n_f)
Initial_Theta = np.random.rand(n_u, n_f)
#print("Initial_X =", Initial_X)
#print("Initial_Theta =", Initial_Theta)


# ### Cost function or Objective function of collborative filtering algorithm

# In[ ]:


# Cost Function
def collabFilterCostFunction(X, Theta, Y, R, reg_lambda):
    cost = 0
    error = (np.dot(X, Theta.T) - Y) * R
    error_sq = np.power(error, 2)
    cost = np.sum(np.sum(error_sq)) / 2
    cost = cost + ((reg_lambda/2) * ( np.sum(np.sum((np.power(X, 2)))) + np.sum(np.sum((np.power(Theta, 2))))))
    return cost


# ### Computation of Gradient Descent of collaborative filtering algorithm

# In[ ]:


# Gradient Descent
def collabFilterGradientDescent(X, Theta, Y, R, alpha, reg_lambda, num_iters):
    cost_history = np.zeros([num_iters, 1])
    
    for i in range(num_iters):
        error = (np.dot(X, Theta.T) - Y) * R
        X_grad = np.dot(error, Theta) + reg_lambda * X
        Theta_grad = np.dot(error.T, X) + reg_lambda * Theta
        
        X = X - alpha * X_grad 
        Theta = Theta - alpha * Theta_grad
        
        cost_history[i] = collabFilterCostFunction(X, Theta, Y, R, reg_lambda)
        
    return X, Theta, cost_history


# ## (3) Train the collborative filtering model

# In[ ]:


# Tune hyperparameters
alpha = 0.0001
num_iters = 100000
reg_lambda = 1

# Perform gradient descent to find optimal parameters
X, Theta = Initial_X, Initial_Theta
X, Theta, cost_history = collabFilterGradientDescent(X, Theta, Y, R, alpha, reg_lambda, num_iters)
cost = collabFilterCostFunction(X, Theta, Y, R, reg_lambda)
print("Final cost =", cost)


# ### Plot cost vs number of iterations

# In[ ]:


fig, axes = plt.subplots(figsize=(15,6))
axes.plot(cost_history, 'k--')
axes.set_xlabel('# of iterations')
axes.set_ylabel('Cost')
axes.set_title('Cost / iteration')
plt.show()


# ### Since we have considered only 2 genres (and hence 2 features), we plot the learned feature parameters of movies to visualize the pattern.
# ### We find below that the algorithm has learnt the features pretty well and hence the movies of same genre and clustered together. 
# ### In this particular example, we considered movies of genres "Action" and "Romance". From the visualization, it can be concluded that one axis represents "Degree of Action" and another axis represents "Degree of Romance".
# ### As a next step, we can run K-Means clustering to further verify our understanding.

# In[ ]:


fig, axes = plt.subplots(figsize=(12,12))
axes.scatter(X[:,0], X[:,1], color='red', marker='D')

for val, movie in zip(X, Y.index):
    axes.text(val[0], val[1], movie)

axes.set_xlabel('Feature$_1$ of Movies')
axes.set_ylabel('Feature$_2$ of Movies')
axes.set_title('Movies and its Features')

axes.spines['right'].set_visible(False)
axes.spines['left'].set_visible(True)
axes.spines['top'].set_visible(False)
axes.spines['bottom'].set_visible(True)
axes.set_xlim(0,)
axes.set_ylim(0,)
axes.set_facecolor('linen')

plt.show()


# ### For a random user, what are her preferred movies, and what is our recommendation for her based on result of collaborative filtering algorithm?

# In[ ]:


user_idx = np.random.randint(n_u)
pred_rating = []
print("Original rating of an user:\n", Y.iloc[:,user_idx].sort_values(ascending=False))

predicted_ratings = np.dot(X, Theta.T)
predicted_ratings = sorted(zip(predicted_ratings[:,user_idx], Y.index), reverse=True)
print("\nPredicted rating of the same user:")
_ = [print(rating, movie) for rating, movie in predicted_ratings]

