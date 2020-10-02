#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# 1. User's Dataset
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('../input/ml-100k/u.user', sep='|', names=u_cols,
                    encoding='latin-1', parse_dates=True) 
# 2. Rating dataset
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('../input/ml-100k/u.data', sep='\t', names=r_cols,
                      encoding='latin-1')

# 3.Movies Dataset
m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url','unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('../input/ml-100k/u.item', sep='|', names=m_cols,
                     encoding='latin-1')


# In[ ]:


#users
print(users.shape)
users.head(4)


# In[ ]:


#ratings
print(ratings.shape)
ratings.head(4)


# In[ ]:


#items
print(movies.shape)
movies.head(4)


# In[ ]:


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_train = pd.read_csv('../input/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('../input/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
ratings_train.shape, ratings_test.shape


# In[ ]:


ratings_train.head(4),ratings_test.head(4)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x=ratings_train.rating,data=ratings_train)


# In[ ]:


no_of_rated_movies_per_user = ratings_train.groupby(by='user_id')['rating'].count().sort_values(ascending=False)
no_of_rated_movies_per_user.head()


# Looking PDF and CDF of number of rated movies per user
# 
# **CDF**: cummulative denisty function. It is the total probability of anything below it. Its range is 0-1. Also known as density function. Represented as F(x).
# 
# **PDF** : Probability denisty function. It is probability at one point. also know as probability mass function(PMF). Represented as f(x).
# ![1](https://qph.ec.quoracdn.net/main-qimg-e50787cd6024e1945ef5632192b70a69)
# 
# So both F(x) and f(x) as inter related to each other.
# if we do the derivative of F(x) we get the f(x) and vice versa if we integrate f(x) we get the F(x).
# 
# **In detail Explaination:**
# Generally Random variables are two types 
# 
# 1.** Continuos**: Which is solved by integral.
# ![image.png](attachment:image.png)  
# 2.** Discrete** : Which is solved by summation.
# 
# So we can say the *PDF* is the continuous function.
# 
# [For more Explanation please refer this vedio](http://youtu.be/DIsZFAV9Hy0)

# In[ ]:


ax1 = plt.subplot(121)
sns.kdeplot(no_of_rated_movies_per_user, shade=True, ax=ax1)
plt.xlabel('No of ratings by user')
plt.title("PDF")

ax2 = plt.subplot(122)
sns.kdeplot(no_of_rated_movies_per_user, shade=True, cumulative=True,ax=ax2)
plt.xlabel('No of ratings by user')
plt.title('CDF')

plt.show()


# In[ ]:


no_of_rated_movies_per_user.describe()


# In[ ]:


quantiles = no_of_rated_movies_per_user.quantile(np.arange(0,1.01,0.01), interpolation='higher')
quantiles


# In[ ]:


plt.title("Quantiles and their Values")
quantiles.plot()
# quantiles with 0.05 difference
plt.scatter(x=quantiles.index[::5], y=quantiles.values[::5], c='orange', label="quantiles with 0.05 intervals")
# quantiles with 0.25 difference
plt.scatter(x=quantiles.index[::25], y=quantiles.values[::25], c='m', label = "quantiles with 0.25 intervals")
plt.ylabel('No of ratings by user')
plt.xlabel('Value at the quantile')
plt.legend(loc='best')

# annotate the 25th, 50th, 75th and 100th percentile values....
for x,y in zip(quantiles.index[::25], quantiles[::25]):
    plt.annotate(s="({} , {})".format(x,y), xy=(x,y), xytext=(x-0.05, y+500)
                ,fontweight='bold')

plt.show()


# In[ ]:


quantiles[::5]


# In[ ]:


print('\n No of ratings at last 5 percentile : {}\n'.format(sum(no_of_rated_movies_per_user>= 301)) )


# In[ ]:


no_of_ratings_per_movie = ratings_train.groupby(by='movie_id')['rating'].count().sort_values(ascending=False)

fig = plt.figure(figsize=plt.figaspect(.5))
ax = plt.gca()
plt.plot(no_of_ratings_per_movie.values)
plt.title('# RATINGS per Movie')
plt.xlabel('Movie')
plt.ylabel('No of Users who rated a movie')
ax.set_xticklabels([])

plt.show()


# In[ ]:


n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]


# In[ ]:


data_matrix = np.zeros((n_users, n_items))
for line in ratings.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]


# In[ ]:


from sklearn.metrics.pairwise import pairwise_distances 
user_similarity = pairwise_distances(data_matrix, metric='cosine')
item_similarity = pairwise_distances(data_matrix.T, metric='cosine')


# In[ ]:


def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #We use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


# In[ ]:


user_prediction = predict(data_matrix, user_similarity, type='user')
item_prediction = predict(data_matrix, item_similarity, type='item')


# In[ ]:


import graphlab
train_data = graphlab.SFrame(ratings_base)
test_data = graphlab.SFrame(ratings_test)


# In[ ]:


popularity_model = graphlab.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')


# In[ ]:


#Get recommendations for first 5 users and print them
#users = range(1,6) specifies user ID of first 5 users
#k=5 specifies top 5 recommendations to be given
popularity_recomm = popularity_model.recommend(users=range(1,6),k=5)
popularity_recomm.print_rows(num_rows=25)


# In[ ]:


ratings_base.groupby(by='movie_id')['rating'].mean().sort_values(ascending=False).head(20)


# In[ ]:


#Train Model
item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='pearson')

#Make Recommendations:
item_sim_recomm = item_sim_model.recommend(users=range(1,6),k=5)
item_sim_recomm.print_rows(num_rows=25)


# In[ ]:


model_performance = graphlab.compare(test_data, [popularity_model, item_sim_model])
graphlab.show_comparison(model_performance,[popularity_model, item_sim_model])


# In[ ]:


import turicreate
train_data = turicreate.SFrame(ratings_train)
test_data = turicreate.Sframe(ratings_test)


# In[ ]:


popularity_model = turicreate.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')


# In[ ]:


popularity_recomm = popularity_model.recommend(users=[1,2,3,4,5],k=5)
popularity_recomm.print_rows(num_rows=25)


# In[ ]:


#Training the model
item_sim_model = turicreate.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='cosine')

#Making recommendations
item_sim_recomm = item_sim_model.recommend(users=[1,2,3,4,5],k=5)
item_sim_recomm.print_rows(num_rows=25)


# In[ ]:




