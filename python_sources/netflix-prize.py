#!/usr/bin/env python
# coding: utf-8

# <img src='https://miro.medium.com/max/1400/1*00tVH8JxG3NcaKLJW0DoHA.png'>

# In[ ]:


# this is just to know how much time will it take to run this entire ipython notebook 
from datetime import datetime
# globalstart = datetime.now()
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('nbagg')

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import seaborn as sns
sns.set_style('whitegrid')
import os
from scipy import sparse
from scipy.sparse import csr_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import random


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <h3> 3.3.6 Creating sparse matrix from data frame </h3>

# <h4> 3.3.6.1 Creating sparse matrix from train data frame </h4>

# In[ ]:


start = datetime.now()
# if os.path.isfile('kaggle/input/train_sparse_matrix.npz'):
print("It is present in your pwd, getting it from disk....")
# just get it from the disk instead of computing it
train_sparse_matrix = sparse.load_npz('/kaggle/input/netflix2movie/train_sparse_matrix.npz')
print("DONE..")
# else: 
#     print("We are creating sparse_matrix from the dataframe..")
#     # create sparse_matrix and store it for after usage.
#     # csr_matrix(data_values, (row_index, col_index), shape_of_matrix)
#     # It should be in such a way that, MATRIX[row, col] = data
#     train_sparse_matrix = sparse.csr_matrix((train_df.rating.values, (train_df.user.values,
#                                                train_df.movie.values)),)
    
#     print('Done. It\'s shape is : (user, movie) : ',train_sparse_matrix.shape)
#     print('Saving it into disk for furthur usage..')
#     # save it into disk
#     sparse.save_npz("train_sparse_matrix.npz", train_sparse_matrix)
#     print('Done..\n')

print(datetime.now() - start)


# <p><b>The Sparsity of Train Sparse Matrix</b></p>

# In[ ]:


us,mv = train_sparse_matrix.shape
elem = train_sparse_matrix.count_nonzero()

print("Sparsity Of Train matrix : {} % ".format(  (1-(elem/(us*mv))) * 100) )


# <h4> 3.3.6.2 Creating sparse matrix from test data frame </h4>

# In[ ]:


start = datetime.now()
# if os.path.isfile('test_sparse_matrix.npz'):
print("It is present in your pwd, getting it from disk....")
# just get it from the disk instead of computing it
test_sparse_matrix = sparse.load_npz('/kaggle/input/netflix2movie/test_sparse_matrix.npz')
print("DONE..")
# else: 
#     print("We are creating sparse_matrix from the dataframe..")
#     # create sparse_matrix and store it for after usage.
#     # csr_matrix(data_values, (row_index, col_index), shape_of_matrix)
#     # It should be in such a way that, MATRIX[row, col] = data
#     test_sparse_matrix = sparse.csr_matrix((test_df.rating.values, (test_df.user.values,
#                                                test_df.movie.values)))
    
#     print('Done. It\'s shape is : (user, movie) : ',test_sparse_matrix.shape)
#     print('Saving it into disk for furthur usage..')
#     # save it into disk
#     sparse.save_npz("test_sparse_matrix.npz", test_sparse_matrix)
#     print('Done..\n')
    
print(datetime.now() - start)


# <p><b>The Sparsity of Test data Matrix</b></p>

# In[ ]:


us,mv = test_sparse_matrix.shape
elem = test_sparse_matrix.count_nonzero()

print("Sparsity Of Test matrix : {} % ".format(  (1-(elem/(us*mv))) * 100) )


# <h3>3.3.7 Finding Global average of all movie ratings, Average rating per user, and Average rating per movie</h3>

# In[ ]:


# get the user averages in dictionary (key: user_id/movie_id, value: avg rating)

def get_average_ratings(sparse_matrix, of_users):
    
    # average ratings of user/axes
    ax = 1 if of_users else 0 # 1 - User axes,0 - Movie axes

    # ".A1" is for converting Column_Matrix to 1-D numpy array 
    sum_of_ratings = sparse_matrix.sum(axis=ax).A1
    # Boolean matrix of ratings ( whether a user rated that movie or not)
    is_rated = sparse_matrix!=0
    # no of ratings that each user OR movie..
    no_of_ratings = is_rated.sum(axis=ax).A1
    
    # max_user  and max_movie ids in sparse matrix 
    u,m = sparse_matrix.shape
    # creae a dictonary of users and their average ratigns..
    average_ratings = { i : sum_of_ratings[i]/no_of_ratings[i]
                                 for i in range(u if of_users else m) 
                                    if no_of_ratings[i] !=0}

    # return that dictionary of average ratings
    return average_ratings


# <h4> 3.3.7.1 finding global average of all movie ratings </h4>

# In[ ]:


train_averages = dict()
# get the global average of ratings in our train set.
train_global_average = train_sparse_matrix.sum()/train_sparse_matrix.count_nonzero()
train_averages['global'] = train_global_average
train_averages


# <h4> 3.3.7.2 finding average rating per user</h4>

# In[ ]:


train_averages['user'] = get_average_ratings(train_sparse_matrix, of_users=True)
print('\nAverage rating of user 10 :',train_averages['user'][10])


# <h4> 3.3.7.3 finding average rating per movie</h4>

# In[ ]:


train_averages['movie'] =  get_average_ratings(train_sparse_matrix, of_users=False)
print('\n AVerage rating of movie 15 :',train_averages['movie'][15])


#  

# <h4> 3.3.7.4 PDF's & CDF's of Avg.Ratings of Users & Movies (In Train Data)</h4>

# In[ ]:


# start = datetime.now()
# # draw pdfs for average rating per user and average
# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=plt.figaspect(.5))
# fig.suptitle('Avg Ratings per User and per Movie', fontsize=15)

# ax1.set_title('Users-Avg-Ratings')
# # get the list of average user ratings from the averages dictionary..
# user_averages = [rat for rat in train_averages['user'].values()]
# sns.distplot(user_averages, ax=ax1, hist=False, 
#              kde_kws=dict(cumulative=True), label='Cdf')
# sns.distplot(user_averages, ax=ax1, hist=False,label='Pdf')

# ax2.set_title('Movies-Avg-Rating')
# # get the list of movie_average_ratings from the dictionary..
# movie_averages = [rat for rat in train_averages['movie'].values()]
# sns.distplot(movie_averages, ax=ax2, hist=False, 
#              kde_kws=dict(cumulative=True), label='Cdf')
# sns.distplot(movie_averages, ax=ax2, hist=False, label='Pdf')

# plt.show()
# print(datetime.now() - start)


# <h3> 3.3.8 Cold Start problem </h3>

# > We might have to handle __346 movies__ (small comparatively) in test data

#  

# <h3> 3.4.2 Computing Movie-Movie Similarity matrix </h3>

# In[ ]:


start = datetime.now()
if not os.path.isfile('/kaggle/input/netflix2movie/m_m_sim_sparse.npz'):
    print("It seems you don't have that file. Computing movie_movie similarity...")
    start = datetime.now()
    m_m_sim_sparse = cosine_similarity(X=train_sparse_matrix.T, dense_output=False)
    print("Done..")
    # store this sparse matrix in disk before using it. For future purposes.
    print("Saving it to disk without the need of re-computing it again.. ")
#     sparse.save_npz("/kaggle/input/m_m_sim_sparse.npz", m_m_sim_sparse)
    print("Done..")
else:
    print("It is there, We will get it.")
    m_m_sim_sparse = sparse.load_npz("/kaggle/input/netflix2movie/m_m_sim_sparse.npz")
    print("Done ...")

print("It's a ",m_m_sim_sparse.shape," dimensional matrix")

print(datetime.now() - start)


# In[ ]:


m_m_sim_sparse.shape


# - Even though we have similarity measure of each movie, with all other movies, We generally don't care much about least similar movies.
# 
# 
# - Most of the times, only top_xxx similar items matters. It may be 10 or 100.
# 
# 
# - We take only those top similar movie ratings and store them  in a saperate dictionary.

# In[ ]:


movie_ids = np.unique(m_m_sim_sparse.nonzero()[1])


# In[ ]:


# start = datetime.now()
# similar_movies = dict()
# for movie in movie_ids:
#     # get the top similar movies and store them in the dictionary
#     sim_movies = m_m_sim_sparse[movie].toarray().ravel().argsort()[::-1][1:]
#     similar_movies[movie] = sim_movies[:100]
# print(datetime.now() - start)

# # just testing similar movies for movie_15
# similar_movies[15]


#  

# <h3> 3.4.3 Finding most similar movies using similarity matrix </h3>

# __ Does Similarity really works as the way we expected...? __ <br>
# _Let's pick some random movie and check for its similar movies...._

# In[ ]:


# First Let's load the movie details into soe dataframe..
# movie details are in 'netflix/movie_titles.csv'

movie_titles = pd.read_csv("/kaggle/input/netflix3movie/movie_titles.csv", sep=',', header = None,
                           names=['movie_id', 'year_of_release', 'title'], verbose=True,
                      index_col = 'movie_id', encoding = "ISO-8859-1")

movie_titles.head()


#  <h1> 4.  Machine Learning Models </h1>

# <img src='images/models.jpg' width=500px>

# In[ ]:


def get_sample_sparse_matrix(sparse_matrix, no_users, no_movies, path, verbose = True):
    """
        It will get it from the ''path'' if it is present  or It will create 
        and store the sampled sparse matrix in the path specified.
    """

    # get (row, col) and (rating) tuple from sparse_matrix...
    row_ind, col_ind, ratings = sparse.find(sparse_matrix)
    users = np.unique(row_ind)
    movies = np.unique(col_ind)

    print("Original Matrix : (users, movies) -- ({} {})".format(len(users), len(movies)))
    print("Original Matrix : Ratings -- {}\n".format(len(ratings)))

    # It just to make sure to get same sample everytime we run this program..
    # and pick without replacement....
    np.random.seed(15)
    sample_users = np.random.choice(users, no_users, replace=False)
    sample_movies = np.random.choice(movies, no_movies, replace=False)
    # get the boolean mask or these sampled_items in originl row/col_inds..
    mask = np.logical_and( np.isin(row_ind, sample_users),
                      np.isin(col_ind, sample_movies) )
    
    sample_sparse_matrix = sparse.csr_matrix((ratings[mask], (row_ind[mask], col_ind[mask])),
                                             shape=(max(sample_users)+1, max(sample_movies)+1))

    if verbose:
        print("Sampled Matrix : (users, movies) -- ({} {})".format(len(sample_users), len(sample_movies)))
        print("Sampled Matrix : Ratings --", format(ratings[mask].shape[0]))

    print('Saving it into disk for furthur usage..')
    # save it into disk
    sparse.save_npz(path, sample_sparse_matrix)
    if verbose:
            print('Done..\n')
    
    return sample_sparse_matrix


# <h2> 4.1 Sampling Data </h2>

# <h3>4.1.1 Build sample train data from the train data</h3>

# In[ ]:


start = datetime.now()
path = "/kaggle/input/netflix2movie/final_train_sparse_matrix.npz"
if os.path.isfile(path):
    print("It is present in your pwd, getting it from disk....")
    # just get it from the disk instead of computing it
    sample_train_sparse_matrix = sparse.load_npz(path)
    print("DONE..")
else: 
    # get 10k users and 1k movies from available data 
    sample_train_sparse_matrix = get_sample_sparse_matrix(train_sparse_matrix, no_users=22500, no_movies=2700,
                                             path = path)

print(datetime.now() - start)


# <h3>4.1.2 Build sample test data from the test data</h3>

# In[ ]:


start = datetime.now()

path = "/kaggle/input/netflix2movie/final_test_sparse_matrix.npz"
if os.path.isfile(path):
    print("It is present in your pwd, getting it from disk....")
    # just get it from the disk instead of computing it
    sample_test_sparse_matrix = sparse.load_npz(path)
    print("DONE..")
else:
    pass
    # get 5k users and 500 movies from available data 
#     sample_test_sparse_matrix = get_sample_sparse_matrix(test_sparse_matrix, no_users=5000, no_movies=500,
#                                                  path = "sample/small/sample_test_sparse_matrix.npz")

print(datetime.now() - start)


#  

# <h2>4.2 Finding Global Average of all movie ratings, Average rating per User, and Average rating per Movie (from sampled train)</h2>

# In[ ]:


sample_train_averages = dict()


# <h3>4.2.1 Finding Global Average of all movie ratings</h3>

# In[ ]:


# get the global average of ratings in our train set.
global_average = sample_train_sparse_matrix.sum()/sample_train_sparse_matrix.count_nonzero()
sample_train_averages['global'] = global_average
sample_train_averages


# <h3>4.2.2 Finding Average rating per User</h3>

# In[ ]:


sample_train_averages['user'] = get_average_ratings(sample_train_sparse_matrix, of_users=True)
print('\nAverage rating of user 1515220 :',sample_train_averages['user'][1515220])


# <h3>4.2.3 Finding Average rating per Movie</h3>

# In[ ]:


sample_train_averages['movie'] =  get_average_ratings(sample_train_sparse_matrix, of_users=False)
print('\n AVerage rating of movie 15153 :',sample_train_averages['movie'][15153])


#  

# <h2> 4.3 Featurizing data </h2>

# In[ ]:


print('\n No of ratings in Our Sampled train matrix is : {}\n'.format(sample_train_sparse_matrix.count_nonzero()))
print('\n No of ratings in Our Sampled test  matrix is : {}\n'.format(sample_test_sparse_matrix.count_nonzero()))


# <h3> 4.3.1 Featurizing data for regression problem </h3>

# <h4> 4.3.1.1 Featurizing train data </h4>

# In[ ]:


# get users, movies and ratings from our samples train sparse matrix
sample_train_users, sample_train_movies, sample_train_ratings = sparse.find(sample_train_sparse_matrix)


# In[ ]:


############################################################
# It took me almost 10 hours to prepare this train dataset.#
############################################################
start = datetime.now()
if os.path.isfile('/kaggle/input/blabla/reg_train.csv'):
    print("File already exists you don't have to prepare again..." )
else:
    print('preparing {} tuples for the dataset..\n'.format(len(sample_train_ratings)))
    final_list=[]
#     with open('/kaggle/input/netflix3movie/reg_train.csv', mode='w') as reg_data_file:
    count = 0
    for (user, movie, rating)  in zip(sample_train_users, sample_train_movies, sample_train_ratings):
        st = datetime.now()
    #     print(user, movie)    
        #--------------------- Ratings of "movie" by similar users of "user" ---------------------
        # compute the similar Users of the "user"        
        user_sim = cosine_similarity(sample_train_sparse_matrix[user], sample_train_sparse_matrix).ravel()
        top_sim_users = user_sim.argsort()[::-1][1:] # we are ignoring 'The User' from its similar users.
        # get the ratings of most similar users for this movie
        top_ratings = sample_train_sparse_matrix[top_sim_users, movie].toarray().ravel()
        # we will make it's length "5" by adding movie averages to .
        top_sim_users_ratings = list(top_ratings[top_ratings != 0][:5])
        top_sim_users_ratings.extend([sample_train_averages['movie'][movie]]*(5 - len(top_sim_users_ratings)))
    #     print(top_sim_users_ratings, end=" ")    


        #--------------------- Ratings by "user"  to similar movies of "movie" ---------------------
        # compute the similar movies of the "movie"        
        movie_sim = cosine_similarity(sample_train_sparse_matrix[:,movie].T, sample_train_sparse_matrix.T).ravel()
        top_sim_movies = movie_sim.argsort()[::-1][1:] # we are ignoring 'The User' from its similar users.
        # get the ratings of most similar movie rated by this user..
        top_ratings = sample_train_sparse_matrix[user, top_sim_movies].toarray().ravel()
        # we will make it's length "5" by adding user averages to.
        top_sim_movies_ratings = list(top_ratings[top_ratings != 0][:5])
        top_sim_movies_ratings.extend([sample_train_averages['user'][user]]*(5-len(top_sim_movies_ratings))) 
    #     print(top_sim_movies_ratings, end=" : -- ")

        #-----------------prepare the row to be stores in a file-----------------#
        row = list()
        row.append(user)
        row.append(movie)
        # Now add the other features to this data...
        row.append(sample_train_averages['global']) # first feature
        # next 5 features are similar_users "movie" ratings
        row.extend(top_sim_users_ratings)
        # next 5 features are "user" ratings for similar_movies
        row.extend(top_sim_movies_ratings)
        # Avg_user rating
        row.append(sample_train_averages['user'][user])
        # Avg_movie rating
        row.append(sample_train_averages['movie'][movie])

        # finalley, The actual Rating of this user-movie pair...
        row.append(rating)
        count = count + 1

        # add rows to the file opened..
#         reg_data_file.write(','.join(map(str, row)))
#         reg_data_file.write('\n')    
        final_list.append(row)
        if (count)%1000 == 0:
            # print(','.join(map(str, row)))
            print("Done for {} rows----- {}".format(count, datetime.now() - start))


print(datetime.now() - start)


# In[ ]:


pd.DataFrame(final_list).to_csv('final_reg_train_aseem', index=False, header=False)


# __Reading from the file to make a Train_dataframe__

# In[ ]:


# reg_train = pd.read_csv('sample/small/reg_train.csv', names = ['user', 'movie', 'GAvg', 'sur1', 'sur2', 'sur3', 'sur4', 'sur5','smr1', 'smr2', 'smr3', 'smr4', 'smr5', 'UAvg', 'MAvg', 'rating'], header=None)
# reg_train.head()


# -----------------------
# 
# - __GAvg__ : Average rating of all the ratings 
# 
# 
# - __Similar users rating of this movie__:
#     - sur1, sur2, sur3, sur4, sur5 ( top 5 similar users who rated that movie.. )
#     
# 
# 
# - __Similar movies rated by this user__:
#     - smr1, smr2, smr3, smr4, smr5 ( top 5 similar movies rated by this movie.. )
# 
# 
# - __UAvg__ : User's Average rating
# 
# 
# - __MAvg__ : Average rating of this movie
# 
# 
# - __rating__ : Rating of this movie by this user.
# 
# -----------------------

# In[ ]:


# get users, movies and ratings from the Sampled Test 
sample_test_users, sample_test_movies, sample_test_ratings = sparse.find(sample_test_sparse_matrix)


# In[ ]:


start = datetime.now()

if os.path.isfile('/kaggle/input/blabla/reg_test.csv'):
    print("It is already created...")
else:

    print('preparing {} tuples for the dataset..\n'.format(len(sample_test_ratings)))
#     with open('reg_test.csv', mode='w') as reg_data_file:
    count = 0 
    final_list_test=[]
    for (user, movie, rating)  in zip(sample_test_users, sample_test_movies, sample_test_ratings):
        st = datetime.now()

    #--------------------- Ratings of "movie" by similar users of "user" ---------------------
        #print(user, movie)
        try:
            # compute the similar Users of the "user"        
            user_sim = cosine_similarity(sample_train_sparse_matrix[user], sample_train_sparse_matrix).ravel()
            top_sim_users = user_sim.argsort()[::-1][1:] # we are ignoring 'The User' from its similar users.
            # get the ratings of most similar users for this movie
            top_ratings = sample_train_sparse_matrix[top_sim_users, movie].toarray().ravel()
            # we will make it's length "5" by adding movie averages to .
            top_sim_users_ratings = list(top_ratings[top_ratings != 0][:5])
            top_sim_users_ratings.extend([sample_train_averages['movie'][movie]]*(5 - len(top_sim_users_ratings)))
            # print(top_sim_users_ratings, end="--")

        except (IndexError, KeyError):
            # It is a new User or new Movie or there are no ratings for given user for top similar movies...
            ########## Cold STart Problem ##########
            top_sim_users_ratings.extend([sample_train_averages['global']]*(5 - len(top_sim_users_ratings)))
            #print(top_sim_users_ratings)
        except:
            print(user, movie)
            # we just want KeyErrors to be resolved. Not every Exception...
            raise



        #--------------------- Ratings by "user"  to similar movies of "movie" ---------------------
        try:
            # compute the similar movies of the "movie"        
            movie_sim = cosine_similarity(sample_train_sparse_matrix[:,movie].T, sample_train_sparse_matrix.T).ravel()
            top_sim_movies = movie_sim.argsort()[::-1][1:] # we are ignoring 'The User' from its similar users.
            # get the ratings of most similar movie rated by this user..
            top_ratings = sample_train_sparse_matrix[user, top_sim_movies].toarray().ravel()
            # we will make it's length "5" by adding user averages to.
            top_sim_movies_ratings = list(top_ratings[top_ratings != 0][:5])
            top_sim_movies_ratings.extend([sample_train_averages['user'][user]]*(5-len(top_sim_movies_ratings))) 
            #print(top_sim_movies_ratings)
        except (IndexError, KeyError):
            #print(top_sim_movies_ratings, end=" : -- ")
            top_sim_movies_ratings.extend([sample_train_averages['global']]*(5-len(top_sim_movies_ratings)))
            #print(top_sim_movies_ratings)
        except :
            raise

        #-----------------prepare the row to be stores in a file-----------------#
        row = list()
        # add usser and movie name first
        row.append(user)
        row.append(movie)
        row.append(sample_train_averages['global']) # first feature
        #print(row)
        # next 5 features are similar_users "movie" ratings
        row.extend(top_sim_users_ratings)
        #print(row)
        # next 5 features are "user" ratings for similar_movies
        row.extend(top_sim_movies_ratings)
        #print(row)
        # Avg_user rating
        try:
            row.append(sample_train_averages['user'][user])
        except KeyError:
            row.append(sample_train_averages['global'])
        except:
            raise
        #print(row)
        # Avg_movie rating
        try:
            row.append(sample_train_averages['movie'][movie])
        except KeyError:
            row.append(sample_train_averages['global'])
        except:
            raise
        #print(row)
        # finalley, The actual Rating of this user-movie pair...
        row.append(rating)
        #print(row)
        count = count + 1

        # add rows to the file opened..
#         reg_data_file.write(','.join(map(str, row)))
        #print(','.join(map(str, row)))
#         reg_data_file.write('\n') 
        final_list_test.append(row)
        if (count)%100 == 0:
            #print(','.join(map(str, row)))
            print("Done for {} rows----- {}".format(count, datetime.now() - start))
    print("",datetime.now() - start)  


# In[ ]:


pd.DataFrame(final_list).to_csv('final_reg_test_aseem.csv', index=False, header=False)


#  
