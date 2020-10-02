#!/usr/bin/env python
# coding: utf-8

# # Movielens-1m analysis using scikit-Surprise

# In[ ]:


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import SVD, SVDpp, NMF
from surprise import SlopeOne, CoClustering
from surprise import accuracy
from surprise.model_selection import train_test_split
import os


# 
# ## Analyzing the dataset
# Attention: put the engine='python' inside the read.csv because it do not suport regex and data set separate content by '::'
# 
# Read the number of columns and rows inthe dataset.

# In[ ]:


reviews = pd.read_csv('../input/ml-1m/ml-1m/ratings.dat', names=['userID', 'movieID', 'rating', 'time'], delimiter='::', engine= 'python')
print('Rows:', reviews.shape[0], '; Columns:', reviews.shape[1], '\n')

reviews.head()


# More informartion about dataset

# In[ ]:


print('No. of Unique Users    :', reviews.userID.nunique())
print('No. of Unique Movies :', reviews.movieID.nunique())
print('No. of Unique Ratings  :', reviews.rating.nunique())


# In[ ]:


rts_gp = reviews.groupby(by=['rating']).agg({'userID': 'count'}).reset_index()
rts_gp.columns = ['Rating', 'Count']


# In[ ]:


plt.barh(rts_gp.Rating, rts_gp.Count, color='royalblue')
plt.title('Overall Count of Ratings', fontsize=15)
plt.xlabel('Count', fontsize=15)
plt.ylabel('Rating', fontsize=15)
plt.grid(ls='dotted')
plt.show()


# ## Some algorith with surprise

# Load again the movielens-100k dataset, but using the surprise class. UserID::MovieID::Rating::Timestamp
# 

# In[ ]:


file_path = os.path.expanduser('../input/ml-1m/ml-1m/ratings.dat')
reader = Reader(line_format='user item rating timestamp', sep='::')
data = Dataset.load_from_file(file_path, reader=reader)

trainset, testset = train_test_split(data, test_size=.15)


# ### k-NN inspired algorithms - KNN Basic

# Algorithm configuration: K = neighbors number. 
# 
# Name = Similarity measure. 
# 
# User based = using item or user.
# 
# Algorithm KNNBasic with 50 neighbors
# 
# Similarity algorithm: pearson
# 
# 

# In[ ]:


algoritmo = KNNBasic(k=50, sim_options={'name': 'pearson', 'user_based': True, 'verbose' : True})


# Use the trainset to trains the algorithm

# In[ ]:


algoritmo.fit(trainset)


# ### We will hide a rating for a user and ask algorithm to predict the rating. 
# 
# User id selected: 49. He is: 18 - 24 year old. He is programmer and live in Huston, Texas

# In[ ]:


uid = str(49)  


# Movie: Negotiator, The (1998)::Action|Thriller. 
# Real rating: 4
# Rating range: 1-5
# 

# In[ ]:


iid = str(2058)  # raw item id


# Get the prediction for the specific users and movie.

# In[ ]:


print("Prediction for rating: ")
pred = algoritmo.predict(uid, iid, r_ui=4, verbose=True)


# Now run the trained model against the testset

# In[ ]:


test_pred = algoritmo.test(testset)


# Analisys of Root-mean-square deviation

# In[ ]:


print("Deviation RMSE: ")
accuracy.rmse(test_pred, verbose=True)


# Analisys of Mean absolute error 

# In[ ]:


# Avalia MAE
print("Analisys MAE: ")
accuracy.mae(test_pred, verbose=True)


# ## k-NN inspired algorithms - KNN With Means

# In[ ]:


# KNNWithMeans with 50 neighbors, user based
algoritmo = KNNWithMeans(k=50, sim_options={'name': 'cosine', 'user_based': False, 'verbose' : True})

algoritmo.fit(trainset)

# Hide the real rating and try to predict
# real rating is 4
# Select User and Movie
uid = str(49)
iid = str(2058)

# Predict the rating
print("\nMaking prediction")
pred = algoritmo.predict(uid, iid, r_ui=4, verbose=True)

test_pred = algoritmo.test(testset)

# Deviation RMSE
print("\nDeviation RMSE: ")
accuracy.rmse(test_pred, verbose=True)

# Analisys MAE
print("\nAnalisys MAE: ")
accuracy.mae(test_pred, verbose=True)

