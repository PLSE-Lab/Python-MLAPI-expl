#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

import surprise  #Scikit-Learn library for recommender systems. 


# Creating an Explicit Latent Matrix Factorization Recommender System for Books10k Dataset.
# We're basing this off of this paper: https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf 

# With Surprise, Data Wrangling is almost automated. We'll load the dataset into Pandas, then leverage the surprise package 

# Probabilistic Matrix Factorization is a prediction method where you estimate the best factorization for the Ratings Matrix. Taking an example, lets say there's a rating matrix of 100k users and 10k items. We'll try to factor this dataset into its latent factors. Think of PCA or Singular Value Decompisition. That would be great to perform, but we have a huge issue. Our dataset is FILLED with NANs. This means that SVD and PCA is undefined, and so instead, we'll have to factor this differently. Primarily, we'll initialize two random matrices, one which estimates the altent factors for the user, and one which estimates the latent factors for the items. 
# 
# In our example, we initialize a 100k by 40 matrix, to get 40 latent factors out of the 100k users we have, and a 40 by 10k matrix, which estimates the latent factors of our matrix. 

# In[2]:


raw=pd.read_csv('../input/ratings.csv')
raw.drop_duplicates(inplace=True)
print('we have',raw.shape[0], 'ratings')
print('the number of unique users we have is:', len(raw.user_id.unique()))
print('the number of unique books we have is:', len(raw.book_id.unique()))
print("The median user rated %d books."%raw.user_id.value_counts().median())
print('The max rating is: %d'%raw.rating.max(),"the min rating is: %d"%raw.rating.min())
raw.head()


# Loading dataset into Surprise:
# For a dataset to be loaded into surprise from a pandas dataframe, you have to specify a reader object and ensure that the columns the dataset are in the order: user,item, rating. 

# In[3]:


#swapping columns
raw=raw[['user_id','book_id','rating']] 
# when importing from a DF, you only need to specify the scale of the ratings.
reader = surprise.Reader(rating_scale=(1,5)) 
#into surprise:
data = surprise.Dataset.load_from_df(raw,reader)


# Surprise lets you implement your own algorithm while still being able to use its very powerful tools for measuring error and choosing hyperparameters. 
# To do so, we have to make a class inheriting from the AlgoBase class. We'll define it along with a fit and estimate method.
# The fit method uses Stochastic Gradient Descent to be able to estimate the best probabilistic matrix factors.

# In[4]:


class ProbabilisticMatrixFactorization(surprise.AlgoBase):
# Randomly initializes two Matrices, Stochastic Gradient Descent to be able to optimize the best factorization for ratings.
    def __init__(self,learning_rate,num_epochs,num_factors):
       # super(surprise.AlgoBase)
        self.alpha = learning_rate #learning rate for Stochastic Gradient Descent
        self.num_epochs = num_epochs
        self.num_factors = num_factors
    def fit(self,train):
        #randomly initialize user/item factors from a Gaussian
        P = np.random.normal(0,.1,(train.n_users,self.num_factors))
        Q = np.random.normal(0,.1,(train.n_items,self.num_factors))
        #print('fit')

        for epoch in range(self.num_epochs):
            for u,i,r_ui in train.all_ratings():
                residual = r_ui - np.dot(P[u],Q[i])
                temp = P[u,:] # we want to update them at the same time, so we make a temporary variable. 
                P[u,:] +=  self.alpha * residual * Q[i]
                Q[i,:] +=  self.alpha * residual * temp 

                
        self.P = P
        self.Q = Q

        self.trainset = train
    
    
    def estimate(self,u,i):
        #returns estimated rating for user u and item i. Prerequisite: Algorithm must be fit to training set.
        #check to see if u and i are in the train set:
        #print('gahh')

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            #print(u,i, '\n','yep:', self.P[u],self.Q[i])
            #return scalar product of P[u] and Q[i]
            nanCheck = np.dot(self.P[u],self.Q[i])
            
            if np.isnan(nanCheck):
                return self.trainset.global_mean
            else:
                return np.dot(self.P[u,:],self.Q[i,:])
        else:# if its not known we'll return the general average. 
           # print('global mean')
            return self.trainset.global_mean
                
        


# Now, lets train the data on the whole dataset, just to gauge the effectiveness of the model we just made.

# In[5]:


Alg1 = ProbabilisticMatrixFactorization(learning_rate=0.05,num_epochs=4,num_factors=10)
data1 = data.build_full_trainset()
Alg1.fit(data1)
print(raw.user_id.iloc[4],raw.book_id.iloc[4])
Alg1.estimate(raw.user_id.iloc[4],raw.book_id.iloc[4])


# Another cool thing about Suprise is it allows you to use their built in GridSearch, which is inspired by sk-learn's GridSearchCV. Its a handy tool that allows us to put in a whole range of hyper-parameters, and picks the best one. 
# 
# Lets use it!

# In[7]:


gs = surprise.model_selection.GridSearchCV(ProbabilisticMatrixFactorization, param_grid={'learning_rate':[0.005,0.01],
                                                                            'num_epochs':[5,10],
                                                                            'num_factors':[10,20]},measures=['rmse', 'mae'], cv=2)
gs.fit(data)


# In[8]:


print('rsme: ',gs.best_score['rmse'],'mae: ',gs.best_score['mae'])
best_params = gs.best_params['rmse']
print('rsme: ',gs.best_params['rmse'],'mae: ',gs.best_params['mae'])


# In[9]:



bestVersion = ProbabilisticMatrixFactorization(learning_rate=best_params['learning_rate'],num_epochs=best_params['num_epochs'],num_factors=best_params['num_factors'])


# In[10]:


#we can use k-fold cross validation to evaluate the best model. 
kSplit = surprise.model_selection.KFold(n_splits=10,shuffle=True)
for train,test in kSplit.split(data):
    bestVersion.fit(train)
    prediction = bestVersion.test(test)
    surprise.accuracy.rmse(prediction,verbose=True)


# Which performed even better than in Grid Search Above, with even 0.2 less Root Means Squared error. Probabalistic Matrix Factorization is powerful tool to use, and Surprise allows us to alleviate some of the pain of data wrangling. 

# 
