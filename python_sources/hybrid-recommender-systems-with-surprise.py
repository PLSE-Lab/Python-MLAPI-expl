#!/usr/bin/env python
# coding: utf-8

# ## Hybrid Recommender Systems with Suprise: A weighted appproach
# ### We're using Suprise(a sci-kit package for recommender systems) to ensure that the recommender systems that we are using are optimized-- so then we can ensemble them and not worry about flaws in individual implementation. 

# In[104]:


import surprise
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# First, lets read and clean the dataset so we know what we are working with. This I copied from my other kernel regarding Probablistic Matrix Factorization. 

# In[105]:


raw=pd.read_csv('../input/ratings.csv')
raw.drop_duplicates(inplace=True)
print('we have',raw.shape[0], 'ratings')
print('the number of unique users we have is:', len(raw.user_id.unique()))
print('the number of unique books we have is:', len(raw.book_id.unique()))
print("The median user rated %d books."%raw.user_id.value_counts().median())
print('The max rating is: %d'%raw.rating.max(),"the min rating is: %d"%raw.rating.min())
raw.head()


# Whenever you're loading a dataset into Surprise, you can use their dataset Reader class, which alleviates a great deal of pain. You can specify a lot of file formats-- but for pandas dataframes, which we're using, you can specify the ratings and the Dataframe.

# In[106]:


#swapping columns
raw=raw[['user_id','book_id','rating']] 
raw.columns = ['n_users','n_items','rating']

rawTrain,rawholdout = train_test_split(raw, test_size=0.25 )
# when importing from a DF, you only need to specify the scale of the ratings.
reader = surprise.Reader(rating_scale=(1,5)) 
#into surprise:
data = surprise.Dataset.load_from_df(rawTrain,reader)
holdout = surprise.Dataset.load_from_df(rawholdout,reader)


# # In Pseudo Code, our Algorithm is as follows: 
# 1. We split the dataset into 10 folds, where we train on 9 of the folds and test on the remaining one, which randomly alternates.. 
# 2. We run several recommender systems on the dataset, and optimize the recommender systems on the 75% system.
# 3. intialize a weighted variable alpha to be 1/q, where q is the number of recommender systems we use. 
# 4. let the rated matrix equal alpha * sum(predicted Ratings Matrices) and compare that with the real rating. 
# 5. Using Gradient Descent, optimize the alpha term over parameter space to be able to optimize to give the most weight to the model which can represent the best prediction.

# ### First, lets pick some algorithms to include into our ensemble. We'll choose four. 
# 1. Collaborative Filtering
# 2. Matrix Factorization
# 3. collaborative filtering with co-clustering
# 4. Collaborative Filtering based on the popular Slope One Algorithm

# Implementing Collaborative Filtering, Number one on our list:
# Collaborative filtering is a recommender system that recommends based off of similiarity between items. The big idea is that items that are similiar should be similiarly liked by the same user. For example, if you liked Alien, and you really liked Predator, there's a good chance you'll enjoy Alien Versus Predator. We're just doing the same thing with books here. 
# If you'd like to read more, read up here: http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf 

# In[107]:


kSplit = surprise.model_selection.split.KFold(n_splits=10, shuffle=True) # split data into folds. 


# In[108]:


sim_options = sim_options = {'name': 'cosine',
               'user_based': False  # compute  similarities between items
               }
collabKNN = surprise.KNNBasic(k=40,sim_options=sim_options) #try removing sim_options. You'll find memory errors. 
rmseKNN = []
rmseSVD = []
rmseCo = []
rmseSlope = []
for trainset, testset in kSplit.split(data): #iterate through the folds.
    collabKNN.fit(trainset)
    predictionsKNN = collabKNN.test(testset)
    rmseKNN.append(surprise.accuracy.rmse(predictionsKNN,verbose=True))#get root means squared error
    


# ### Beautiful, lets train more. You're welcome to edit this notebook and try different hyperparameters. The main purpose of this notebook is to show you the ensemble methods, but you can use Suprise's Grid Search CV to find the best possible Hyperparameters. 

# ### Second, lets train our Matrix Factorization Algorithm. 
# This algorithm was created by Simon Funk during the Netflix Prize, and it is called FunkSVD. The big idea behind this algorithm is you try to estimate the best latent factors for the ratings. So, if you have a 100k users and 10k books, you factor the 100k x 10k matrix into the number of factors. In turn, you would be making two 100k x 30 and 30 x 10k matrices. You multiply them together to get the predicted rating. This lets us optimize on the latent factors between users, such as users that are similiar together because they all rated action films, and latent factors between items, like book series like Goosebumps and Steven King. We multiply each of these to get the predicted rating. 
# 
# If you'd like to read more, look it up here: https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf 

# In[ ]:


funkSVD = surprise.prediction_algorithms.matrix_factorization.SVD(n_factors=30,n_epochs=10,biased=True)


# In[ ]:


min_error = 1
for trainset, testset in kSplit.split(data): #iterate through the folds.
    funkSVD.fit(trainset)
    predictionsSVD = funkSVD.test(testset)
    rmseSVD.append(surprise.accuracy.rmse(predictionsSVD,verbose=True))#get root means squared error
    
    


# Beautiful, lets train a recommender system using co-clustering collaborative filtering. 
# Co-clustering is where you cluster users and items together, using clustering techniques. You identify three clusters. You'll have to sum three things to get a predicted rating:
#     1. You find the cluster for the specified rating of user u and item i, and identify the mean of that cluster. So you find the mean of cluster u_i.
#     2. find the mean of the cluster of item i and subtract that from the average rating of that item.
#     3. find the mean of cluster of user u and substract that from the average rating of that user. 
#     
# For most of these, you'll find that the RSME remains the same for all of the K-Folds. 
# 
# If you want to learn more about Co-Clustering, read more here: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.113.6458&rep=rep1&type=pdf 

# In[ ]:


coClus = surprise.prediction_algorithms.co_clustering.CoClustering(n_cltr_u=4,n_cltr_i=4,n_epochs=25) 
for trainset, testset in kSplit.split(data): #iterate through the folds.
    coClus.fit(trainset)
    predictionsCoClus = coClus.test(testset)
    rmseCo.append(surprise.accuracy.rmse(predictionsCoClus,verbose=True))#get root means squared error


# Training our last model, we will use the Slope One Collaborative Filtering Algorithm. This algorithm computes the slope of each of the relevant items rated by a user, finds the difference, then computes the prediction. Its a blunt instrument, but its a good heuristic that might improve our ensemble method.  You can read more here: https://arxiv.org/abs/cs/0702144 

# In[ ]:


slopeOne = surprise.prediction_algorithms.slope_one.SlopeOne()


# In[ ]:


for trainset, testset in kSplit.split(data): #iterate through the folds.
    slopeOne.fit(trainset)
    predictionsSlope = slopeOne.test(testset)
    rmseSlope.append(surprise.accuracy.rmse(predictionsSlope,verbose=True))#get root means squared error


# Beautiful, we now have four recommender systems begging to be placed into an Ensemble Method. This is where the fun begins. First, lets plot each one to see how they performed. 
# 
# Then, lets implemented the Ensemble algorithm outlined above.
# 

# In[ ]:


#plotting the prediction data:
import matplotlib.pyplot as plt
for prediction in compiledPredictions:
    modelPrediction = plt.plot(rmseKNN,label='knn')
    modelPrediction = plt.plot(rmseSVD,label='svd')
    modelPrediction = plt.plot(rmseCo,label='cluster')
    modelPrediction = plt.plot(rmseSlope,label='slope')

    modelPrediction = plt.xlabel('folds')
    modelPrediction = plt.ylabel('accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Perfect, It looks like our KNN is outperforming the rest. Lets try to hybridize the models so we can get the best parts of every model. To do this, we're going to use Suprise to make a new algorithm, and make it out-perform the rest. 

# In[ ]:





# Now we'll make a class in Surprise and inherit it from Algobase.

# In[ ]:


class HybridFacto(surprise.AlgoBase):
    def __init__(self,epochs, learning_rate,num_models):
        self.alpha = np.array([1/len(num_models)]*len(num_models))
        self.epochs = epochs
        self.learning_rate = learning_rate
    
    def fit(self,holdout):
        holdout=holdout.build_full_trainset().build_testset()
        for epoch in range(self.epochs):
            
            predictions = np.array([collabKNN.test(holdout),funkSVD.test(holdout),coClus.test(holdout),slopeOne.test(holdout)])
            maeGradient = [surprise.accuracy.mae(prediction) for prediction in predictions]
            newalpha = self.alpha - learning_rate * maeGradient  
            #convergence check:
            if newalpha - self.alpha < 0.001:
                break
            self.alpha = newalpha
            
    def estimate(self,u,i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
        algoResults = np.array([collabKNN.predict(u,i),funkSVD.predict(u,i),coClus.predict(u,i),slopeOne.predict(u,i)])
        return np.sum(np.dot(self.alpha,algoResults))
        


# lets test it out:

# In[ ]:


hybrid = HybridFacto(epochs=10,0.05,4)
hybrid.fit(holdout)
rmseHyb = []
for trainset, testset in kSplit.split(data): #iterate through the folds.
    predhybrid = Hyhybrid.test(testset)
    rmseHyb.append(surprise.accuracy.rmse(predhybrid))


# and lets plot it!
# 

# In[ ]:


#plotting the prediction data:
for prediction in compiledPredictions:
    modelPrediction = plt.plot(rmseKNN,label='knn')
    modelPrediction = plt.plot(rmseSVD,label='svd')
    modelPrediction = plt.plot(rmseCo,label='cluster')
    modelPrediction = plt.plot(rmseSlope,label='slope')
    modelPrediction = plt.plot(rmseHyb,label='Hybrid')

    modelPrediction = plt.xlabel('folds')
    modelPrediction = plt.ylabel('accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

