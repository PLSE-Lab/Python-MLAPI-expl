#!/usr/bin/env python
# coding: utf-8

# #### Whenever we browse any website to search any product we experience that the website offers some other products which we might not have explicitly searched...this is what "Recommender System" does. It tries to identify patterns in the searches and recommends based on that.
# 
# #### Here I will show you 3 approaches to design a simple recommender system -
# 
# ### ***1. Gaussian Mixture Model and Expectation-Maximization Algorithm***
# 
# ### ***2. Altenating Least Squares using Non-Negative Matrix Factorization***
# 
# ### ***3. Stacked Auto-Encoder***

# #### Import the libraries.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image


# #### Load the data.

# In[ ]:


#Reading Users file:
u_cols = ['User_ID', 'Age', 'Sex', 'Occupation', 'ZIP_Code']
users = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.user',
                    sep='|', names=u_cols,encoding='latin-1')

#Reading Ratings file:
r_cols = ['User_ID', 'Movie_ID', 'Rating', 'Timestamp']
ratings = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.data',
                      sep='\t', names=r_cols,encoding='latin-1')


# In[ ]:


users.shape


# In[ ]:


users.head()


# #### So there are 943 users.

# In[ ]:


ratings.shape


# In[ ]:


ratings.head()


# #### There are 100,000 ratings of those 943 users.

# In[ ]:


len(ratings['Movie_ID'].unique())


# #### So there are 1682 unique movies that were rated.

# In[ ]:


nb_users  = users['User_ID'].nunique()
nb_movies = ratings['Movie_ID'].nunique()

print("There are %d unique users and %d unique movies; so we need to prepare " 
      "an matrix of size %d by %d." %(nb_users, nb_movies, nb_users, nb_movies))


# In[ ]:


ratings_matrix = ratings.pivot_table(index=['User_ID'],columns=['Movie_ID'],values='Rating').reset_index(drop=True)
ratings_matrix.fillna(0, inplace = True)

data_matrix = np.array(ratings_matrix)
print(data_matrix.shape)
print(data_matrix)


# #### Since we have reformatted the data into our desired format now we will proceed towards fitting the data into different models.

# # Gaussian Mixture Model and Expectation-Maximization Algorithm

# #### A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. One can think of mixture models as generalizing k-means clustering to incorporate information about the covariance structure of the data as well as the centers of the latent Gaussians. Bayesian Information Criterion (BIC) is used as a performance measure (lower the better) in Gaussian Mixture model.
# 
# #### In the context of each user, we must sample a user type and then the rating profile from the Gaussian distribution associated with the type. We will use the Expectation Maximization (EM) algorithm to estimate such a mixture from a partially observed rating matrix. The EM algorithm proceeds by iteratively assigning (softly) users to types (E-step) and subsequently re-estimating the Gaussians associated with each type (M-step). Once we have the mixture, we can use it to predict values for all the missing entries in the data matrix.
# 
# #### The Scikit-Learn Gaussian Mixture model uses EM algorithm.

# In[ ]:


from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
import itertools


# #### Build a GMM model using covariance_type='full' so that the covariance matrix become a square matrix and n_components=2.

# In[ ]:


gmm_model = GaussianMixture(n_components=2, covariance_type='full', 
                            tol=0.001, reg_covar=1e-06, max_iter=100, 
                            n_init=1, init_params='kmeans', weights_init=None, 
                            means_init=None, precisions_init=None, random_state=42, 
                            warm_start=False, verbose=0, verbose_interval=10)
gmm_model.fit(data_matrix)


# In[ ]:


print(gmm_model.means_.shape)
print(gmm_model.covariances_.shape)
print(gmm_model.weights_.shape)


# #### So the GMM model assigned 943 users into 2 components and since there is 1682 movies (features); each feature has its own mean and variance for each component.
# 
# #### If we use "Predict" method we will get 943 outputs that specifies the component label but since we have to predict what will be the rating of 1682 movies for each user we have to implement below logic.

# In[ ]:


Image("../input/input/4.JPG")


# #### The entire computation will be done in log-domain to avoid Numerical instability; so the log of Posterior probability will be as below.

# In[ ]:


Image("../input/input/5.JPG")


# In[ ]:


#Fill Missing Values i.e Recommend
inver0 = np.linalg.inv(gmm_model.covariances_[0])
inver1 = np.linalg.inv(gmm_model.covariances_[1])
deter0 = np.linalg.det(gmm_model.covariances_[0])
deter1 = np.linalg.det(gmm_model.covariances_[1])

n = data_matrix.shape[0]
d = data_matrix.shape[1]
K = gmm_model.means_.shape[0]
mean = gmm_model.means_
variance = gmm_model.covariances_
weight = np.log(gmm_model.weights_)
calc = np.zeros((n, K))
ind = np.zeros((n, d))
soft = calc
add = np.zeros((n,))
dim = np.zeros((n,))
X_pred = ind
    
ind = np.where(data_matrix != 0, 1, 0)            
dim = np.sum(ind, axis=1)

for i in range(n):
    for j in range(K):
        res = data_matrix[i] - mean[j]
        res = np.multiply(res, ind[i])
        #Multivariate Gaussian
        if j == 0:
            A = (res.T @ inver0) @ res
            C = (dim[i]/2)*np.log(2*np.pi) + np.log(deter0 + 1e-16)/2
        else:
            A = (res.T @ inver1) @ res
            C = (dim[i]/2)*np.log(2*np.pi) + np.log(deter1 + 1e-16)/2
        B = 2
        calc[i, j] = weight[j] + (-A/B) - C

add = logsumexp(calc, axis = 1)

#Since the entire computation is done in log-domain to avoid Numerical instability
#we need to bring it back in its original domain
soft = np.exp(np.subtract(np.transpose(calc), add))

lg = np.sum(add)
    
X_calc = np.transpose(soft) @ gmm_model.means_

#We will use predicted value if the entry is 0 in original rating matrix
data_matrix_pred_GMM = np.where(data_matrix == 0, X_calc, data_matrix)

for i in range(data_matrix_pred_GMM.shape[0]):
    for j in range(data_matrix_pred_GMM.shape[1]):
        data_matrix_pred_GMM[i, j] = round(data_matrix_pred_GMM[i, j])

#For measuring the performance we have to use the predicted matrix
for i in range(X_calc.shape[0]):
    for j in range(X_calc.shape[1]):
        X_pred[i, j] = round(X_calc[i, j])


# In[ ]:


print("Original Rating Matrix: \n", data_matrix)


# In[ ]:


print("Rating Matrix After Applying GMM: \n", data_matrix_pred_GMM)


# #### Measure the performance: we will consider the entries from the original ratings matrix and see their predicted values and then compute RMSE.

# In[ ]:


ind_matrix = np.zeros((nb_users, nb_movies))
ind_matrix = np.where(data_matrix != 0, 1, 0)

x = np.multiply(X_pred, ind_matrix)
RMSE_GMM = np.sqrt(np.mean((x - data_matrix)**2))
print("RMSE of GMM Model is %f." %RMSE_GMM)


# # Altenating Least Squares using Non-Negative Matrix Factorization

# #### But before we dig into the details of ALS we want to understand what is Non Negative Matrix Factorization(NMF). NMF is finding out two non-negative matrices (W, H) whose product approximates the non- negative matrix X. This factorization can be used for example for dimensionality reduction, source separation or topic extraction.

# In[ ]:


# Understanding Non-Negative Matrix Factorization(NMF)
from sklearn.decomposition import NMF
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve

X = np.array([[1, 2, 3], [5, 10, 15]])
print("X is:\n", X)
model = NMF(n_components=2, init='random', random_state=42)
W = model.fit_transform(X)
H = model.components_
print("W is:\n", W)
print("H is:\n", H)
print("The Result of Matrix Multiplication of W and H is Same as X:\n", np.matmul(W, H))


# #### The Loss function of ALS is as below where X_u is the user_vector, y_i is the item_vector which are created applying NMF on the rating matrix and r_ui is basically the rating. There are Regularization terms as well.
# 
# #### One important point here is that to minimize this Loss function only user_id and item_id combinations for which there is a rating is considered (u,i belongs to S which is the non-zero rating matrix).

# In[ ]:


Image("../input/inputimage/1.JPG")


# #### The minimization of the Loss function yields below matrix form for the user_vector and item_vector.

# In[ ]:


Image("../input/inputimage/2.JPG")


# In[ ]:


Image("../input/inputimage/3.JPG")


# #### Now we will implement ALS.

# In[ ]:


model = NMF(n_components=2, init='random', random_state=42)
user_vec = model.fit_transform(data_matrix)
item_vec = model.components_.T

def implicit_ALS(ratings, user_vec, item_vec, lambda_val, iteration, typ):                 
    
    ctr = 1

    if typ == 'user':
        while ctr <= iteration:
            YTY = item_vec.T.dot(item_vec)
            lambdaI = np.eye(YTY.shape[0]) * lambda_val

            for u in range(user_vec.shape[0]):
                user_vec[u, :] = solve((YTY + lambdaI), 
                                        ratings[u, :].dot(item_vec))
            ctr += 1

        return user_vec
    
    if typ == 'item':
        while ctr <= iteration:
            XTX = user_vec.T.dot(user_vec)
            lambdaI = np.eye(XTX.shape[0]) * lambda_val
            
            for i in range(item_vec.shape[0]):
                item_vec[i, :] = solve((XTX + lambdaI), 
                                        ratings[:, i].T.dot(user_vec))
            ctr += 1
        return item_vec
        
    
user_vec = implicit_ALS(data_matrix, user_vec, item_vec, lambda_val=0.2,
                        iteration=20, typ='user')
item_vec = implicit_ALS(data_matrix, user_vec, item_vec, lambda_val=0.2,
                        iteration=20, typ='item')

def predict_all():
        """ Predict ratings for every user and item. """
        predictions = np.zeros((user_vec.shape[0], 
                                item_vec.shape[0]))
        for u in range(user_vec.shape[0]):
            for i in range(item_vec.shape[0]):
                predictions[u, i] = predict(u, i)
                
        return predictions
def predict(u, i):
    """ Single user and item prediction. """
    return user_vec[u, :].dot(item_vec[i, :].T)

predict = predict_all()


data_matrix_pred_ALS = np.where(data_matrix == 0, predict, data_matrix)

for i in range(data_matrix_pred_ALS.shape[0]):
    for j in range(data_matrix_pred_ALS.shape[1]):
        data_matrix_pred_ALS[i, j] = round(data_matrix_pred_ALS[i, j])

#For measuring the performance we have to use the predicted matrix
X_pred = np.zeros((nb_users, nb_movies))
for i in range(predict.shape[0]):
    for j in range(predict.shape[1]):
        X_pred[i, j] = round(predict[i, j])


# In[ ]:


print("Original Rating Matrix: \n", data_matrix)


# In[ ]:


print("Rating Matrix After Applying ALS: \n", data_matrix_pred_ALS)


# #### Measure the performance: we will consider the entries from the original ratings matrix and see their predicted values and then compute RMSE.

# In[ ]:


ind_matrix = np.zeros((nb_users, nb_movies))
ind_matrix = np.where(data_matrix != 0, 1, 0)

x = np.multiply(X_pred, ind_matrix)
RMSE_ALS = np.sqrt(np.mean((x - data_matrix)**2))
print("RMSE of ALS Model is %f." %RMSE_ALS)


# # Auto-Encoder

# #### Auto Encoder is a Unsupervised learning technique that creates a representation of actual data in a reduced dimension. We will use one Auto Encoder on top of another one to make it Stacked.

# #### Import the PyTorch modules.

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


# #### Converting the data into Torch tensors.

# In[ ]:


data_matrix_torch = torch.FloatTensor(data_matrix)


# #### Creating the architecture of Stacked AutoEncoder model.

# In[ ]:


class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def prediction(self, x):
        pred = self.forward(x)
        return pred.detach().numpy()
    
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)


# #### Train the Stacked Auto Encoder.

# In[ ]:


nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0
    for id_user in range(nb_users):
        input = Variable(data_matrix_torch[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data * mean_corrector)
            s +=1
            optimizer.step()
    print('epoch: ' + str(epoch) + ' loss: ' + str((train_loss/s).item()))


# #### Predict using this Stacked Auto-Encoder.

# In[ ]:


predict_SAE = np.zeros((nb_users, nb_movies))
for id_user in range(nb_users):
    input = Variable(data_matrix_torch[id_user]).unsqueeze(0)
    predict_SAE[id_user] = sae.prediction(input)

#We will use predicted value if the entry is 0 in original rating matrix
data_matrix_pred_SAE = np.where(data_matrix == 0, predict_SAE, data_matrix)

for i in range(data_matrix_pred_SAE.shape[0]):
    for j in range(data_matrix_pred_SAE.shape[1]):
        data_matrix_pred_SAE[i, j] = round(data_matrix_pred_SAE[i, j])

#For measuring the performance we have to use the predicted matrix
X_pred = np.zeros((nb_users, nb_movies))
for i in range(predict_SAE.shape[0]):
    for j in range(predict_SAE.shape[1]):
        X_pred[i, j] = round(predict_SAE[i, j])


# In[ ]:


print("Original Rating Matrix: \n", data_matrix)


# In[ ]:


print("Rating Matrix after Applying Stacked Auto-Encoder: \n", data_matrix_pred_SAE)


# #### Measure the performance: we will consider the entries from the original ratings matrix and see their predicted values and then compute RMSE.

# In[ ]:


ind_matrix = np.zeros((nb_users, nb_movies))
ind_matrix = np.where(data_matrix != 0, 1, 0)

x = np.multiply(X_pred, ind_matrix)
RMSE_SAE = np.sqrt(np.mean((x - data_matrix)**2))
print("RMSE of SAE Model is %f." %RMSE_SAE)


# #### Since the RMSE of the Stacked Auto-Encoder model is comparatively less than GMM and ALS based models and we can see that the output rating matrix is completely filled up we will use this matrix to show how it can be used to *recommend*.
# 
# #### We will first select a particular movie from the 'movies' dataset and using its 'Movie_ID' we will select the top 5 entries from the output rating matrix for which the rating is greater than/equal to 4 and the movie was not already rated because if it is rated then the user has watched the movie and there is no point in recommending same movie to that user. We can use the index of these top 5 entries to fetch the user information.
# 
# #### Alternatively, we will select a particular user and select top 5 movies which are not previously rated by the user and our recommender system predicted that the predicted rating will be greater than/equal to 4 for them.

# In[ ]:


m_cols = ['Movie_ID', 'Title', 'Release_Date', 'Video_Release_Date', 'IMDB_URL']
movies = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.item', sep='|',
                     names=m_cols, usecols=range(5),encoding='latin-1')
movies.head(10)


# #### Lets's select the movie as 'Richard III (1995)' and see how we can recommend it.

# In[ ]:


movie_id = movies[movies['Title'] == 'Richard III (1995)']['Movie_ID'].values.item()
print("Movie ID is:", movie_id)


# #### Since the Movie_ID is 10, in our predicted rating matrix the columns index will be 9.

# In[ ]:


#Create an indicator matrix to ensure the movie was not rated previously
ind_matrix = np.zeros((nb_users, nb_movies))
ind_matrix = np.where(data_matrix == 0, 1, 0)

#Multiply predicted rating matrix with this indicator matrix to consider only
#the predicted ones
pred = np.multiply(data_matrix_pred_SAE, ind_matrix)
pred = pred[:, 9]
pred_df = pd.DataFrame(pred)
pred_df.columns = ['Rating']
pred_df = pred_df[pred_df['Rating'] >= 4]
pred_df = pred_df.head(5)
pred_df


# #### So we can see the top 5 users who will possibly rate 'Richard III (1995)' as 4 have their User_ID as 3, 4, 5, 6, 8 (add 1 to the index).

# In[ ]:


user_id = [3, 4, 5, 6, 8]
users_recommend = users[users['User_ID'].isin(user_id)]
users_recommend


# #### We will now select a particular user and recommend movies to that user.

# In[ ]:


users.tail()


# #### Let's select User_ID as 940; so in the output predicted rating matrix the row index will be 939.

# In[ ]:


#Create an indicator matrix to ensure the movie was not rated previously
ind_matrix = np.zeros((nb_users, nb_movies))
ind_matrix = np.where(data_matrix == 0, 1, 0)

#Multiply predicted rating matrix with this indicator matrix to consider
#only the predicted ones
pred = np.multiply(data_matrix_pred_SAE, ind_matrix)
pred = pred[939, :]
pred_df = pd.DataFrame(pred)
pred_df.columns = ['Rating']
pred_df = pred_df[pred_df['Rating'] >= 4]
pred_df = pred_df.head(5)
pred_df


# #### So we can see the top 5 movies which will possibly be rated by User_ID 940 as 4 have their Movie_ID as 1, 2, 10, 11, 15 (add 1 to the index).

# In[ ]:


movie_id = [1, 2, 10, 11, 15]
movie_recommend = movies[movies['Movie_ID'].isin(movie_id)]
movie_recommend


# #### I tried to showcase how a tiny Recommender System can be created using different techniques and briefly explained their basics.
# 
# #### Please upvote if you like it!
