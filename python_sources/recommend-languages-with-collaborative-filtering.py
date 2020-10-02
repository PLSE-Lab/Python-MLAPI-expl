#!/usr/bin/env python
# coding: utf-8

# # Collaborative Filtering #
# 
# Collaborative filtering is a popular approach to creating recommender systems. Suppose we have the following problem:
# 
# User A gave five stars to _Titanic_ and _The Notebook_, and one star to _Captain America_ and _Dark Knight Rises_. Should we recommend to User A a film like _Love Actually_? What about _Batman vs Superman_?
# 
# In this case it's easy to tell what User A prefers since we have prior knowledge about the films. But how can we develop recommendations without this knowledge? 
# 
# It turns out that if we have rating data from many users it's possible to learn the underlying features that define a movie (genre, certain actors/directors, etc.) as well as the preferences of each user for each feature. Then, by identifying the features for a particular movie, and the preferences of a particular user, we can decide whether that movie would be a good recommendation. 
# 
# # Low Rank Matrix Factorization #
# 
# One method of implementing collaborative filtering is low rank matrix factorization. Suppose we have a matrix of ratings, with one column for each user and one row for each item. Call this matrix $Y$. A small number of the elements in the matrix will have values (the rating user A gives to item B, if it exists), but the vast majority of elements will be unknown. We also have a matrix $R$ which contains 1 in locations where a rating exists and 0 where a rating has not been given. Both $R$ and $Y$ have dimension $n_{items}$ by $n_{users}$. 
# 
# The process described above of learning underlying features and preferences is described mathematically by:
# 
# $ Y = X \theta^{T} $
# 
# $X$ is a matrix with dimension $n_{items}$ by $n_{features}$ and $\theta$ is a matrix with dimension $n_{users}$ by $n_{features}$. Each row in $X$ is the representation of a particular item as a linear combination of features, and each row in $\theta$ represents the preferences of a particular user for each feature. Note that a feature might be some characteristic such as genre or author - but it may also be very difficult to interpret. The beauty of the algorithm is that it will automatically choose the most meaningful features.
# 
# # Training and Implementation #
# 
# The goal of our algorithm is to find appropriate matrices $X$ and $\theta$ such that $X \theta^{T}$ approximates $Y$. However, it only needs to approximate $Y$'s values where a rating has actually been given - for some user A who has not rated item B, the value given by $X \theta^{T}$ at column A and row B is the _predicted_ rating that user A will give to item B. 
# 
# We can find $X$ and $\theta$ using gradient descent. The loss function will be the sum-of-squares loss between $X \theta^{T}$ and $Y$, _but only evaluated on locations where a rating has been given_, i.e. where $R_{i,j}=1$. Similarly, the gradient is only computed at these locations as well. 
# 
# For this particular analysis, the goal is to recommend programming languages for you to learn, given some languages you already like. We will approximate the preferences of users with the languages used in GitHub repositories. The items we rate will be languages. Once we compute $X$ and $\theta$, the recommended languages will be those with the highest predicted ratings. 

# # Acquiring Data From BigQuery #

# We will import a relatively small number of repositories (for speed) and their language details. 

# In[ ]:


import os
import pandas
import numpy as np

from google.cloud import bigquery
client = bigquery.Client()

#random sample of 300 repositories
QUERY = """
        SELECT repo_name, language
        FROM `bigquery-public-data.github_repos.languages`
        ORDER BY rand()
        LIMIT 300
        """

query_job = client.query(QUERY)

#filter out repositories with only one language
iterator = query_job.result(timeout=30)
rows = list(iterator)
rows = list(filter(lambda row: len(row.language)>1,rows))

#print some repositories
for i in range(10):
    print('Repository '+str(i+1))
    for j in rows[i].language:
        print(j[u'name']+': '+str(j[u'bytes'])+' bytes')
    print('')
print('...')
print(str(len(rows))+' repositories')


# # List of Languages #

# Create a list of all languages in the given sample.

# In[4]:


#create dictionary of language names to matrix columns
names = {}
for i in range(len(rows)):
    for j in rows[i].language:
        if j[u'name'] in names:
            names[j[u'name']]+=1
        else:
            names[j[u'name']]=1

#filter out languages that only occur once
names = [n for n in names if names[n]>1]
for i in range(10):
    print(names[i])
print('...')

#print some languages
name_to_index = {}
for j,i in enumerate(names):
    name_to_index[i] = j
print(str(len(names))+" languages")


# # Repository-Language Matrix #

# Create a matrix where each row represents a repository and each column represents a language. This matrix is our Y (i.e. what we are trying to predict). Here, if a language A is used in repository B, this is considered as repository B giving A a rating. If A is not used in B, then there is no rating (rather than a rating of 0). 
# 
# The value in the matrix is the log of number of bytes in the repository in that particular language.

# In[ ]:


from math import log

#create matrix
global mat
mat = np.zeros((len(rows),len(names)))
for i,row in enumerate(rows):
    #total = sum([log(lang[u'bytes']+1) for lang in row[1]])
    for lang in row.language:
        if lang[u'name'] in name_to_index:
            mat[i][name_to_index[lang[u'name']]] = log(lang[u'bytes'])
            #mat[i][name_to_index[lang[u'name']]] = log(lang[u'bytes']+1)/total
mat = mat[~np.all(mat==0,axis=1)]


# # PCA #

# Using PCA we can define roughly the number of features we want to identify the low rank matrix factorization. The graph below shows the amount of unexplained variance plotted against the number of components used. The "elbow" of the graph (at around n=12) is typically used.

# In[ ]:


from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

#compute PCA
n_components = min(50,len(names))
pca = PCA(n_components=n_components)
transformed = pca.fit_transform(mat) 

#display result
evr = [1-sum(pca.explained_variance_ratio_[:i+1]) for i in range(len(pca.explained_variance_ratio_))]
plt.plot(range(1,n_components+1),evr)


# # Loss Function and Gradient #

# We define some useful functions. 
# 
# init_mask: Create a mask matrix that indicates where Y has meaningful values. 
# 
# loss: Sum-of-squares loss, with a regularization term to prevent overfitting. The matrices theta and X are multiplied to give a "best guess", which is then compared with the target matrix Y, but only in locations where a rating has been given.
# 
# gradient: Derivative of loss with respect to theta and X, with a regularization term. 
# 
# These functions will be useful in performing gradient descent.

# In[ ]:


filter_size = min(100,len(mat[0]))
mat = mat[:,range(filter_size)] if len(mat[0])>filter_size else mat #for speed

#mask (R matrix)
def init_mask(Y):
    f = np.vectorize(lambda x: 1 if x>0 else 0)
    return f(Y),len(Y),len(Y[0])

#loss (sum of squares, with regularization)
def loss(args,Y,mask,n_repos,n_langs,n_features,reg_param):
    theta = np.reshape(args[:n_repos*n_features],(n_repos,n_features))
    X = np.reshape(args[n_repos*n_features:],(n_langs,n_features))
    g = np.vectorize(lambda x: x*x)
    return 0.5*np.sum(np.multiply(g(np.subtract(np.matmul(theta,np.transpose(X)),Y)),mask))+reg_param/2*np.sum(g(args))

#gradient
def gradient(args,Y,mask,n_repos,n_langs,n_features,reg_param):
    theta = np.reshape(args[:n_repos*n_features],(n_repos,n_features))
    X = np.reshape(args[n_repos*n_features:],(n_langs,n_features))
    X_grad = np.matmul(np.transpose(np.multiply(np.subtract(np.matmul(theta,np.transpose(X)),Y),mask)),theta)+reg_param*X
    theta_grad = np.matmul(np.multiply(np.subtract(np.matmul(theta,np.transpose(X)),Y),mask),X)+reg_param*theta
    return np.concatenate((np.reshape(theta_grad,-1),np.reshape(X_grad,-1)))


# # Training #

# Gradient descent is performed using loss and gradient as defined above. This will iteratively improve matrices theta and X, so that their product more closely matches the target matrix Y. 

# In[ ]:


import scipy.optimize as op

def train(Y,mask,n_repos,n_langs,n_features=10,reg_param=0.000001):
    #reshape into 1D format preferred by fmin_cg
    theta = np.random.rand(n_repos,n_features)
    X = np.random.rand(n_langs,n_features)
    args = np.concatenate((np.reshape(theta,-1),np.reshape(X,-1)))

    #use fmin_cg to perform gradient descent
    args = op.fmin_cg(lambda x: loss(x,Y,mask,n_repos,n_langs,n_features,reg_param),args,lambda x: gradient(x,Y,mask,n_repos,n_langs,n_features,reg_param))

    #reshape into a usable format
    theta = np.reshape(args[:n_repos*n_features],(n_repos,n_features))
    X = np.reshape(args[n_repos*n_features:],(n_langs,n_features))
    
    return theta,X


# # Recommendations #

# Now, we create a function for recommendations. Unfortunately Kaggle's front end doesn't allow for user input, so we will test some inputs manually.

# In[ ]:


def recommend(string,Y):
    #process input
    print('Training...')
    langs = string.split(' ')
    lc_names = {str(name).lower(): name_to_index[name] for name in name_to_index}

    #create extra row to append to Y matrix
    test = np.zeros((1,len(names)))
    known = set()
    for lang in langs:
        if lang.lower() in lc_names:
            test[0][lc_names[lang.lower()]] = 1
            known.add(lc_names[lang.lower()])

    #training
    Y = np.concatenate((Y,test[:,range(filter_size)]),0)
    mask,n_repos,n_langs = init_mask(Y)
    theta,X = train(Y,mask,n_repos,n_langs)
    Y = Y[:-1]
    
    #plot features
    for i in range(np.shape(X)[1]):
        col = sorted([(X[j,i],j) for j in range(n_langs)],reverse=True)
        #print('')
        #for k in range(10):
            #print(names[col[k][1]])

    #find top predictions
    predictions = np.matmul(theta,np.transpose(X))[-1].tolist()
    predictions = sorted([(abs(j),i) for i,j in enumerate(predictions)],reverse=True)

    #print predictions
    print('')
    i = 0
    for val,name in predictions:
        if name not in known:
            print(str(i+1)+': '+names[name]+' - '+str(val))
            i+=1
        if i>=5:
            break


# The recommender system adds the extra input row to target matrix Y. Then, training is performed on your language preferences simultaneously as those of all the repositories in the sample. Finally, the trained matrices theta and X are multiplied, and the last row corresponds to the predicted ratings based on your preferences. The highest values are the languages recommended to you. 
# 
# # Some Examples #

# In[ ]:


recommend('python r',mat)


# In[ ]:


recommend('html css',mat)


# In[ ]:


recommend('c',mat)


# In[ ]:


recommend('java',mat)


# In[ ]:


recommend('ruby',mat)


# Feel free to fork this notebook and try out some more examples, or adjust the parameters (regularization especially seems to make a big difference).
