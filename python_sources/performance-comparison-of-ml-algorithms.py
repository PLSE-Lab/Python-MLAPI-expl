#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# reading data from csv files and converting to matrix 

import pandas as pd 

test  = pd.read_csv("../input/test.csv")  
train = pd.read_csv("../input/train.csv") 


# In[ ]:


# suffling data 

from sklearn.utils import shuffle

test  = shuffle(test)
train = shuffle(train)


# In[ ]:


# separating data inputs and output lables 

trainData  = train.drop('Activity' , axis=1).values
trainLabel = train.Activity.values

testData  = test.drop('Activity' , axis=1).values
testLabel = test.Activity.values


# In[ ]:


# encoding labels 

from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()

# encoding test labels 
encoder.fit(testLabel)
testLabelE = encoder.transform(testLabel)

# encoding train labels 
encoder.fit(trainLabel)
trainLabelE = encoder.transform(trainLabel)


# 

# In[ ]:


# applying supervised neural network using multi-layer preceptron 

import sklearn.neural_network as nn 

mlpSGD  =  nn.MLPClassifier(hidden_layer_sizes=(90,)                          , max_iter=1000 , alpha=1e-4                          , solver='sgd' , verbose=10                           , tol=1e-19 , random_state=1                          , learning_rate_init=.001) 
mlpADAM =  nn.MLPClassifier(hidden_layer_sizes=(90,)                          , max_iter=1000 , alpha=1e-4                          , solver='adam' , verbose=10                          , tol=1e-19 , random_state=1                          , learning_rate_init=.001) 

nnModelSGD  = mlpSGD .fit(trainData , trainLabelE)
nnModelADAM = mlpADAM.fit(trainData , trainLabelE)


# In[ ]:


# ploting nn convergence and testing score 

import matplotlib.pyplot as plt 
import numpy             as np

X1 = np.linspace(1, nnModelSGD.n_iter_  , nnModelSGD.n_iter_ )
X2 = np.linspace(1, nnModelADAM.n_iter_ , nnModelADAM.n_iter_)

plt.plot(X1 , nnModelSGD.loss_curve_ , label = 'SGD Convergence' )
plt.plot(X2 , nnModelADAM.loss_curve_, label = 'ADAM Convergence')
plt.title('Error Convergence ')
plt.ylabel('Cost function')
plt.xlabel('Iterations')
plt.legend()
plt.show()


# In[ ]:


# generating test scores for both classifiers 

print("Training set score for SDG : %f" % mlpSGD .score(trainData, trainLabelE))
print("Training set score for ADAM: %f" % mlpADAM.score(trainData, trainLabelE))
print("Test set score for SDG : %f"     % mlpSGD .score(testData , testLabelE ))
print("Test set score for ADAM: %f"     % mlpADAM.score(testData , testLabelE ))


# ## K - Nearest Neighbor ##

# In[ ]:


# applying supervised k -Nearest Neighbor

from sklearn.neighbors import KNeighborsClassifier as knn

knnclf = knn(n_neighbors=20 , n_jobs=2 , weights='distance')

knnModel = knnclf.fit(trainData , trainLabelE)

print("Training set score for KNN: %f" % knnModel.score(trainData , trainLabelE))
print("Testing  set score for KNN: %f" % knnModel.score(testData  , testLabelE ))


# In[ ]:




