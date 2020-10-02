#!/usr/bin/env python
# coding: utf-8

# An attempt to better understand KNN by making one from scratch in Python. Just using train_test_split to split the data and Counter to get most common element from a list, and numpy arrays. I hope I did everything correctly, feel free to leave feedback!

# In[ ]:


import pandas as pd
import numpy as np
#Using to split data randomly
from sklearn.model_selection import train_test_split
#Using to get most common element from list
from collections import Counter


# Start by reading in the data, dropping the ID Column and Unnamed, and any other null data

# In[ ]:


def readData():
    data = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
    #Drop Id as it has no influence on diagnosis, and Unnamed for Nan Vals
    data = data.drop(['id', 'Unnamed: 32'], axis = 1)
    data = data.dropna()
    return data


# Split data into a train and a test set. Converting them to numpy arrays, and returning them.

# In[ ]:



def splitData(data, testSize):
    y = data['diagnosis'].to_numpy()
    data = data.drop(['diagnosis'], axis = 1)
    X = data.to_numpy()
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=testSize)
    return Xtrain, Xtest, ytrain, ytest


# Our Xtest are the features of which we will actually try to predict the class of. We will do this by comparing them to the Xtrain values. So for each Xtest values (Xtestval), We will create a list (distances), of which will have the euclidean distance between xtestval and each Xtrain value. We will also need to make sure we keep track of the index of the Xtrain value to look up the corresponding ytrain value (the actual class of the Xtrain value). We will return this list sorted from lowest (best) to highest distances.

# In[ ]:


def euclidean(Xtrain, Xtestval):
    distances = []
    for i in range(len(Xtrain)):
        #euclidean equation
        distance = np.sqrt(np.sum(np.square(Xtestval-Xtrain[i])))
        distances.append([distance, i])
    return sorted(distances)


# This function will make a list of predictions (predict). It does this by taking the index of the k smallest distances (which is already sorted in the distances list so really the first k values of the list), and finding the value of the corresponding index of the y train values (the actual class of those xtrain features). We will then return the most common value in the predict list, which is our predicted class of the Xtest features.

# In[ ]:


def predict(Xtrain, ytrain, Xtestval, k):
    distances = euclidean(Xtrain, Xtestval)
    predict = []
    for i in range(k):
        predict.append(ytrain[distances[i][1]])
        
    return Counter(predict).most_common(1)[0][0]


# This is our actual KNN function, which just gets a predicted class for each set of values in our Xtest. Note k=3 in this example.

# In[ ]:


def KNN(Xtrain, Xtest, ytrain):
    predictions = []
    for i in range(len(Xtest)):
        predictions.append(predict(Xtrain, ytrain, Xtest[i], 3))
        
    return predictions


# This function just takes our predicted values and compares them to the actual ytest values, calculating the % we got correct.

# In[ ]:


def accuracy(ytest, predictions):
    correct = 0
    for i in range(len(predictions)):
        if(predictions[i]==ytest[i]):
            correct += 1
        
    score = (correct/len(ytest))*100
    return score


# main

# In[ ]:


data = readData()
Xtrain, Xtest, ytrain, ytest = splitData(data, 0.2)
predictions = KNN(Xtrain, Xtest, ytrain)


# In[ ]:


print(accuracy(ytest, predictions))


# On average when I ran it was between 90-95% when k=3 and testSize = 0.2
