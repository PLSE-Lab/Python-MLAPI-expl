#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# Most neutron stars are observed as pulsars. Pulsars are rotating neutron stars observed to have pulses of radiation at very regular intervals that typically range from milliseconds to seconds. Pulsars have very strong magnetic fields which funnel jets of particles out along the two magnetic poles. These accelerated particles produce very powerful beams of light.(NASA)
# 
# In this kernel we will try to write supervised machine learning algorithms to predict the class of pulsars.
# 
# 
# 

# 1. [Logistic Regression](#1)
# 1. [Logistic Regression with sklearn](#2)
# 1. [KNN](#3)
# 1. [SVM](#4)
# 1. [Naive Bayes](#5)
# 1. [Decision Tree](#6)
# 1. [Random Forest](#7)
# 1. [Conclusion](#8)
# 

# * Importing libraries and data.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/pulsar_stars.csv")


# In[ ]:


df.info()


# In[ ]:


df.head()


# * Setting x and y and normalize data.

# In[ ]:


x_data = df.drop(["target_class"], axis = 1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
y = df.target_class.values

compareScore = []


# * Train test split process

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


# <a id="1"></a> <br>
# **Logistic Regression**

# * Initializing parameters and creating sigmoid function

# In[ ]:


def initializeWeightBias(dimension):
    w = np.full((dimension,1), 0.01)
    b = 0.0
    return w, b

def sigmoidFunc(z):
    y_head = 1 / (1 + np.exp(-z))
    return y_head
    


# * Forward and backward propagation

# In[ ]:


def fwPropagation(w, b, x_train, y_train):
    #forward
    z = np.dot(w.T, x_train.T) + b
    y_head = sigmoidFunc(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss)) / x_train.shape[0]
    
    #backward
    deriWeight = (np.dot(x_train.T,((y_head-y_train.T).T)))/x_train.shape[0] 
    deriBias = np.sum(y_head-y_train.T)/x_train.shape[0]
    graDescents = {"deriWeight" : deriWeight, "deriBias" : deriBias}
    
    return graDescents, cost
    


# * Updating parameters.  (Learning)

# In[ ]:


def update(iterNumber, w, b, x_train, y_train, learningRate) :
    costList = []
    index = []
    for i in range(iterNumber + 1):
        graDescents, cost = fwPropagation(w, b, x_train, y_train)
        w = w - learningRate*graDescents["deriWeight"]
        b = b - learningRate*graDescents["deriBias"]
        
        if i % 10 == 0:
            costList.append(cost)
            index.append(i)
            print("Cost after {} iteration = {}".format(i, cost))
            
    parameters = {"weight" : w, "bias" : b}
    
    return parameters, costList, index     


# * We can plot the graph for seeing decrease of cost if we will.

# In[ ]:


def plotGraph(index, costList):
    plt.plot(index, costList)
    plt.ylabel("Cost")
    plt.show()
    


# * Predict process

# In[ ]:


def predict(w, b, x_test):
    z = np.dot(w.T, x_test.T) + b
    y_head = sigmoidFunc(z)
    yPrediction = np.zeros((1,x_test.shape[0]))
    
    for i in range(y_head.shape[1]):
        if y_head[0,i] <= 0.5:
            yPrediction[0,i] = 0
        else:
            yPrediction[0,i] = 1
            
    return yPrediction


# * And now, we are at beautiful part :)

# In[ ]:


def logisticRegression(x_train, y_train, x_test, y_test, iterNumber, learningRate):
    dimension = x_train.shape[1]
    w, b = initializeWeightBias(dimension)
    parameters, costList, index = update(iterNumber, w, b, x_train, y_train, learningRate)
    
    predictionTest = predict(parameters["weight"], parameters["bias"], x_test)
    
    #printing errors
    print("Test accuracy: {} %".format(100 - np.mean(np.abs(predictionTest - y_test)) * 100))
    
    plotGraph(index, costList)
    
logisticRegression(x_train, y_train, x_test, y_test, iterNumber = 30, learningRate = 0.5)


# %90 accuracy, not bad.

# <a id="2"></a> <br>
# * Now, we'll do it with sklearn. 

# In[ ]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)

lrScore = lr.score(x_test, y_test) * 100
compareScore.append(lrScore)

print("Test accuracy: {} %".format(lrScore))


# And %97 accuracy with sklearn logistic regression. It is very good rate. Let's look at other supervised learning algorithms, try to predict classes and see the accuracy of predictions.

# <a id="3"></a> <br>
# **K-Nearest Neighbour (KNN) Classification**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(x_train, y_train)
print("Test accuracy: {}%".format(knn.score(x_test,y_test)*100))


# We got a good rate but we don't know whether this n_neighbours value(currently 4) providing best accuracy or not. 
# 
# So we should figure it out.

# In[ ]:


scoreList = []
n = 15
for i in range(1,n):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(x_train, y_train)
    scoreList.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,n), scoreList)
plt.ylabel("Accuracy rate")
plt.show()


# As you can see above, best value for best accuracy is n_neighbours = 8.

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=8)

knn.fit(x_train, y_train)

knnScore = knn.score(x_test,y_test)*100
compareScore.append(knnScore)

print("Test accuracy: {}%".format(knnScore))


# <a id="4"></a> <br>
# **Support Vector Machine (SVM) Classification**

# In[ ]:


from sklearn.svm import SVC
svm = SVC(random_state=42, gamma = "scale")

svm.fit(x_train, y_train)

svmScore = svm.score(x_test,y_test)*100
compareScore.append(svmScore)

print("Test accuracy: {}%".format(svmScore))


# This was a very good prediction too.

# <a id="5"></a> <br>
# **Naive Bayes Classification**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

nb.fit(x_train, y_train)

nbScore = nb.score(x_test,y_test)*100
compareScore.append(nbScore)

print("Test accuracy: {}%".format(nbScore))


# This was a good accuracy too but this is a bit lower than the others. That means this classification method is not convenient for this dataset so much. 

# <a id="6"></a> <br>
# **Decision Tree Classification**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42, max_depth=4)

dt.fit(x_train, y_train)

dtScore = dt.score(x_test, y_test)*100
compareScore.append(dtScore)

print("Test accuracy: {}%".format(dtScore))


# <a id="7"></a> <br>
# **Random Forest Classification**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42, n_estimators=10)

rf.fit(x_train, y_train)

rfScore = rf.score(x_test, y_test)*100
compareScore.append(rfScore)

print("Test accuracy: {}%".format(rfScore))


# We got 98% which is the best by far. 

# * Comparison of models' accuracy

# In[ ]:


from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

algoList = ["LogisticRegression", "KNN", "SVM", "NaiveBayes", "DecisionTree", "RandomForest"]
comparison = {"Models" : algoList, "Accuracy" : compareScore}
dfComparison = pd.DataFrame(comparison)

newIndex = (dfComparison.Accuracy.sort_values(ascending = False)).index.values
sorted_dfComparison = dfComparison.reindex(newIndex)


data = [go.Bar(
               x = sorted_dfComparison.Models,
               y = sorted_dfComparison.Accuracy,
               name = "Scores of Models",
               marker = dict(color = "rgba(116,173,209,0.8)",
                             line=dict(color='rgb(0,0,0)',width=1.0)))]

layout = go.Layout(xaxis= dict(title= 'Models',ticklen= 5,zeroline= False))

fig = go.Figure(data = data, layout = layout)

iplot(fig)


# <a id="8"></a> <br>
# ## Conclusion
# In a nutshell, we wrote a bunch of supervised machine learning algorithms and we made really good predictions. Please upvote if you like and leave a comment.
