#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# I will comparison ML Classification Algorithms.
# 
# 1. Logistic Regression
# 2. KNN
# 3. SVM
# 4. Naive B.
# 5. Random Forest 
# 
# <img src="https://mobilemonitoringsolutions.com/wp-content/uploads/2018/09/mldepressiongif.gif" width="500px">
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import itertools
plt.style.use('fivethirtyeight')
import seaborn as sns
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Read and Examination Dataset

# In[ ]:


#read data
data = pd.read_csv("../input/diabetes.csv")


# In[ ]:


#data.info()


# In[ ]:


#data.head()


# In[ ]:


#Split Data as M&B
p = data[data.Outcome == 1]
n = data[data.Outcome == 0]


# # Visualization

# In[ ]:


sns.countplot(x='Outcome',data=data)
plt.title("Count 0 & 1")
plt.show()


# **Analysis of Diabetic Cases**

# In[ ]:


#General Analysis

data1 = data[data["Outcome"]==1]
columns = data.columns[:8]
plt.subplots(figsize=(18,18))
length =len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    plt.ylabel("Count")
    data1[i].hist(bins=20,edgecolor='black')
    plt.title(i)
plt.show()


# **Analysis of Non-Diabetic Cases**

# In[ ]:


#General Analysis

data1 = data[data["Outcome"]==0]
columns = data.columns[:8]
plt.subplots(figsize=(18,18))
length =len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    plt.ylabel("Count")
    data1[i].hist(bins=20,edgecolor='black')
    plt.title(i)
plt.show()


# In[ ]:


#Visualization, Scatter Plot

plt.scatter(p.Pregnancies,p.Glucose,color = "brown",label="Diabet Positive",alpha=0.4)
plt.scatter(n.Pregnancies,n.Glucose,color = "Orange",label="Diabet Negative",alpha=0.2)
plt.xlabel("Pregnancies")
plt.ylabel("Glucose")
plt.legend()
plt.show()

#We appear that it is clear segregation.


# In[ ]:


#Visualization, Scatter Plot

plt.scatter(p.Age,p.Pregnancies,color = "lime",label="Diabet Positive",alpha=0.4)
plt.scatter(n.Age,n.Pregnancies,color = "black",label="Diabet Negative",alpha=0.2)
plt.xlabel("Age")
plt.ylabel("Pregnancies")
plt.legend()
plt.show()

#We appear that it is clear segregation.


# In[ ]:


#Visualization, Scatter Plot

plt.scatter(p.Glucose,p.Insulin,color = "lime",label="Diabet Positive",alpha=0.4)
plt.scatter(n.Glucose,n.Insulin,color = "black",label="Diabet Negative",alpha=0.1)
plt.xlabel("Glucose")
plt.ylabel("Insulin")
plt.legend()
plt.show()

#We appear that it is clear segregation.


# # Edit and Separate Dataset

# In[ ]:


#separate data as x (features) & y (labels)
y= data.Outcome.values
x1= data.drop(["Outcome"],axis= 1) #we remowe diagnosis for predict


# In[ ]:


#normalization
x = (x1-np.min(x1))/(np.max(x1)-np.min(x1))


# # Comparison of ML Classification Algorithms

# In[ ]:


#Train-Test-Split 
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest =  train_test_split(x,y,test_size=0.3,random_state=42)


# # Logistic Regression Classification

# In[ ]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()


# In[ ]:


#K-fold CV
from sklearn.model_selection import cross_val_score
accuraccies = cross_val_score(estimator = LR, X= xtrain, y=ytrain, cv=10)
print("Average Accuracies: ",np.mean(accuraccies))
print("Standart Deviation Accuracies: ",np.std(accuraccies))


# In[ ]:


LR.fit(xtrain,ytrain)
print("Test Accuracy {}".format(LR.score(xtest,ytest))) 

LRscore = LR.score(xtest,ytest)


# In[ ]:


#Confusion Matrix

yprediciton1= LR.predict(xtest)
ytrue = ytest

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(ytrue,yprediciton1)

#CM visualization

import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(CM,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Prediction(Ypred)")
plt.ylabel("Ytrue")
plt.show()


# # K-NN

# In[ ]:


#Create-KNN-model
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 40) #n_neighbors = K value


# In[ ]:


#K-fold CV
from sklearn.model_selection import cross_val_score
accuraccies = cross_val_score(estimator = KNN, X= xtrain, y=ytrain, cv=10)
print("Average Accuracies: ",np.mean(accuraccies))
print("Standart Deviation Accuracies: ",np.std(accuraccies))


# In[ ]:


KNN.fit(xtrain,ytrain) #learning model
prediction = KNN.predict(xtest)
#Prediction
print("{}-NN Score: {}".format(40,KNN.score(xtest,ytest)))

KNNscore = KNN.score(xtest,ytest)


# In[ ]:


#Find Optimum K value
scores = []
for each in range(1,100):
    KNNfind = KNeighborsClassifier(n_neighbors = each)
    KNNfind.fit(xtrain,ytrain)
    scores.append(KNNfind.score(xtest,ytest))

plt.figure(1, figsize=(10, 5))
plt.plot(range(1,100),scores,color="black",linewidth=2)
plt.title("Optimum K Value")
plt.xlabel("K Values")
plt.ylabel("Score(Accuracy)")
plt.grid(True)
plt.show()


# In[ ]:


#Confusion Matrix

yprediciton2= KNN.predict(xtest)
ytrue = ytest

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(ytrue,yprediciton2)

#CM visualization

import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(CM,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Prediction(Ypred)")
plt.ylabel("Ytrue")
plt.show()


# # SVM

# In[ ]:


#SVM with Sklearn

from sklearn.svm import SVC

SVM = SVC(random_state=42)


# In[ ]:


#K-fold CV
from sklearn.model_selection import cross_val_score
accuraccies = cross_val_score(estimator = SVM, X= xtrain, y=ytrain, cv=10)
print("Average Accuracies: ",np.mean(accuraccies))
print("Standart Deviation Accuracies: ",np.std(accuraccies))


# In[ ]:


SVM.fit(xtrain,ytrain)  #learning 
#SVM Test 
print ("SVM Accuracy:", SVM.score(xtest,ytest))

SVMscore = SVM.score(xtest,ytest)


# In[ ]:


#Confusion Matrix

yprediciton3= SVM.predict(xtest)
ytrue = ytest

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(ytrue,yprediciton3)

#CM visualization

import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(CM,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Prediction(Ypred)")
plt.ylabel("Ytrue")
plt.show()


# # Naive Bayes Classification

# In[ ]:


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()


# In[ ]:


#K-fold CV
from sklearn.model_selection import cross_val_score
accuraccies = cross_val_score(estimator = NB, X= xtrain, y=ytrain, cv=10)
print("Average Accuracies: ",np.mean(accuraccies))
print("Standart Deviation Accuracies: ",np.std(accuraccies))


# In[ ]:


NB.fit(xtrain,ytrain) #learning
#prediction
print("Accuracy of NB Score: ", NB.score(xtest,ytest))

NBscore= NB.score(xtest,ytest)


# In[ ]:


#Confusion Matrix

yprediciton4= NB.predict(xtest)
ytrue = ytest

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(ytrue,yprediciton4)

#CM visualization

import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(CM,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Prediction(Ypred)")
plt.ylabel("Ytrue")
plt.show()


# # Decision Tree
# <img src="https://emerj.com/wp-content/uploads/2018/04/3049155-poster-p-1-machine-learning-is-just-a-big-game-of-plinko.gif" width="500px">
# 

# In[ ]:


#Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()


# In[ ]:


#K-fold CV
from sklearn.model_selection import cross_val_score
accuraccies = cross_val_score(estimator = DTC, X= xtrain, y=ytrain, cv=10)
print("Average Accuracies: ",np.mean(accuraccies))
print("Standart Deviation Accuracies: ",np.std(accuraccies))


# In[ ]:


DTC.fit(xtrain,ytrain) #learning
#prediciton
print("Decision Tree Score: ",DTC.score(xtest,ytest))
DTCscore = DTC.score(xtest,ytest)


# In[ ]:


#Confusion Matrix

yprediciton5= DTC.predict(xtest)
ytrue = ytest

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(ytrue,yprediciton5)

#CM visualization

import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(CM,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Prediction(Ypred)")
plt.ylabel("Ytrue")
plt.show()


# # Random Forest

# In[ ]:


#Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()
DTC.fit(xtrain,ytrain) #learning
#prediciton
print("Decision Tree Score: ",DTC.score(xtest,ytest))


# In[ ]:


#Random Forest

from sklearn.ensemble import RandomForestClassifier
RFC= RandomForestClassifier(n_estimators = 24, random_state=42) #n_estimator = DT


# In[ ]:


#K-fold CV
from sklearn.model_selection import cross_val_score
accuraccies = cross_val_score(estimator = RFC, X= xtrain, y=ytrain, cv=10)
print("Average Accuracies: ",np.mean(accuraccies))
print("Standart Deviation Accuracies: ",np.std(accuraccies))


# In[ ]:


RFC.fit(xtrain,ytrain) # learning
print("Random Forest Score: ",RFC.score(xtest,ytest))
RFCscore=RFC.score(xtest,ytest)


# In[ ]:


#Find Optimum K value
scores = []
for each in range(1,30):
    RFfind = RandomForestClassifier(n_estimators = each)
    RFfind.fit(xtrain,ytrain)
    scores.append(RFfind.score(xtest,ytest))
    
plt.figure(1, figsize=(10, 5))
plt.plot(range(1,30),scores,color="black",linewidth=2)
plt.title("Optimum N Estimator Value")
plt.xlabel("N Estimators")
plt.ylabel("Score(Accuracy)")
plt.grid(True)
plt.show()


# In[ ]:


#Confusion Matrix

yprediciton6= RFC.predict(xtest)
ytrue = ytest

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(ytrue,yprediciton6)

#CM visualization

import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(CM,annot = True, linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Prediction(Ypred)")
plt.ylabel("Ytrue")
plt.show()


# # Artificial Neural Network

# In[ ]:


#Import Library
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential 
from keras.layers import Dense


# In[ ]:


def buildclassifier():
    classifier = Sequential() #initialize NN
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform',activation = 'tanh', input_dim =xtrain.shape[1]))
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform',activation = 'tanh'))
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform',activation = 'relu'))
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform',activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform',activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
    return classifier


# In[ ]:


classifier = KerasClassifier(build_fn = buildclassifier, epochs = 200)
accuracies = cross_val_score(estimator = classifier, X = xtrain, y= ytrain, cv = 6)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))


# # ML Algorithms F1-Scores 

# In[ ]:


#F1-Score For Logistic Regression
from sklearn.metrics import f1_score
LRf1 = f1_score(ytrue, yprediciton1, average='weighted') 
LRf1


# In[ ]:


#K-NN
KNNf1= f1_score(ytrue, yprediciton2, average='weighted') 
KNNf1


# In[ ]:


#SVM
SVMf1=f1_score(ytrue, yprediciton3, average='weighted') 
SVMf1


# In[ ]:


#naive bayes
NBf1 = f1_score(ytrue, yprediciton4, average='weighted') 
NBf1


# In[ ]:


#Decision Tree
DTf1=f1_score(ytrue, yprediciton5, average='weighted') 
DTf1


# In[ ]:


#RandomForest
RFf1=f1_score(ytrue, yprediciton6, average='weighted') 
RFf1


# # Scatter Plot For Comparasion of ML Algorithms Prediciton Scores

# In[ ]:



scores=[LRscore,KNNscore,SVMscore,NBscore,DTCscore,RFCscore,mean]
AlgorthmsName=["Logistic Regression","K-NN","SVM","Naive Bayes","Decision Tree", "Random Forest","Artificial Neural Network"]

#create traces

trace1 = go.Scatter(
    x = AlgorthmsName,
    y= scores,
    name='Algortms Name',
    marker =dict(color='rgba(0,255,0,0.5)',
               line =dict(color='rgb(0,0,0)',width=2)),
                text=AlgorthmsName
)
data = [trace1]

layout = go.Layout(barmode = "group",
                  xaxis= dict(title= 'ML Algorithms',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Prediction Scores',ticklen= 5,zeroline= False))
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# # Scatter Plot For Comparasion of ML Algorithms Prediciton Scores (F1)

# In[ ]:


scoresf1=[LRf1,KNNf1,SVMf1,NBf1,DTf1,RFf1]
#create traces

trace1 = go.Scatter(
    x = AlgorthmsName,
    y= scoresf1,
    name='Algortms Name',
    marker =dict(color='rgba(225,126,0,0.5)',
               line =dict(color='rgb(0,0,0)',width=2)),
                text=AlgorthmsName
)
data = [trace1]

layout = go.Layout(barmode = "group", 
                  xaxis= dict(title= 'ML Algorithms',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Prediction Scores(F1)',ticklen= 5,zeroline= False))
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# # Conclusion
# 
# 1. Thank you for investigation my kernel.
# 2. I was comparison ML Classification Algorithms with the Pima Indians Diabetes Database.
# 3. I found the best result with Random Forest and SVM.
# 4. I expect your opinion and criticism.
# 
# # If you like this kernel, Please Upvote :) Thanks
# 
# <img src="https://media1.giphy.com/media/l0ExvuzJGJNZJZ47S/giphy.gif?cid=790b76115cc05331372f4d64593e8962" width="500px">
# 
# 
