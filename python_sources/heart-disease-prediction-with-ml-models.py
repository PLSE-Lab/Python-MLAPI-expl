#!/usr/bin/env python
# coding: utf-8

# **Introduction
# **
# 
# Hello, I'm going to preciton with Logistic regression KNN Classification SVM Decision Tree and Random Forest. I will use Heart Disease UCI dataset. If you wonder anything about dateset,you can read here. (https://www.kaggle.com/ronitf/heart-disease-uci)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # ML Classification
# 
# ----Content
# 
# 1-Import Dataset
# 
# 2-Investigation Dataset
# 
# 3-Visualizaition Dataset
# 
# 4-ML Algoritms importation and predicition
# 
# 5-Visualizaition of results
# 
# 6-Conclusion

# In[ ]:


#read data
data = pd.read_csv("../input/heart.csv")


# In[ ]:


#data.info()


# In[ ]:


#data.tail()


# In[ ]:


#Split Data as M&B
A = data[data.target == 1]
B = data[data.target == 0]


# The "goal" field refers to the presence of heart disease in the patient.
# Target = 1 => presence of heart disease
# Target = 0 => no of heart disease

# In[ ]:


#Visualization, Scatter Plot

plt.scatter(A.chol,A.age,color = "Black",label="1",alpha=0.4)
plt.scatter(B.chol,B.age,color = "Orange",label="0",alpha=0.4)
plt.xlabel("Cholesterol")
plt.ylabel("Age")
plt.legend()
plt.show()

#We appear that it is clear segregation.


# In[ ]:


#Visualization, Scatter Plot

plt.scatter(A.trestbps,A.age,color = "Black",label=" 1",alpha=0.3)
plt.scatter(B.trestbps,B.age,color = "Lime",label="0",alpha=0.3)
plt.xlabel("Resting Blood Pressure ")
plt.ylabel("Age")
plt.legend()
plt.show()


# In[ ]:


#Visualization, Scatter Plot

plt.scatter(A.trestbps,A.chol,color = "Black",label="1",alpha=0.3)
plt.scatter(B.trestbps,B.chol,color = "red",label="0",alpha=0.3)
plt.xlabel("Resting Blood Pressure ")
plt.ylabel("Cholesterol")
plt.legend()
plt.show()


# In[ ]:


#Seperate data
y =data.target.values
x1=data.drop(["target"],axis=1)


# In[ ]:


#Normalization 
x = (x1 - np.min(x1))/(np.max(x1)-np.min(x1)).values


# In[ ]:


#Split For Train and Test
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=42)


# In[ ]:


#transposition
xtrain = xtrain.T
xtest = xtest.T
ytrain = ytrain.T
ytest = ytest.T


# # Logistic Regression Prediction

# In[ ]:


#LR with sklearn
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xtrain.T,ytrain.T)
print("Test Accuracy {}".format(LR.score(xtest.T,ytest.T))) 
LRscore =LR.score(xtest.T,ytest.T)


# In[ ]:


#Confusion Matrix

yprediciton1= LR.predict(xtest.T)
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


# # KNN Prediction

# In[ ]:


#Create-KNN-model
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


#Find Optimum K value
scores = []
for each in range(1,50):
    KNNfind = KNeighborsClassifier(n_neighbors = each)
    KNNfind.fit(xtrain.T,ytrain.T)
    scores.append(KNNfind.score(xtest.T,ytest.T))
    
plt.plot(range(1,50),scores,color="black")
plt.xlabel("K Values")
plt.ylabel("Score(Accuracy)")
plt.show()


# In[ ]:


KNNfind = KNeighborsClassifier(n_neighbors = 24) #n_neighbors = K value
KNNfind.fit(xtrain.T,ytrain.T) #learning model
prediction = KNNfind.predict(xtest.T)
print("{}-NN Score: {}".format(25,KNNfind.score(xtest.T,ytest.T)))
KNNscore = KNNfind.score(xtest.T,ytest.T)


# In[ ]:


#Confusion Matrix

yprediciton2= KNNfind.predict(xtest.T)
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


# # SVM Prediction

# In[ ]:


#SVM with Sklearn

from sklearn.svm import SVC

SVM = SVC(random_state=42)
SVM.fit(xtrain.T,ytrain.T)  #learning 
#SVM Test 
print ("SVM Accuracy:", SVM.score(xtest.T,ytest.T))

SVMscore = SVM.score(xtest.T,ytest.T)


# # DT Prediction

# In[ ]:


#Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(random_state=2)
DTC.fit(xtrain.T,ytrain.T) #learning
#prediciton
print("Decision Tree Score: ",DTC.score(xtest.T,ytest.T))
DTCscore = DTC.score(xtest.T,ytest.T)


# # Random Forest Prediction

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
#Find Optimum K value
scores = []
for each in range(1,50):
    RFfind = RandomForestClassifier(n_estimators = each,random_state=5)
    RFfind.fit(xtrain.T,ytrain.T)
    scores.append(RFfind.score(xtest.T,ytest.T))
    
plt.figure(1, figsize=(10, 5))
plt.plot(range(1,50),scores,color="black",linewidth=2)
plt.title("Optimum N Estimator Value")
plt.xlabel("N Estimators")
plt.ylabel("Score(Accuracy)")
plt.grid(True)
plt.show()


# In[ ]:


RFfind= RandomForestClassifier(n_estimators = 24, random_state=5) #n_estimator = DT
RFfind.fit(xtrain.T,ytrain.T) # learning
print("Random Forest Score: ",RFfind.score(xtest.T,ytest.T))
RFCscore=RFfind.score(xtest.T,ytest.T)


# In[ ]:


#Confusion Matrix

yprediciton2= RFfind.predict(xtest.T)
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


# ### Artifical Neural Network

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


#Split For Train and Test
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.15, random_state=42)


# In[ ]:


classifier = KerasClassifier(build_fn = buildclassifier, epochs = 800)
accuracies = cross_val_score(estimator = classifier, X = xtrain, y= ytrain, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))


# In[ ]:


scores=[LRscore,KNNscore,SVMscore,DTCscore,RFCscore,mean]
AlgorthmsName=["Logistic Regression","K-NN","SVM","Decision Tree", "Random Forest",'Artificial Neural Network']

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


# # Conclusion
# 
# 1- Thank you for investigation my kernel.
# 
# 2- I compared ML algorithms with Heart Disease Dataset.
# 
# 3- I found optimum value by aid of for loop.
# 
# 4- Finally, I obtained the same score values.
# 
# 5- I expect your opinion and criticism.
# # If you like this kernel, Please Upvote :) Thanks
# 
# <img src="https://media.giphy.com/media/1oF1KAEYvmXBMo6uTS/giphy.gif" width="500px">
