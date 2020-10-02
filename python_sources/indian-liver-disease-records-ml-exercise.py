#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
#     In this kernel I will practice machine learning with liver disease dataset.
# 
# 1. [A Quick Look on the Data](#1)
# 1. [Getting Data Ready](#2)
# 1. [Data Description](#3)
# 1. [Visualisation](#4)
#     * [Correlation Between Columns](#5)
#     * [Correlation Between Bilirubin Levels and Liver Disease](#6)
#     * [Correlation Between Enzymes and Liver Disease](#7)
#     * [Correlation Between Albumin-Globulin Ratio and Liver Disease](#8)
#     * [Gender and Disease](#9)
#     * [Age and Disease](#10)
#     * [Albumin-Globulin Ratio and Disease](#11)
#     * [Bilirubin Levels and Disease](#12)
#     * [Enzymes and Disease](#13)
#     * [Protein Levels and Disease](#14)
# 1. [Machine Learning](#15)
#     * [Normalisation](#16)
#     * [Prediction with Logistic Regression](#17)
#         * [Defining Functions](#18)
#             * [Initialising](#19)
#             * [Sigmoid Funtion](#20)
#             * [Other Functions](#21)
#         * [Logistic Regression With Library](#22)
#     * [Prediction with KNN](#23)
#     * [Prediction with SVM](#24)
#     * [Prediction with Naive Bayes](#25)
#     * [Prediction with Decision Tree](#26)
#     * [Prediction with Random Forest Regression](#27)
#  
# 1. [Comparision of Prediction Scores](#28)
# 1. [Result](#29)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv("/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv")


# <a id="1"></a>
# ## A Quick Look on the Data

# In[ ]:


data.head()


# In[ ]:


data.info()


# <a id="2"></a> 
# ## Getting Data Ready

# In[ ]:


#Renaming the columns.

data.rename(columns={'Dataset': 'target', 'Alamine_Aminotransferase': 'Alanine_Aminotransferase', 'Total_Protiens': 'Total_Proteins'}, inplace = True)


# In[ ]:


data.target.unique()


# In[ ]:


data.target = [0 if each == 2 else 1 for each in data.target]


# In[ ]:


#Data contains object variables, I want integers or float variables.

data.Gender = [1 if each == 'Male' else 0 for each in data.Gender]


# In[ ]:


data.dtypes


# In[ ]:


data.isna().sum()


# In[ ]:


#Filling null values.
data['Albumin_and_Globulin_Ratio'].mean()


# In[ ]:


data.fillna(0.94, inplace = True)


# In[ ]:


data.info()


# <a id="3"></a>
# ## Data Description
# 
# * Age: Age of the patients
# * Gender: Sex of the patients (1 for male and 0 for female)
# * Total_Bilirubin: Total Billirubin in mg/dL
# * Direct_Bilirubin: Conjugated Billirubin in mg/dL
# * Alkaline_Phosphotase: ALP in IU/L (an enzyme)
# * Alanine_Aminotransferase: ALT in IU/L (an enzyme)
# * Aspartate_Aminotransferase: AST in IU/L (an enzyme)
# * Total_Protiens: Total Proteins g/dL
# * Albumin: Albumin in g/dL
# * Albumin_and_Globulin_Ratio: A/G ratio
# * target: patient has liver disease or not (1 for having the disease and 0 for not having)

# <a id="4"></a>
# # Visualisation

# In order to understand the correlations between the columns better, I am going to visualise the data.

# <a id="5"></a>
# ## Correlation Between Columns

# In[ ]:


correlation = data.corr()
fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(correlation, annot = True, linewidths = 0.5, ax = ax)
plt.show()


# <a id="6"></a>
# ## Correlation Between Bilirubin Levels and Liver Disease

# In[ ]:


list_ = ["Age", "Total_Bilirubin", "Direct_Bilirubin", "target"]

sns.heatmap(data[list_].corr(), annot = True, fmt = ".2f")
plt.show()


# <a id="7"></a>
# ## Correlation Between Enzymes and Liver Disease

# In[ ]:


list2 = ["Alkaline_Phosphotase", "Alanine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Proteins", "target"]
sns.heatmap(data[list2].corr(), annot = True, fmt = ".2f")
plt.show()


# <a id="8"></a>
# ## Correlation Between Albumin - Globulin Levels and Liver Disease

# In[ ]:


list3 = ["Albumin_and_Globulin_Ratio", "Albumin", "target"]
sns.heatmap(data[list3].corr(), annot = True, fmt = ".2f")
plt.show()


# <a id="9"></a>
# ## Gender and Disease

# In[ ]:


f, axes = plt.subplots(1, 2, figsize = (12, 8))

sns.countplot(x = "target", data = data, ax=axes[0])
sns.countplot(x = "target", hue = 'Gender', data = data, ax=axes[1])
plt.show()


# In[ ]:


print("Number of people who suffers from liver disease: {}" .format(data.target.value_counts()[1]))
print("Number of people who does not suffer from liver disease: {}" .format(data.target.value_counts()[0]))


# *Maen are more likely to have a liver disease. Nevertheless, men are the majority among the healthy poeple.*

# <a id="10"></a>
# ## Age - Disease

# In[ ]:


g = sns.FacetGrid(data, col = "target", height = 7)
g.map(sns.distplot, "Age", bins = 25)
plt.show()


# Two graphs are similar, I don't think age feature will be useful in prediction.

# <a id="11"></a>
# ## Albumin-Globulin Ratio and Disease

# In[ ]:


#I want to add a column showing globulin values.

ratio = data.Albumin_and_Globulin_Ratio.values
albumin = data.Albumin.values
globulin = []
for i in range(0, 583):
    globulin.append(float("{:.2f}".format(albumin[i] / ratio [i])))


# In[ ]:


data.insert(9, 'Globulin', globulin, True)


# I added a globulin column to the data.

# In[ ]:


data.head()


# In[ ]:


g = sns.FacetGrid(data, col = "target", row = "Gender", height = 5)
g.map(plt.hist, "Albumin", bins = 25)
plt.show()


# In[ ]:


g = sns.FacetGrid(data, col = "target", row = "Gender", height = 5)
g.map(plt.hist, "Globulin", bins = 25)
plt.show()


# In[ ]:


g = sns.FacetGrid(data, col = "target", row = "Gender", height = 5)
g.map(plt.hist, "Albumin_and_Globulin_Ratio", bins = 25)
plt.show()


# Apparently, both albumin and globulin levels are higher in patients.

# <a id="12"></a>
# ## Bilirubin Level and Disease

# In[ ]:


patient = data[data.target == 1]
healthy = data[data.target != 1]

trace0 = go.Scatter(
    x = patient['Total_Bilirubin'],
    y = patient['Direct_Bilirubin'],
    name = 'Patient',
    mode = 'markers', 
    marker = dict(color = '#616ADE',
        line = dict(
            width = 1)))

trace1 = go.Scatter(
    x = healthy['Total_Bilirubin'],
    y = healthy['Direct_Bilirubin'],
    name = 'healthy',
    mode = 'markers',
    marker = dict(color = '#F3EC1F',
        line = dict(
            width = 1)))

layout = dict(title = 'Total Bilirubin vs Conjugated Bilirubin',
              yaxis = dict(title = 'Conjugated Bilirubin', zeroline = False),
              xaxis = dict(title = 'Total Bilirubin', zeroline = False)
             )

data2 = [trace0, trace1]

fig = go.Figure(data=data2,
                layout=layout)

fig.show()


# Bilirubin levels in patients are higher.

# <a id="13"></a>
# ## Enzymes and Disease

# Alkaline phosphotase, alaine aminotransferase and aspartate aminotransferase are enzymes mainly found in liver.

# In[ ]:


g = sns.FacetGrid(data, col = "target", height = 7)
g.map(sns.distplot, "Alkaline_Phosphotase", bins = 25)
plt.show()


# In[ ]:


g = sns.FacetGrid(data, col = "target", height = 7)
g.map(sns.distplot, "Alanine_Aminotransferase", bins = 25)
plt.show()


# In[ ]:


g = sns.FacetGrid(data, col = "target", height = 7)
g.map(sns.distplot, "Aspartate_Aminotransferase", bins = 25)
plt.show()


# All of the three enzymes are lower in patients.

# <a id="14"></a>
# ## Proteins and Disease

# In[ ]:


g = sns.FacetGrid(data, col = "target", height = 7)
g.map(sns.distplot, "Total_Proteins", bins = 25)
plt.show()

print("Mean of the total protein level in patiens:", float("{:.2f}".format( data['Total_Proteins'][data.target == 1].mean())))
print("Mean of the total protein level in healthy people:", float("{:.2f}".format(data['Total_Proteins'][data.target == 0].mean())))


# Protein levels aren't very distinctive between patients and non-patients.

# <a id="15"></a>
# # Machine Learning

# In[ ]:


#Importing necessary libraries

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[ ]:


#I want to store the scores in a dictionary to see which prediction method works best.

scores = {}


# <a id="16"></a>
# ## Normalisation

# First, I need to create arrays x and y for training and testing.

# In[ ]:


data = data.drop(columns = ['Total_Proteins', 'Age', 'Gender'])


# In[ ]:


y = data.target.values
x_ = data.drop(columns = ["target"])


# In[ ]:


#Normalisation
x = ((x_ - np.min(x_)) / (np.max(x_) - np.min(x_))).values


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[ ]:


x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T


# I chose weight as 0.01 and bias as 0.00

# <a id="17"></a>
# ## Prediction with Logistic Regression

# <a id="18"></a>
# ### Defining Functions

# Before using the library I will define functions for predictions.

# <a id="19"></a>
# ### Initialising

# In[ ]:


def initialise(dimension):
    
    weight = np.full((dimension,1),0.01)
    bias = 0.0
    return weight,bias


# <a id="20"></a>
# ### Sigmoid Function

# In[ ]:


def sigmoid(z):
    y_head = 1/(1 + np.exp(-z))
    return y_head


# <a id="21"></a>
# ### Defining Other Functions

# In[ ]:


def forward_backward(weight, bias, x_train, y_train):
    z = np.dot(weight.T, x_train) + bias
    y_head = sigmoid(z)
    loss = -((y_train * np.log(y_head)) + ((1 - y_train) * np.log(1 - y_head)))
    cost = (np.sum(loss))/x_train.shape[1]
    
    derivative_weight = (np.dot(x_train,((y_head - y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head - y_train)/x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    
    return cost, gradients

def update(weight, bias, x_train, y_train, learning_rate, number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    for i in range(number_of_iterarion):
        cost, gradients = forward_backward(weight, bias, x_train, y_train)
        cost_list.append(cost)
        
        weight = weight - learning_rate * gradients["derivative_weight"]
        bias = bias - learning_rate * gradients["derivative_bias"]
        
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
    parameters = {"weight": weight, "bias": bias}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel("Number of Iterarions")
    plt.ylabel("Cost")
    plt.show()
    
    return parameters, gradients, cost_list

def predict(weight, bias, x_test):
    z = sigmoid(np.dot(weight.T, x_test) + bias)
    Y_prediction = np.zeros((1, x_test.shape[1]))
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate,  num_iterations):
    dimension =  x_train.shape[0]
    weight, bias = initialise(dimension)
    parameters, gradients, cost_list = update(weight, bias, x_train, y_train, learning_rate, num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    
    print("test accuracy: {} %".format(float("{:.2f}".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))))
    scores['Logistic Regression with Functions'] = float("{:.2f}".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))


# In[ ]:


logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 0.01, num_iterations = 200)


# <a id="22"></a>
# ## Prediction with Library

# In[ ]:


lr = LogisticRegression()

lr.fit(x_train.T, y_train.T)
print("test accuracy = {}%" .format(float("{:.2f}".format(lr.score(x_test.T, y_test.T) * 100))))


# The result is the same. Accuracy score is around 75%.

# In[ ]:


scores['Logistic Regression Score'] = float("{:.2f}".format(lr.score(x_test.T, y_test) * 100))


# <a id="23"></a>
# ## Prediction with KNN

# In[ ]:


x_train = x_train.T
x_test = x_test.T


# In[ ]:


knn_scores = []
for each in range(1, 15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train, y_train)
    knn_scores.append(knn2.score(x_test, y_test))

plt.figure(figsize = (10, 8))
plt.plot(range(1, 15), knn_scores)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()


# With the k value 9, I have the best score.

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 9)
knn.fit(x_train, y_train)

prediction = knn.predict(x_test)


# In[ ]:


print("KNN score: {}" .format(float("{:.2f}".format(knn.score(x_test, y_test) * 100))))


# In[ ]:


scores['KNN Score'] = (float("{:.2f}".format(knn.score(x_test, y_test) * 100)))


# <a id = 24></a>
# ## Prediction with SVM

# In[ ]:


svm = SVC(random_state = 1)
svm.fit(x_train, y_train)


# In[ ]:


print("SVM Score is: {}" .format(float("{:.2f}".format(svm.score(x_test, y_test) * 100))))


# In[ ]:


scores['SVM Score'] = (float("{:.2f}".format(svm.score(x_test, y_test) * 100)))


# <a id = 25></a>
# 
# ## Prediction with Naive Bayes

# In[ ]:


nb = GaussianNB()
nb.fit(x_train, y_train)


# In[ ]:


print("Naive Bayes Score is: {}" .format(float("{:.2f}".format(nb.score(x_test, y_test) * 100))))


# In[ ]:


scores['Naive Bayes Score'] = (float("{:.2f}".format(nb.score(x_test, y_test) * 100)))


# <a id = 26></a>
# ## Prediction with Decision Tree

# In[ ]:


dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)


# In[ ]:


print("Decision Tree Score is: {}" .format(float("{:.2f}".format(dt.score(x_test, y_test) * 100))))


# In[ ]:


scores['Decision Tree Score'] = (float("{:.2f}".format(dt.score(x_test, y_test) * 100)))


# <a id = 27></a>
# ## Prediction with Random Forest

# In[ ]:


rf = RandomForestClassifier(n_estimators = 100, random_state=1)
rf.fit(x_train, y_train)


# In[ ]:


print("Random Forest Score is: {}" .format(float("{:.2f}".format(rf.score(x_test, y_test) * 100))))


# In[ ]:


scores['Random Forest Score'] = (float("{:.2f}".format(rf.score(x_test, y_test) * 100)))


# <a id="28"></a>
# 
# # Comparision of the Prediction Results

# In[ ]:


lists = sorted(scores.items())

x_axis, y_axis = zip(*lists)

plt.figure(figsize = (15, 10))
plt.plot(x_axis, y_axis)
plt.show()


# I get the best score with knn, but even that is not enough.

# <a id="29"></a>
# 
# # Result

# The best score I could get is 75%
