#!/usr/bin/env python
# coding: utf-8

# <a id="0"></a> <br>
# **Notebook Content**
# 1. [Introduction](#1)
#     1. [Importing libraries](#2)
#     1. [Reading Data](#3)
#     1. [Null values](#4)
#     1. [Data types](#5)
# 1. [Feature Extracting](#6)
#     1. [Dropping variables](#7)
#     1. [Convertion](#8)
#     1. [Data visualization](#9)
# 1.  [Building Model](#10)
#     1. [Splitting Data](#11)
#     1. [Cross-validation](#12)
#     1. [KNNClassifier](#13)
# 1. [Evaluation](#14)
#     1. [Model prediction](#15)
#     1. [Report generation](#16)

# <a id="1"></a> 
# # 1-Introduction
# For this dataset I am using K-Nearest Neighbors Algorithm. The dataset contains 33 columns/features. The target variable is 'diagnosis'. You can find more info on dataset from the Data tab.

# <a id="2"></a> 
# ## A-Importing libraries
# I am using pandas to manipulate data and matplotlib and seaborn for the data visualization

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# <a id="3"></a>
# ## B-Reading Data

# In[ ]:


bc = pd.read_csv(r'../input/data.csv')
bc.head(10)


# <a id="4"></a> 
# ## C-Null values
# Checking for missing values.  

# In[ ]:


bc.isnull().sum()


# <a id="5"></a> 
# ## D-Data types

# In[ ]:


bc.dtypes


# <a id="6"></a> 
# # 2- Feature Extracting

# <a id="7"></a> 
# ## A-Dropping variables
# By looking at the dataset, we can see that variable 'id' and variable 'Unnamed: 32' will have no impact on the model building. So we can drop the unneccessary variable to prevent data leakage.
# 

# In[ ]:


drop_cols = ['Unnamed: 32','id']
bc = bc.drop(drop_cols, axis = 1)


# In[ ]:


bc.shape


# <a id="8"></a> 
# ## B-Convertion
# For the model building the dataset need to be in a numerical format. The only variable that is not in numerical format is 'diagnosis'. Since it is a categorical variable, we can take 1 for M & 0 for B

# In[ ]:


bc['diagnosis'] = bc['diagnosis'].map({'M': 1, 'B': 0})
bc.head()


# In[ ]:


bc['diagnosis'].value_counts()


# <a id="9"></a> 
# ## C-Data visualization

# In[ ]:


# plotting the labels with the frequency 
Labels = ['Benign','Malignant']
classes = pd.value_counts(bc['diagnosis'], sort = True)
classes.plot(kind = 'bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), Labels)
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[ ]:


# Plotting the features with each other.
groups = bc.groupby('diagnosis')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.perimeter_mean, group.texture_mean, marker='o', ms=3.5, linestyle='', 
            label = 'Malignant' if name == 1 else 'Benign')
ax.legend()
plt.xlabel("perimeter_mean")
plt.ylabel("texture_mean")
plt.show()    


# In[ ]:


groups = bc.groupby('diagnosis')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.radius_mean, group.texture_mean, marker='o', ms=3.5, linestyle='', 
            label = 'Malignant' if name == 1 else 'Benign')
ax.legend()
plt.ylabel("texture_mean")
plt.xlabel("radius_mean")
plt.show()    


# In[ ]:


groups = bc.groupby('diagnosis')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.area_mean, group.texture_mean, marker='o', ms=3.5, linestyle='', 
            label = 'Malignant' if name == 1 else 'Benign')
ax.legend()
plt.ylabel("texture_mean")
plt.xlabel("area_mean")
plt.show()    


# In[ ]:


groups = bc.groupby('diagnosis')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.smoothness_mean, group.compactness_mean, marker='o', ms=3.5, linestyle='', 
            label = 'Malignant' if name == 1 else 'Benign')
ax.legend()
plt.ylabel("compactness_mean")
plt.xlabel("smoothness_mean")
plt.show()   


# Draws  heatmap with input as the correlation matrix calculted by(bc.corr()) using seaborn

# In[ ]:


import seaborn as sns
plt.figure(figsize=(12,12)) 
sns.heatmap(bc.corr(),annot=True,cmap='cubehelix_r') 
plt.show()


# <a id="10"></a> 
# # 3-Building Model

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


# <a id="11"></a> 
# ## A-Splitting Data
# 

# In[ ]:


bc_labels = pd.DataFrame(bc['diagnosis'])
bc_features = bc.drop(['diagnosis'], axis = 1)


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(bc_features,bc_labels,test_size=0.20)


# <a id="12"></a>
# ## B-Cross-validation
# Trying to find the optimal number of  neighbors with the lowest Misclassification error. 

# In[ ]:


#Performing cross validation
neighbors = []
cv_scores = []
from sklearn.model_selection import cross_val_score
#perform 10 fold cross validation
for k in range(1,10):
    neighbors.append(k)
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn,X_train,y_train,cv=10, scoring = 'accuracy')
    cv_scores.append(scores.mean())
    


# In[ ]:


#Misclassification error versus k
MSE = [1-x for x in cv_scores]

#determining the best k
optimal_k = neighbors[MSE.index(min(MSE))]
print('The optimal number of neighbors is %d ' %optimal_k)

#plot misclassification error versus k

plt.figure(figsize = (10,6))
plt.plot(neighbors, MSE)
plt.xlabel('Number of neighbors')
plt.ylabel('Misclassification Error')
plt.show()


# <a id="13"></a> 
# ## C-KNNClassifier

# In[ ]:


knn=KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train,y_train)


# <a id="14"></a> 
# # 4-Evaluation

# <a id="15"></a> 
# ## A-Model Prediction
# 

# In[ ]:


from sklearn.metrics import classification_report
y_pred = knn.predict(X_test)


# <a id="15"></a> 
# ## B-Report generation
# 

# In[ ]:


# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))

# Accuracy score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[ ]:


#calculating confusion matrix for knn
tn,fp,fn,tp=confusion_matrix(y_test,y_pred).ravel()


# In[ ]:


print("K-Nearest Neighbours")
print("Confusion Matrix")
print("tn =",tn,"fp =",fp)
print("fn =",fn,"tp =",tp)


# In[ ]:


Labels = ['Benign','Malignant']
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=Labels, yticklabels=Labels, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

