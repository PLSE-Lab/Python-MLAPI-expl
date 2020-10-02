#!/usr/bin/env python
# coding: utf-8

# # K-Nearest Neighbour using SciKit-learn

# ### Importing required libraries

# In[ ]:


import pandas as pd
import seaborn as sns
sns.set() #Apply the default seaborn theme


# In[ ]:


dataset = pd.read_csv('../input/heart-disease-uci/heart.csv') #Loading the dataset


# ### Understanding the dataset

# In[ ]:


dataset.head()


# In[ ]:


type(dataset)


# In[ ]:


dataset.shape


# In[ ]:


dataset.dtypes


# In[ ]:


dataset.describe()


# In[ ]:


dataset.columns


# In[ ]:


dataset.info()


# ### Exploratory Data Analysis

# ##### Sex distribution (0: Female, 1: Male)

# In[ ]:


sns.countplot(dataset['sex'])


# As we can see from the graph, there are more number of males than females in this dataset.

# ##### Chest pain types

# In[ ]:


sns.countplot(dataset['cp'])


# There are total 4 types of chest pain: Typical Anigma, Atypical Anigma, Non-angical pain and asymptomatic pain. 
# As we can see from the graph that typical anigma is the one which occurs the most in human beings. 
# That is quite informative but we need more information about it!

# In[ ]:


sns.countplot(dataset['cp'], hue='sex', data = dataset)


# This is much more informative than the previous one!

# ##### Target (Disease found or not)

# In[ ]:


sns.countplot(dataset['target'], hue='sex', data = dataset)


# As we are looking from this dataset is that males have a quite more probablity of having a heart attack than females.

# In[ ]:


sns.countplot(dataset['cp'], hue='target', data = dataset)


# To be quite more informative, we can see that those who're having non-angical type of pain appears to be having a heart disease more than anybody else. 

# #### NA values

# In[ ]:


dataset.isna().sum()


# There are no NA values in the dataset. But is there any better way to look for it?

# In[ ]:


sns.heatmap(dataset.isnull(),cbar = False)


# The graph tells us in detail that there are no NA values present in the dataset at all.

# #### Age distribution

# In[ ]:


sns.countplot(dataset['age'])


# We can see that the count plot doesn't help us in understanding much since age is a type of continuous variable, we'll use histogram or say distplot to understand the distribution of the dataset.

# In[ ]:


sns.distplot(dataset['age'])


# #### Corelation between variables

# In[ ]:


dataset.corr()


# There's another way to look at the correlations:

# In[ ]:


sns.heatmap(dataset.corr(), annot = True)


# ### Splitting our dataset to create the model (Train and Test dataset)

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = dataset.drop('target', axis = 1)
y = dataset['target']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 4)


# ### Importing library to create our KNN model

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


# Creating the model
model = KNeighborsClassifier(n_neighbors=1) #Let's try for one neighbour


# In[ ]:


model.fit(X_train, y_train) # Fitting the dataset into the model


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


# To see the confusion matrix
from sklearn.metrics import confusion_matrix


# In[ ]:


# This is the confusion matrix of the predicted data and the actual data
confusion_matrix(y_test, y_pred)


# In[ ]:


# To see the accuracy of the model
from sklearn.metrics import accuracy_score


# In[ ]:


# Accuracy of the model when the total neighbours were one
accuracy_score(y_test,y_pred)


# In[ ]:


# Now let's what will be the accuracy when we'll be having 2 nearest neighbours
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:


# For 3 now,
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:


# Now let's check that what is the number of neighbours that fit our model, best.
accuracy = []
for i in range(1,50):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy.append(accuracy_score(y_test,y_pred))


# In[ ]:


accuracy


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.plot(range(1,50), accuracy, marker='o')


# In[ ]:


# This is the highest accuracy we are getting of our model
max(accuracy)


# In[ ]:


accuracy.index(max(accuracy))


# In[ ]:


import numpy as np


# In[ ]:


error_rate = []
for i in range(1,50):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    error_rate.append(np.mean(y_test != y_pred))


# In[ ]:


plt.plot(range(1,50), error_rate, marker='o')


# **Improving our Accuracy**

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[ ]:


X = dataset.drop('target', axis = 1)
y = dataset['target']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 4)
X_train_sc = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)  


# In[ ]:


y_train = y_train.values
y_train_sc = sc.fit_transform(y_train.reshape(-1,1)) #Reshaping it, otherwise we'll get a 1-D array and it'll also give us an error
y_train_sc_flatten = y_train_sc.flatten() #Flattening it
y_test = y_test.values
y_test_sc = sc.fit_transform(y_test.reshape(-1,1))
y_test_sc_flatten = y_test_sc.flatten()


# You are passing floats to a classifier which expects categorical values as the target vector. If you convert it to int it will be accepted as input (although it will be questionable if that's the right way to do it). It would be better to convert your training scores by using scikit's labelEncoder function. 

# In[ ]:


from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
y_train_sc_flatten_encoded = lab_enc.fit_transform(y_train_sc_flatten)
y_test_sc_flatten_encoded = lab_enc.fit_transform(y_test_sc_flatten)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


accuracy_improved = []
for i in range(1,50):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train_sc, y_train_sc_flatten_encoded)
    y_pred = model.predict(X_test)
    accuracy_improved.append(accuracy_score(y_test_sc_flatten_encoded,y_pred))


# In[ ]:


accuracy_improved


# In[ ]:


max(accuracy_improved)

