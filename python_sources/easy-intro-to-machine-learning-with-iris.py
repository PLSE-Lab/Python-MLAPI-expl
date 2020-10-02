#!/usr/bin/env python
# coding: utf-8

# ## Hello ! This is my first kernel in Kaggle using IRIS dataset.
# 
# This notebook aims to give a brief introduction to machine learning  while touching on the importance of selecting the right features when training your models. 
# 
# If you like this notebook please make sure to UPVOTE!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


iris_df = pd.read_csv("../input/Iris.csv") # load iris data into a dataframe


# In[ ]:


iris_df.head(5) # Let's take peak on the first 5 rows of iris dataframe


# In[ ]:


#For the purpose of this notebook we don't need the ID column, so let's drop it

iris_df.drop("Id", axis=1, inplace=True)


# In[ ]:


#Plot histograms for the different features of the flowers
plt.figure(figsize=(15,8)) 
iris_df.hist(edgecolor='black', linewidth=1.2)
plt.show()


# In[ ]:


from sklearn.linear_model import LogisticRegression  #It is a regression model used to analyse data which the output is a binary 
from sklearn.cross_validation import train_test_split #used to split the dataset into training and test data
from sklearn.neighbors import KNeighborsClassifier  # KNN is a model normally used to infer a prediction based on similarities
from sklearn.tree import DecisionTreeClassifier #It is a model used for classification, as the name suggests it constructs a tree based on the features of the dataset to infer a result
from sklearn import metrics #for checking the model accuracy


# ### 6 Simple Steps
# 1. Before we use a model we need to prepare the data, for example in this notebook, we removed the column "Id" because we don't use it but in certain datasets you might need to do more than that, such a fillna or clean up additional features
# 2. Once the data is preprocessed, you need to split the data into test and train data. Train data is used to train the model, while Test data is used to validate the model
# 3. Choose a model (LogisticRegression, KNN, DecisionTreeClassifier)
# 4 Fit the model using .fit with the training data
# 5 Validate the model using predict()
# 6 Check the accuracy of your model and repeat the above steps to improve the accuracy 

# In[ ]:


train, test = train_test_split(iris_df, test_size = 0.2)# Splitting the data into training and test data in this case it split 80% to train data and 20% to test data 
print(train.shape)
print(test.shape)


# In[ ]:


train_X = train[['PetalLengthCm','PetalWidthCm', 'SepalLengthCm','SepalWidthCm']]# Select flower features on train data
train_y=train.Species# print the output of train data
test_X= test[['PetalLengthCm','PetalWidthCm', 'SepalLengthCm','SepalWidthCm']] #  Select flower features on test data
test_y =test.Species   #print the output of test data


# In[ ]:


### Logistic Regression
model_lr = LogisticRegression()
model_lr.fit(train_X,train_y)
prediction_lr=model_lr.predict(test_X)
print('Model {Logistic Regression} Accuracy:',metrics.accuracy_score(prediction_lr,test_y))


# In[ ]:


### Decision Tree
model_dtc=DecisionTreeClassifier()
momodel_dtc=model_dtc.fit(train_X,train_y)
prediction_dtc=model_dtc.predict(test_X)
print('Model {Decision Tree} accuracy:',metrics.accuracy_score(prediction_dtc,test_y))


# In[ ]:


###KNN
model_knn=KNeighborsClassifier(n_neighbors=3) #Searches in the 3 closest neighbours
model_knn.fit(train_X,train_y)
prediction_knn=model_knn.predict(test_X)
print(prediction_knn)
print('Model {KNN} accuracy:',metrics.accuracy_score(prediction_knn,test_y))


# ###**Summary**
# You can see how different models perform to the features we used to train the model. One thing you should know, is that the IRIS data is a great dataset because it provides you with the different features of given species of flower. It also show that when you use too many features for your model it can create some unforeseen noise.
# 
# For example if you use only Petal's attribute the accuracy will improve due to the correlation of petals width and length.
# 
# 

# In[ ]:





# In[ ]:


train_petals_X = train[['PetalLengthCm','PetalWidthCm']]# Select flower features on train data
train_petals_y=train.Species# print the output of train data
test_petals_X= test[['PetalLengthCm','PetalWidthCm']] #  Select flower features on test data
test_petals_y =test.Species   #print the output of test data


# In[ ]:


### Logistic Regression (Petals feature Only)
model_petals = LogisticRegression()
model_petals.fit(train_petals_X,train_petals_y)
prediction_petals=model_petals.predict(test_petals_X)
print('Change in accuracy : %f' % (metrics.accuracy_score(prediction_petals,test_petals_y) - metrics.accuracy_score(prediction_lr,test_y)))


# In[ ]:


### DecisionTree Classifier (Petals feature Only)
model_petals = DecisionTreeClassifier()
model_petals.fit(train_petals_X,train_petals_y)
prediction_petals=model_petals.predict(test_petals_X)
print('Change in accuracy : %f' % (metrics.accuracy_score(prediction_petals,test_petals_y) - metrics.accuracy_score(prediction_dtc,test_y)))


# In[ ]:


#KNN (Petals feature Only)
model_petals=KNeighborsClassifier(n_neighbors=3)
model_petals.fit(train_petals_X,train_petals_y)
prediction_petals=model_petals.predict(test_petals_X)
print('Change in accuracy : %f' % (metrics.accuracy_score(prediction_petals,test_petals_y) - metrics.accuracy_score(prediction_knn,test_y)))


# Hopefully this was helpful, if you have any feedbacks please let me know!
# Also don't forget to upvote in case you liked this notebook :)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




