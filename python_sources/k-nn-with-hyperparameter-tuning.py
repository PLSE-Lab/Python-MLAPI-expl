#!/usr/bin/env python
# coding: utf-8

# In this kernel, we will try to classify the Iris species using the k-Nearest Neighbors algorithm. We will also find the best parameters for the model using hyperparameter tuning. 

# # Import the libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # for data visualiztions

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# read the dataset
df = pd.read_csv('/kaggle/input/iris/Iris.csv')
df.head()


# Before we move on to classification, let us see some basic information about our dataset.

# In[ ]:


# get the dimensions of the dataset
df.shape


# In[ ]:


# find the data types of the attributes
df.dtypes


# In[ ]:


# concise summary of the data
df.describe()


# In[ ]:


# find if missing values is present
df.isnull().sum()


# No missing values- our dataset is clean

# In[ ]:


# get the distribution of the target variable
sns.countplot(x="Species", data = df)


# All the species have equal division which is good since all features will have equal influence on predciting the species.

# In[ ]:


# Separate the dependent and independent features
X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df['Species']


# In[ ]:


# Standardize the data and then concatenate it with y
data = X
data_std = (data - data.mean())/(data.max() - data.min())
data = pd.concat([data_std,y], axis=1)


# In[ ]:


# reshape the dataframe using melt()
data = pd.melt(data, id_vars = 'Species', var_name = 'features',value_name = 'value')
data


# By reshaping our data using melt(), we have converted it into a more computer friendly form, where initially we had more than one identifier features. Now, all of them have been packed into one feature which will help us in visualisation.
# 

# In[ ]:


# swarmplot for analysing the different attributes
plt.figure(figsize = (6,6))
sns.swarmplot(x = 'features', y = 'value', hue = 'Species', data = data)
plt.show()


# It is clear from the plot that Petal Length and Petal Width provide very clear distinctions between the different classes whereas the same cannot be said for Sepal Length and Sepal Width 

# In[ ]:


# obtain a correlation heatmap
sns.heatmap(X.corr(), annot=True)


# # Feature Selection

# Feature Selection is a techinque of finding out the features that contribute the most to our model i.e. the best predictors.

# In[ ]:


# split the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
print('Training set shape: ', X_train.shape, y_train.shape)
print('Testing set shape: ', X_test.shape, y_test.shape)


# In[ ]:


from sklearn.feature_selection import chi2, SelectKBest, f_classif


# In[ ]:


# Get the two best(k = 2) features using the SelectKBest method
ft = SelectKBest(chi2, k = 2).fit(X_train, y_train)
print('Score: ', ft.scores_)
print('Columns: ', X_train.columns)


# In[ ]:


ft = SelectKBest(f_classif, k= 2).fit(X_train, y_train)
print('Score: ', ft.scores_)
print('Columns: ', X_train.columns)


# We can now confirm our results and use only Petal Length and Petal Width for prediction

# # Preprocessing

# In[ ]:


X_train_2 = ft.transform(X_train)
X_test_2 = ft.transform(X_test)


# In[ ]:


from sklearn import preprocessing
X_train = preprocessing.StandardScaler().fit(X_train_2).transform(X_train_2.astype(float))
X_test = preprocessing.StandardScaler().fit(X_test_2).transform(X_test_2.astype(float))


# # k-Nearest Neighbors

# k- Nearest Neighbors is one of the most basic algorithms used in supervised machine learning. It classifies new data points based on similarity index which is usually a distance metric. It uses a majority vote will classifying the new data. For example, if there are 3 blue dots and 1 dot near the new data point, it will classify it as a blue dot.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn


# In[ ]:


from sklearn import metrics


# One of the challenges in a k-NN algorithm is finding the best 'k' i.e. the number of neighbors to be used in the majority vote while deciding the class. Generally, it is advisable to test the accuracy of your model for different values of k and then select the best one from them.

# In[ ]:


# calculating the accuracy of models with different values of k
mean_acc = np.zeros(20)
for i in range(1,21):
    #Train Model and Predict  
    knn = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    yhat= knn.predict(X_test)
    mean_acc[i-1] = metrics.accuracy_score(y_test, yhat)

mean_acc


# In[ ]:


loc = np.arange(1,21,step=1.0)
plt.figure(figsize = (10, 6))
plt.plot(range(1,21), mean_acc)
plt.xticks(loc)
plt.xlabel('Number of Neighbors ')
plt.ylabel('Accuracy')
plt.show()


# There are a range of values from 5 to 16 where the accuracy is the highest.

# # Hyperparameter Tuning

# A hyperparameter is a parameter of the model that is set before the start of learning process. Different machine learning models have different hyperparameters. You can find out more about the different hyperparameters of k-NN <a href =  'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html'>here</a>.

# We will use the Exhaustive Grid Search technique for hyperparameter optimization. An exhaustive grid search takes in as many hyperparameters as you would like, and tries every single possible combination of the hyperparameters as well as as many cross-validations as you would like it to perform. An exhaustive grid search is a good way to determine the best hyperparameter values to use, but it can quickly become time consuming with every additional parameter value and cross-validation that you add.

# In[ ]:


from sklearn.model_selection import GridSearchCV


# We will use three hyperparamters- n-neighbors, weights and metric.
# 1. n_neighbors: Decide the best k based on the values we have computed earlier.
# 2. weights: Check whether adding weights to the data points is beneficial to the model or not. 'uniform' assigns no weight, while 'distance' weighs points by the inverse of their distances meaning nearer points will have more weight than the farther points.
# 3. metric: The distance metric to be used will calculating the similarity.

# In[ ]:


grid_params = { 'n_neighbors' : [5,7,9,11,13,15],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}


# In[ ]:


gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=3, n_jobs = -1)


# Since we have provided the class validation score as 3( cv= 3), Grid Search will evaluate the model 6 x 2 x 3 x 3 = 108 times with different hyperparameters.

# In[ ]:


# fit the model on our train set
g_res = gs.fit(X_train, y_train)


# In[ ]:


# find the best score
g_res.best_score_


# In[ ]:


# get the hyperparameters with the best score
g_res.best_params_


# In[ ]:


# use the best hyperparameters
knn = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform',algorithm = 'brute',metric = 'minkowski')
knn.fit(X_train, y_train)


# In[ ]:


# get a prediction
y_hat = knn.predict(X_train)
y_knn = knn.predict(X_test)


# # Model Evaluation

# In[ ]:


print('Training set accuracy: ', metrics.accuracy_score(y_train, y_hat))
print('Test set accuracy: ',metrics.accuracy_score(y_test, y_knn))


# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_knn))


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_knn))


# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, X, y, cv =5)


# In[ ]:


print('Model accuracy: ',np.mean(scores))


# As we see, we have obtained a very high model accuracy of 0.97. It is possible that the accuracy may be increased further by using more hyperparameters or with a different model.

# In[ ]:




