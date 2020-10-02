#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer EDA

# ## Dataset: [Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

# ### In this tutorial, we will learn how to select the best parameters for our models. We will learn how to use GridSearchCV from the sklearn.model_selection package to tune all the parameters.
# 
# ### Hyperparameter optimization or tuning is the problem of choosing a set of optimal hyperparameters for a learning algorithm.
# 
# ### It can be as simple as the following:
# * How many trees should I include in my random forest?
# * What degree of polynomial features should I use for my linear model?
# * What should be the maximum depth allowed for my decision tree?
# * How many layers should I have in my neural network?
# * What should I set my learning rate to for gradient descent?

# ### Import all the necessary header files as follows:
# * pandas : An open source library used for data manipulation, cleaning, analysis and visualization. 
# * numpy : A library used to manipulate multi-dimensional data in the form of numpy arrays with useful in-built functions. 
# * matplotlib : A library used for plotting and visualization of data. 
# * seaborn : A library based on matplotlib which is used for plotting of data. 
# * sklearn.metrics : A library used to calculate the accuracy, precision and recall. 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


# In[ ]:


# Importing the dataset
data = pd.read_csv("../input/data.csv")


# ### Inspecting and cleaning the data

# In[ ]:


# Printing the 1st 5 columns
data.head()


# In[ ]:


# Printing the dimensions of data
data.shape


# In[ ]:


# Viewing the column heading
data.columns


# In[ ]:


# Inspecting the target variable
data.diagnosis.value_counts()


# In[ ]:


data.dtypes


# In[ ]:


# Identifying the unique number of values in the dataset
data.nunique()


# In[ ]:


# Checking if any NULL values are present in the dataset
data.isnull().sum()


# Dropping the Unnamed: 32 and the id column since these do not provide any useful information for our models.

# In[ ]:


data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)


# In[ ]:


# See rows with missing values
data[data.isnull().any(axis=1)]


# In[ ]:


# Viewing the data statistics
data.describe()


# ### Data Visualization

# In[ ]:


# Finding out the correlation between the features
corr = data.corr()
corr.shape


# In[ ]:


# Plotting the heatmap of correlation between features
plt.figure(figsize=(20,20))
sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')
plt.show()


# The above heatmap shows us a correlation between the various features. The closer the value to 1, the higher is the correlation between the pair of features.

# In[ ]:


# Analyzing the target variable

plt.title('Count of cancer type')
sns.countplot(data['diagnosis'])
plt.xlabel('Cancer lethality')
plt.ylabel('Count')
plt.show()


# In[ ]:


# Plotting correlation between diagnosis and radius

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.boxplot(x="diagnosis", y="radius_mean", data=data)
plt.subplot(1,2,2)
sns.violinplot(x="diagnosis", y="radius_mean", data=data)
plt.show()


# Boxplot shows us the minimum, first quartile (Q1), median, third quartile (Q3), and maximum. It is useful for detecting the outliers. <br>
# Violin plot shows us the kernel density estimate on each side.

# In[ ]:


# Plotting correlation between diagnosis and concativity

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.boxplot(x="diagnosis", y="concavity_mean", data=data)
plt.subplot(1,2,2)
sns.violinplot(x="diagnosis", y="concavity_mean", data=data)
plt.show()


# In[ ]:


# Distribution density plot KDE (kernel density estimate)
sns.FacetGrid(data, hue="diagnosis", height=6).map(sns.kdeplot, "radius_mean").add_legend()
plt.show()


# In[ ]:


# Plotting the distribution of the mean radius
sns.stripplot(x="diagnosis", y="radius_mean", data=data, jitter=True, edgecolor="gray")
plt.show()


# In[ ]:


# Plotting bivariate relations between each pair of features (4 features x4 so 16 graphs) with hue = "diagnosis"
sns.pairplot(data, hue="diagnosis", vars = ["radius_mean", "concavity_mean", "smoothness_mean", "texture_mean"])
plt.show()


# ### Once the data is cleaned, we split the data into training set and test set to prepare it for our machine learning model in a suitable proportion.

# In[ ]:


# Spliting target variable and independent variables
X = data.drop(['diagnosis'], axis = 1)
y = data['diagnosis']


# In[ ]:


# Splitting the data into training set and testset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
print("Size of training set:", X_train.shape)
print("Size of test set:", X_test.shape)


# ## Logistic Regression

# In[ ]:


# Logistic Regression

# Import library for LogisticRegression
from sklearn.linear_model import LogisticRegression

# Create a Logistic regression classifier
logreg = LogisticRegression()

# Train the model using the training sets 
logreg.fit(X_train, y_train)


# In[ ]:


# Prediction on test data
y_pred = logreg.predict(X_test)


# In[ ]:


# Calculating the accuracy
acc_logreg = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of Logistic Regression model : ', acc_logreg )


# ## Gaussian Naive Bayes

# In[ ]:


# Gaussian Naive Bayes

# Import library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(X_train,y_train)


# In[ ]:


# Prediction on test set
y_pred = model.predict(X_test)


# In[ ]:


# Calculating the accuracy
acc_nb = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of Gaussian Naive Bayes model : ', acc_nb )


# ## Decision Tree

# In[ ]:


# Decision Tree Classifier

# Import Decision tree classifier
from sklearn.tree import DecisionTreeClassifier

# Create a Decision tree classifier model
clf = DecisionTreeClassifier()


# ## Hyperparameter Optimization
# 
# Steps to tune the parameters:
# 1. Prioritize those parameters which have the most effect on our model. (Example: n-neighbors bor KNN, n-estimators for random forest etc.)
# 2. Set various values to these parameters and store them in a dictionary as shown below.
# 3. Create an object of the GridSearchCV class and assign the parameters to it.
# 4. Fit the training set in the object.
# 5. We will get the best parameters from the best_estimator_ property of the object.
# 6. Use this object to fit training set to your classifier.

# In[ ]:


# Hyperparameter Optimization
parameters = {'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10, 50], 
              'min_samples_split': [2, 3, 50, 100],
              'min_samples_leaf': [1, 5, 8, 10]
             }

# Run the grid search
grid_obj = GridSearchCV(clf, parameters)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Train the model using the training sets 
clf.fit(X_train, y_train)


# In[ ]:


# Prediction on test set
y_pred = clf.predict(X_test)


# In[ ]:


# Calculating the accuracy
acc_dt = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of Decision Tree model : ', acc_dt )


# ## Random Forest

# Follow the above mentioned steps to tune the parameters.

# In[ ]:


# Random Forest Classifier

# Import library of RandomForestClassifier model
from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest Classifier
rf = RandomForestClassifier()

# Hyperparameter Optimization
parameters = {'n_estimators': [4, 6, 9, 10, 15], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1, 5, 8]
             }

# Run the grid search
grid_obj = GridSearchCV(rf, parameters)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the rf to the best combination of parameters
rf = grid_obj.best_estimator_

# Train the model using the training sets 
rf.fit(X_train,y_train)


# In[ ]:


# Prediction on test data
y_pred = rf.predict(X_test)


# In[ ]:


# Calculating the accuracy
acc_rf = round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 )
print( 'Accuracy of Random Forest model : ', acc_rf )


# ## Support Vector Machine

# In[ ]:


# SVM Classifier

# Creating scaled set to be used in model to improve the results
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


# Import Library of Support Vector Machine model
from sklearn import svm

# Create a Support Vector Classifier
svc = svm.SVC()

# Hyperparameter Optimization
parameters = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

# Run the grid search
grid_obj = GridSearchCV(svc, parameters)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the svc to the best combination of parameters
svc = grid_obj.best_estimator_

# Train the model using the training sets 
svc.fit(X_train,y_train)


# In[ ]:


# Prediction on test data
y_pred = svc.predict(X_test)


# In[ ]:


# Calculating the accuracy
acc_svm = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of SVM model : ', acc_svm )


# ## K - Nearest Neighbors

# In[ ]:


# K - Nearest Neighbors

# Import library of KNeighborsClassifier model
from sklearn.neighbors import KNeighborsClassifier

# Create a KNN Classifier
knn = KNeighborsClassifier()

# Hyperparameter Optimization
parameters = {'n_neighbors': [3, 4, 5, 10], 
              'weights': ['uniform', 'distance'],
              'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'leaf_size' : [10, 20, 30, 50]
             }

# Run the grid search
grid_obj = GridSearchCV(knn, parameters)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the knn to the best combination of parameters
knn = grid_obj.best_estimator_

# Train the model using the training sets 
knn.fit(X_train,y_train)


# In[ ]:


# Prediction on test data
y_pred = knn.predict(X_test)


# In[ ]:


# Calculating the accuracy
acc_knn = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of KNN model : ', acc_knn )


# ## Evaluation and comparision of all the models

# In[ ]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Random Forest', 'Support Vector Machines', 
              'K - Nearest Neighbors'],
    'Score': [acc_logreg, acc_nb, acc_dt, acc_rf, acc_svm, acc_knn]})
models.sort_values(by='Score', ascending=False)


# ## We can see from the above table that SVM classifier works best for this dataset.
# ### Before hyperparameter tuning, I was getting an accuracy of mere 92.40 for SVM model but after parameter tuning, we obtained an accuracy of 98.25. Hence parameter tuning is important to get a very high accuracy.  

# Please upvote if you found this kernel useful! :) <br>
# Feedbacks appreciated.
