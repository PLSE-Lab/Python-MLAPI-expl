#!/usr/bin/env python
# coding: utf-8

# # Indian Liver Patient Records
# 
# Here, we have a prolem to make a model that can classify whether a patient has liver problem or not. We implement a Random Forest Classifier with some hyperparameters tuning using Grid Search method.
# 
# In summary, the algorithm below consists of 4 steps:
# 
# **1. Data Preprocessing: **
# From the provided data, we have 583 lines of records, 10 dependent variables and an independent variable. We implement Imputer for handling missing data, encoding the categorical data, Feature Scaling and Dimesnionality Reduction using Principle Component Analysis (PCA) method. 
# 
# 
# **2. Make a prediction model: **
# * Here, we implemented Random Forest Classifier to predict the outcome of given independent variables.
# 
# 
# **3.  Implementing Grid Search: **
# Grid Search is a method to find the best hyperparameters of our model in order to increase the training and testing result
# 
# 
# **4.  K-Fold Cross Validation: **
# In the end, we want to make sure that our optimized classifier does not overfit if it is applied to new testing data. 
# 
# 
# So, let's start our ML journey!

# # 1. Data Prepropcessing

# In[1]:


# Importing the libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Importing the dataset
dataset = pd.read_csv("../input/indian_liver_patient.csv")
dataset_desc = dataset.describe(include = 'all')
print(dataset_desc)

# Identifying the dependent (x) and independent (y) variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 10].values


# From the detail of the data above, we can see that actually there are some different number of data count for "Albumin_and_Globulin_Ratio" which is only 583, while the other features have 579. There are possibly some missing values. To make sure, let's check!

# In[2]:


# Checking missing data
dataset_mis = dataset.isnull().sum()
print(dataset_mis)


# And, it is clearly that "Albumin_and_Globulin_Ratio" attribute has 4 missing values. Let's solve it using Imputer.

# In[3]:


# Taking care of missing data!
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
X[:, 2:10] = imputer.fit_transform(X[:, 2:10])


# Because we have one categorical attribute "Gender", we need to encode this n order our machine learning model can undertand. We implement LabelEncoder to make a unique labels for "Male" and "Female" differently. Then we implement OneHotEncoder to make a dummy variables.

# In[7]:


# Encoding the categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
onehotencoder_X = OneHotEncoder(categorical_features = [1])
onehotencoder_X.fit_transform(X).toarray()


# Spliiting the dtaset into 75 % of training data and 25% of testing data.

# In[9]:


# Splitting the dataset into trainig and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)


# To make our dataset in a good scale, we need to implement feature scaling. Because if we see the data, the have different scales. For example "Age" and "kaline_Phospota" have larger scale than other attributes.

# In[10]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Sometimes, feeding our machine learnig model with all attributes that we have is not the best option. e can implement dimesnionality reduction to reduce features given to the model and decrease the computation cost but with still have the same spirit to obtain the same or even better classifier. In fact, here I only use 6 components of Principle Component Analysis (PCA) resulting a better classifier than using all 10 attributes from the data. The process to find the number of n_components is as below:

# In[11]:


# Dimesnionality Reduction using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train  = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print("Explained Variance:\n", explained_variance)


# From the explained variance above, I decide to use only the first 6 features of PCA which already represent more than 80% of total variance. This number is more than enough to feed the machine learning model and has represented all the features. So, let's update our n_compnents = 6.

# In[14]:


pca = PCA(n_components = 6)
X_train  = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# # 2. Prediction Model - Random Forest Classifier
# 
# We firstly implemented Random Forest Classifier without tuning the hyperparameter. The idea is we will compare the result with the one that will be tuned based on Grid Search method

# In[15]:


# Fitting Random Forest model into the training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)


# In[16]:


# Making a prediction
from sklearn.metrics import classification_report
y_pred = classifier.predict(X_test)
test_accuracy = classification_report(y_test, y_pred)
print(test_accuracy)


# We only got 67% of testing result from the model without doing any tuning. 

# # 3. Grid Search
# We then implemented Grid Search to find the best hyperparameters. You can read from sk-learn documentation for Random Forest Classification [here](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) to know more the hyperparameters that can be tuned to get optimum result.

# In[17]:


# Grid Search to find the best tuning
# Params for Random Forest
parameters = [{'criterion' : ['gini', 'entropy'],
               'max_depth' : [5, 6, 7, 8, 9, 10, 11, 12],
               'max_features' : [1, 2, 3],
               'n_estimators' : [14, 15, 16, 17, 18, 19],
               'random_state' : [7, 8, 9, 10, 11, 12, 13]}]

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1, cv = 10)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
print("Best accuracy of the model for the training set is:", best_accuracy)
best_params = grid_search.best_params_
print("Best parameters of the model for the training set is:", best_params)


# After running the Grid Search, we can find that the best accuracy for the training data is up to 75% with best_params: 
#  - criterion = 'gini'
#  - max_depth = 9
#  - max_features = 1
#  - n_estimators = 16
#  - random_state = 10
#  
#  We then include the best_params in our RandomForestClassifier to update the model. 
#  
# *Notes: On the code we implemented a distributed computing with "n_jobs = -1" in order we can use all core resources of our computer*

# In[18]:


# Tune the hyperparameters of Random Forest Classifier based on best_params resulted from Grid Search method
classifier = RandomForestClassifier(criterion = 'gini',
                                    max_depth = 9,
                                    max_features = 1,
                                    n_estimators = 16,
                                    random_state = 10)
classifier.fit(X_train, y_train)


# In[19]:


# Let's check the test accuracy after optimised
y_pred = classifier.predict(X_test)
test_accuracy_optimized = classification_report(y_test, y_pred)
print(test_accuracy_optimized)


# In[20]:


# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
test_accuracy_optimized_cm = (cm[0,0]+cm[1,1])/146 #146 is the total number of testing data
print("\nTesting accuracy based on the Confusion Matrix:\n", test_accuracy_optimized_cm)


# The model indeed increase the accuracy for predicting the test result from 67% up to 71% after doing the hyperparameter tuning.

# # K-Fold Cross Validation
# 
# To make sure that our model is not over fitting when it is implemented to the new set of test data, we then implemented K-Fold Cross Validation to know the mean accuracy and the the standard deviation from given 10 different set of validation.

# In[21]:


# K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,
                            X = X_train, y = y_train,
                            cv = 10, n_jobs = -1)
print("Showing all 10 of K-Fold Cross Validation accuracies:\n", accuracies)
accuracies_mean = accuracies.mean()
print("\nMean of accuracies:\n", accuracies_mean)
accuracies_std = accuracies.std()
print("\nStandard Deviation:\n", accuracies_std)


# From the validation step, we can clearly see that the mean of accuracy for our model is 75% with standard deviation only 3% meaning that our model has very slight variance in the different set of validation. So, we can be confident enought that our model is not likely overfitting. Yeah, we got a good result at the end!
# 
# Notes for improvement:
# 1. Try to use another model like Neural Networks to get a better result. I have tried to implement Naive Bayes and SVM but they did not beat the Random Forest even after implementing Grid Search.
# 2. Use tpot to find the best model with the best hyperparameters. Not forget to mention XGBoost for the same spirit to get the better result. I would love to implement them if I have time in the near future.
# 
# Enjoy Machine Learning!
# Royan Dawud Aldian
