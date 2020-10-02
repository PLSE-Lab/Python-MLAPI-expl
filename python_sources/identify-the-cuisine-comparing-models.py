#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Kernel contains a model that is used to identify the type of Cuisine Based on the list of ingredients.

# Importing the libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


# importing the training dataset
train_dataset = pd.read_json("../input/train.json")
# count of the rows and columns
print(train_dataset.shape)
# Display the first 5 rows of the data
train_dataset.head()


# In[ ]:


# importing the test dataset
test_dataset = pd.read_json("../input/test.json")
# count of the rows and columns
print(test_dataset.shape)
# Display the first 5 rows of the data
test_dataset.head()


# In[ ]:


# Check if null values are present in the training dataset
sum(train_dataset['ingredients'].isnull())


# In[ ]:


# Check the number of unique Cuisines that are present in the dataset
cuisines = train_dataset['cuisine'].unique()
print(cuisines.shape)


# In[ ]:


# Create a function to extract the food items from the ingredients.
def extract_items(ingredients):
    food_items = []
    for items in ingredients:
        for item in items:
            if item in food_items:
                pass
            elif item not in food_items:
                food_items.append(item)
            else:
                pass
    return food_items

# Call the function with the data in the dataset
ingredients = extract_items(train_dataset['ingredients'])

# List the first 10 ingredients
print(ingredients[:10])


# In[ ]:


# Count of the ingredients
print(len(ingredients))


# In[ ]:


# Add each ingredient in the training dataset with a value of 0
for ingredient in ingredients:
    train_dataset[ingredient] = np.zeros(len(train_dataset["ingredients"]))


# In[ ]:


# View the new addition in the training dataset
train_dataset.head()


# In[ ]:


# Add each ingredient in the test dataset with a value of 0
for ingredient in ingredients:
    test_dataset[ingredient] = np.zeros(len(test_dataset["ingredients"]))


# In[ ]:


# View the new addition in the test dataset
test_dataset.head()


# In[ ]:


# Encoding the categorical variables
# A function to add a value of 1 in the particular cell(set of a row and column),
# created for each ingredient for the row based on the fact that the ingredient is present or not.

def find_item(ingredients_list, dataset):
    position = 0
    for items in ingredients_list:
        for item in items:
            if item in ingredients:
                dataset.loc[position, item] = 1
            else:
                pass
        position = position + 1


# In[ ]:


# Call the function for training data
find_item(train_dataset["ingredients"], train_dataset)


# In[ ]:


# View the changes in the training data
train_dataset.head()


# In[ ]:


# Call the function for test data
find_item(test_dataset["ingredients"], test_dataset)


# In[ ]:


# View the changes in the test data
test_dataset.head()


# In[ ]:


# Fitting Random Forest Classifier to the training dataset

# Importing the libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *


# In[ ]:


# Define X(predictors)
# All the encoded ingredients
X = train_dataset[ingredients]
X.head()


# In[ ]:


# Define Y(Dependent Variable)
y = train_dataset["cuisine"]
y.head()


# In[ ]:


# Splitting the training data in training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[ ]:


X_train.head()


# In[ ]:


X_test.head()


# In[ ]:


y_train.head()


# In[ ]:


y_test.head()


# In[ ]:


# # Random Forest Classifier
# classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)


# In[ ]:


# Predicted result
# y_pred = classifier.predict(X_test)
# y_pred


# In[ ]:


# Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# cm


# In[ ]:


# # Accuracy --0.63
# accuracy_score(y_test, y_pred)


# In[ ]:


# # Random Forest Classifier-Increasing the estimators from 10 to 100
# classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)


# In[ ]:


# # Predicted result
# y_pred = classifier.predict(X_test)
# y_pred


# In[ ]:


# # Accuracy with 100 estimators --0.67
# accuracy_score(y_test, y_pred)


# In[ ]:


# # Random Forest Classifier-Increasing the estimators from 100 to 500
# classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)


# In[ ]:


# # Predicted result
# y_pred = classifier.predict(X_test)
# y_pred


# In[ ]:


# # Accuracy with 100 estimators --0.68
# accuracy_score(y_test, y_pred)


# In[ ]:


# # Naive Bayes Classifier
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()


# In[ ]:


# classifier.fit(X_train, y_train)


# In[ ]:


# # Predicted result
# y_pred = classifier.predict(X_test)
# y_pred


# In[ ]:


# # Accuracy --0.38
# accuracy_score(y_test, y_pred)


# In[ ]:


# # Kernel SVM Classifier
# from sklearn.svm import SVC
# classifier = SVC(kernel = 'rbf', random_state = 0)
# classifier.fit(X_train, y_train)


# In[ ]:


# # Predicted result
# y_pred = classifier.predict(X_test)
# y_pred


# In[ ]:


# # Accuracy
# accuracy_score(y_test, y_pred)


# In[ ]:


# # K-Nearest Neighbors
# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# classifier.fit(X_train, y_train)


# In[ ]:


# # Predicted result
# y_pred = classifier.predict(X_test)
# y_pred


# In[ ]:


# # Accuracy
# accuracy_score(y_test, y_pred)


# In[ ]:


# # # Support Vector Machines
# from sklearn.svm import SVC
# classifier = SVC(kernel = 'linear',random_state = 0)
# classifier.fit(X_train, y_train)


# In[ ]:


# # Predicting the results
# y_pred = classifier.predict(X_test)
# y_pred


# In[ ]:


# # Accuracy --0.76
# accuracy_score(y_test, y_pred)


# In[ ]:


# # # Support Vector Machines
# from sklearn.svm import SVC
# classifier = SVC(C = 100,kernel = 'linear',degree = 3, gamma = 1, random_state = 0, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1)
# classifier.fit(X_train, y_train)


# In[ ]:


# # Predicting the results
# y_pred = classifier.predict(X_test)
# y_pred


# In[ ]:


# # Accuracy --0.70
# accuracy_score(y_test, y_pred)


# In[ ]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[ ]:


# Predicted result
y_pred = classifier.predict(X_test)
y_pred


# In[ ]:


# Accuracy --0.78
accuracy_score(y_test, y_pred)


# In[ ]:


# # XGBoost

# from xgboost import XGBClassifier
# classifier = XGBClassifier()
# classifier.fit(X_train, y_train)


# In[ ]:


# # Predicting the results
# y_pred = classifier.predict(X_test)
# y_pred


# In[ ]:


# # Accuracy --- 68.88
# accuracy_score(y_test, y_pred)


# In[ ]:


y_final_prediction = classifier.predict(test_dataset[ingredients])


# In[ ]:


output = test_dataset['id']
output = pd.DataFrame(output)
output['cuisine'] = pd.Series(y_final_prediction)


# In[ ]:


output.to_csv('sample_submission.csv', index=False)

