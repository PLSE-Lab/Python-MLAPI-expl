#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.metrics import classification_report

# TODO:
# - Clean functions
# - Training
# - Predicting


# # Functions to clean data

# In[ ]:


def remove_columns(data, columns):
    data.drop(columns, axis=1, inplace=True)
    
    return data


# In[ ]:


# set_median(data, 'Age')
def set_median(data, column):
    age_median = np.median(data[data[column].notnull()][column])
    data[column] = data[column].fillna(age_median)
    
    return data


# In[ ]:


def fill_nan_values(data, column, value = -1):
    data[column] = data[column].fillna(value)
    
    return data


# In[ ]:


def factorize_column(data, column):
    labels, uniques = pd.factorize(data[column])
    data[column] = labels
    
    return data


# In[ ]:


def normalize_columns(data, columns):
    scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    
    return data


# # Prepare training data

# In[ ]:


data = pd.read_csv('../input/train.csv')


# In[ ]:


# Remove some columns
data = remove_columns(data, ['PassengerId', 'Name', 'Ticket'])


# In[ ]:


data = set_median(data, 'Age')


# In[ ]:


data = fill_nan_values(data, 'Cabin')
data = factorize_column(data, 'Cabin')


# In[ ]:


data = fill_nan_values(data, 'Embarked')
data = factorize_column(data, 'Embarked')


# In[ ]:


data = normalize_columns(data, ['Age', 'Fare'])


# In[ ]:


# One-hot the data
data = pd.get_dummies(data)


# # Training and validate training data

# In[ ]:


# Split data

output = data['Survived']
data = remove_columns(data, 'Survived')
X_train, X_dev, y_train, y_dev = train_test_split(data, output, test_size = 0.3, random_state = 0)


# In[ ]:


def show_scores(y, predicted):
    acc = accuracy_score(y, predicted)
    fbeta = fbeta_score(y, predicted, beta=0.5)
    
    print(classification_report(y, predicted))


# In[ ]:


# Gaussian default model
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)

pred_train_nb = clf.predict(X_train)
pred_dev_nb = clf.predict(X_dev)

print("TEST")
show_scores(y_train, pred_train_nb)
print("DEV")
show_scores(y_dev, pred_dev_nb)


# In[ ]:


# AdaBoost default model
from sklearn.ensemble import AdaBoostClassifier
clf_ada = AdaBoostClassifier(random_state = 0)
clf_ada.fit(X_train, y_train)

pred_train_ada = clf_ada.predict(X_train)
pred_dev_ada = clf_ada.predict(X_dev)

print("TEST")
show_scores(y_train, pred_train_ada)
print("DEV")
show_scores(y_dev, pred_dev_ada)


# In[ ]:


# Use GridSearch to improve the parameters of AdaBoost classifier
from sklearn.model_selection import GridSearchCV

params = {"n_estimators":[100, 150, 200, 250], "learning_rate":[0.001, 0.01, 0.1, 1, 1.5, 2], "algorithm": ["SAMME", "SAMME.R"]}
grd = GridSearchCV(clf_ada, params, cv=10)

grd.fit(X_train, y_train)


# In[ ]:


print(grd.best_score_)
print(grd.best_params_)


# In[ ]:


# Create a new AdaBoost classifier whith the parameters above
clf_ada_grd = AdaBoostClassifier(algorithm='SAMME', learning_rate=1, n_estimators=150, random_state=0)
clf_ada_grd.fit(X_train, y_train)

p_train = clf_ada_grd.predict(X_train)
p_dev = clf_ada_grd.predict(X_dev)

print("TEST")
show_scores(y_train, p_train)
print("DEV")
show_scores(y_dev, p_dev)


# # Prepare test data

# In[ ]:


# Load test data
test_data = pd.read_csv('../input/test.csv')


# In[ ]:


# Remove some columns
test_data = remove_columns(test_data, ['PassengerId', 'Name', 'Ticket'])


# In[ ]:


test_data = set_median(test_data, 'Age')


# In[ ]:


test_data = fill_nan_values(test_data, 'Cabin')
test_data = factorize_column(test_data, 'Cabin')


# In[ ]:


test_data = fill_nan_values(test_data, 'Embarked')
test_data = factorize_column(test_data, 'Embarked')


# In[ ]:


test_data = fill_nan_values(test_data, 'Fare')


# In[ ]:


test_data = normalize_columns(test_data, ['Age', 'Fare'])


# In[ ]:


# One-hot the data
test_data = pd.get_dummies(test_data)


# # Predict

# In[ ]:


p_gaussian = clf.predict(test_data)
print(p_gaussian)


# In[ ]:


p_ada = clf_ada.predict(test_data)
print(p_ada)


# In[ ]:


p_ada_grd = clf_ada_grd.predict(test_data)
print(p_ada_grd)


# # Create submission file for Kaggle

# In[ ]:


passengers_id = pd.read_csv('../input/test.csv')['PassengerId']
submission = pd.DataFrame({'PassengerId':passengers_id,'Survived':p_ada_grd})

filename = 'titanic_predictions_2.csv'

submission.to_csv(filename,index=False)

print(filename)


# In[ ]:





# In[ ]:




