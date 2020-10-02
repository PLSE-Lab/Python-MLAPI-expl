#!/usr/bin/env python
# coding: utf-8

# # Import all necessary libraries

# In[ ]:


# import general libraries
import numpy as np
import pandas as pd
import os
from IPython.display import display
# import machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# # Analyze the data

# First of all we load the data into our workspace.

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
submissionfile = pd.read_csv("../input/gender_submission.csv")


# Take a closer look to our data.

# In[ ]:


print("Show the head of the data")
display(train_df.head())
print("Describe the data")
display(train_df.describe(include="all"))
print("Show the datatypes of the columns")
train_df.dtypes


# In[ ]:


test_df.head()


# Define the needed columns: 
# ***
# The independent variables are the *X_needed_columns*
# ***
# The dependant variables are the *Y_needed_columns*, in short terms the variable "survived"
# ***

# In[ ]:


# Drop all the empty rows from the training data
train_df = train_df.dropna(axis=0)
# Encode the column Sex into 1,0
train_df["Sex"] = train_df["Sex"].replace(["male","female"],[1,0])
# Encode the Embarked line
train_df["Embarked"] = train_df["Embarked"].replace(train_df["Embarked"].unique(),[0,1,2])
# Change the Age to an integer type
train_df["Age"] = train_df["Age"].apply(lambda x: int(x))
# Change the Fare to an integer type
train_df["Fare"] = train_df["Fare"].apply(lambda x: int(x))


# In[ ]:


needed_columns = ["PassengerId","Survived","Pclass","Sex","Age","SibSp","Fare","Embarked"]
X_needed_columns = ["PassengerId","Pclass","Sex","Age","SibSp","Fare","Embarked"]
Y_needed_columns = ["Survived"]


# In[ ]:



X_train = train_df[X_needed_columns]
y_train = train_df[Y_needed_columns]


# # Prepare the data

# In[ ]:


# Do the same manipulations as before with the train data, expect we do not exclude the nan's, we set them to zero
test_df = test_df.fillna(0)
test_df["Sex"] = test_df["Sex"].replace(["male","female"],[1,0])
test_df["Embarked"] = test_df["Embarked"].replace(test_df["Embarked"].unique(),[0,1,2])
test_df["Age"] = test_df["Age"].apply(lambda x: int(x))
test_df["Fare"] = test_df["Fare"].apply(lambda x: int(x))


# In[ ]:


X_test = test_df[X_needed_columns]
X_test = X_test.fillna(0)


# In[ ]:


results = submissionfile["Survived"]


# # Run different ML models and compare the outcomes

# ## Logistic Regression

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred_logreg = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)
acc_log


# ## C-Support Vector Classification

# In[ ]:


svc = SVC()
svc.fit(X_train, y_train)
Y_pred_svc = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100, 2)
acc_svc


# ## K-Nearest Neighbours

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred_knn = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
acc_knn


# ## Naive Bayes Classifier

# In[ ]:


gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred_gaussian = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
acc_gaussian


# ## Perceptron

# In[ ]:


perceptron = Perceptron()
perceptron.fit(X_train, y_train)
Y_pred_perceptron = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)
acc_perceptron


# ## Linear C-Support Vector Classification

# In[ ]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
Y_pred_linear_svc = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)
acc_linear_svc


# ## Linear classifier with SGD training

# In[ ]:


sgd = SGDClassifier()
sgd.fit(X_train, y_train)
Y_pred_sgd = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)
acc_sgd


# ## Decision tree classifier

# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
Y_pred_decission_tree = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
acc_decision_tree


# ## Random forest classifier

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest


# ## Evaluate the results

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


finalresults = pd.DataFrame(columns=["PassengerID", "Survived"])


# In[ ]:


finalresults["PassengerID"] = X_test["PassengerId"]
finalresults["Survived"] = Y_pred_svc


# In[ ]:


finalresults.to_csv("submission.csv",index=False)


# In[ ]:




