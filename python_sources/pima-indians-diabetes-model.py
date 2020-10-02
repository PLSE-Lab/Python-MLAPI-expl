#!/usr/bin/env python
# coding: utf-8

# # Pima Indians Diabetes
# 
# ## Context
# 
# This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objectives of the dataset is to diagnostically predict whether or not a patient has diabetes, based on a certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients, here, are females at least 21 years old of Pima Indian heritage.
# 
# ## Content
# 
# The datasets consists of several medical predictor variables and one target variable, `Outcome`. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
# 
# ## Acknowledgements
# 
# Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). (Using the ADAP learning algorithm to forecast the onset of diabetes mellitus)[https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/]. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.
# 
# ## Question
# 
# We would like to build a machine learning model to accurately predict whether or not the patients in the dataset have diabetes or not.

# In[ ]:


# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


# In[ ]:


# Load data
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']


# In[ ]:


# Perform classification
names = ["Logistic Model", "Boosting", "Bagging", "Random Forests"]

classifiers = [
    LogisticRegression(),
    AdaBoostClassifier(),
    RandomForestClassifier(max_features=None),
    RandomForestClassifier(max_features='sqrt')
]

X = df.loc[:, features]
Y = df.loc[:, 'Outcome']

X = StandardScaler().fit_transform(X)
X_train, X_test, Y_train, Y_test =     train_test_split(X, Y, test_size=0.2, random_state=42)

# Iterate over classifiers
scores = []
for name, clf in zip(names, classifiers):
    clf.fit(X_train, Y_train)
    scores.append(clf.score(X_test, Y_test))


# In[ ]:


pd.DataFrame({'Classifier': names, 'Accuracy' : scores})

