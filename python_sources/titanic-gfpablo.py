#!/usr/bin/env python
# coding: utf-8

# # TITANIC
# 
# ## Data Handling
# Filling the missing values, and choosing which columns we are going to use.

# In[24]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import IPython
from IPython import display
import seaborn as sns 
import sklearn as skl
#ignore annoying warnings
import warnings
warnings.filterwarnings('ignore')


# Read the CSVs provided

# In[25]:


train_file_path = '../input/train.csv'
train = pd.read_csv(train_file_path)
test_file_path = '../input/test.csv'
test = pd.read_csv(test_file_path)


# We will create a new column, based on the names. This column will be called 'Tittle' and will be very usefull in our aim to fill the missing ages, and to our model too.

# In[26]:


normalized_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}
    
def createTitle(dataframe):
    dataframe['Title'] = dataframe.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
    dataframe.Title = dataframe.Title.map(normalized_titles)

createTitle(train)
createTitle(test)


# Let's group by sex, class and tittle and infer the missing ages.
# 
# **Pay attention! We use the train DataFrame for the inference in the test DataFrame avoiding a data leakage**

# In[27]:


grouped = train.groupby(["Sex", "Pclass", "Title"])

train.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))
test.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))


# Now, we must encode all our categorical values that can cause problems with our models.

# In[28]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
train['Sex_Code'] = label.fit_transform(train['Sex'])
test['Sex_Code'] = label.fit_transform(test['Sex'])

train = pd.get_dummies(train, columns=["Title"])
test = pd.get_dummies(test, columns=["Title"])
train = pd.get_dummies(train, columns=["Pclass"])
test = pd.get_dummies(test, columns=["Pclass"])


# Ending with data handling, we have to scale and center our data with a StandardScaler. Again, we will fit our scaler with **ONLY** train DataFrame

# In[29]:


columns = ["Sex_Code","Age","Title_Miss","Title_Mr","Title_Mrs","Title_Officer","Title_Royalty","Pclass_2","Pclass_3"]
y_train = train["Survived"]

train = train[columns]
X_test = test[columns]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(train.values)
scaled_features = scaler.transform(train.values)
scaled_features_train = pd.DataFrame(scaled_features, index=train.index, columns=train.columns)

scaled_features = scaler.transform(X_test.values)
scaled_features_test = pd.DataFrame(scaled_features, index=X_test.index, columns=X_test.columns)

#We will not use test anymore


# ## Modeling
# We must focus on train DataFrame, our test DataFrame is ready for kaggle, so, we must use our train DataFrame to develop an accurate model. We can choose between a huge variety of models, but I want to try just these:
# - Logistic Regression
# - Support Vector Machines
# - Decission Tree
# - Perceptron
# - Random Forest
# 
# Have to say that we are going to use the cross validation technique for the comparission of all these models.

# In[30]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier


# Logistic regression measures the relationship between the categorical dependent variable (feature) and one or more independent variables (features) by estimating probabilities using a logistic function, which is the cumulative logistic distribution. [Wikipedia]

# In[31]:


# Logistic Regression
lr = LogisticRegression()
cv_results = cross_validate(lr, train, y_train, cv=10, return_train_score=False)
print(cv_results["test_score"].mean())


# Support Vector Machines are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training samples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new test samples to one category or the other, making it a non-probabilistic binary linear classifier. [Wikipedia] 

# In[32]:


# Support Vector Machines
svc = SVC()
cv_results = cross_validate(svc, train, y_train, cv=10, return_train_score=False)
print(cv_results["test_score"].mean())


# K-Nearest Neighbors algorithm (or k-NN for short) is a non-parametric method used for classification and regression. A sample is classified by a majority vote of its neighbors, with the sample being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor. [Wikipedia]

# In[33]:


#K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors = 5)
cv_results = cross_validate(knn, train, y_train, cv=10, return_train_score=False)
print(cv_results["test_score"].mean())


# Perceptron is an algorithm for supervised learning of binary classifiers (functions that can decide whether an input, represented by a vector of numbers, belongs to some specific class or not). It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector. The algorithm allows for online learning, in that it processes elements in the training set one at a time. [Wikipedia]

# In[34]:


# Perceptron
perceptron = Perceptron()
cv_results = cross_validate(perceptron, train, y_train, cv=10, return_train_score=False)
print(cv_results["test_score"].mean())


# This model uses a decision tree as a predictive model which maps features (tree branches) to conclusions about the target value (tree leaves). Tree models where the target variable can take a finite set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees. [Wikipedia]

# In[35]:


# Decision Tree
decision_tree = DecisionTreeClassifier()
cv_results = cross_validate(decision_tree, train, y_train, cv=10, return_train_score=False)
print(cv_results["test_score"].mean())


# Random Forests is one of the most popular. Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees (n_estimators=100) at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. [Wikipedia]

# In[36]:


# Random Forest
random_forest = RandomForestClassifier(n_estimators=50)
cv_results = cross_validate(decision_tree, train, y_train, cv=10, return_train_score=False)
print(cv_results["test_score"].mean())


# As we can see, the best option is Random Forest. Althought, it could be overfitting, instead of it, we are going to upload the logistic regression as sometimes the simplier, the better.
# 
# **Pay attention! We fit and make the predictions again with all our train DataFrame.**

# In[37]:


#Fit random forest and make the predictions over the test
lr.fit(train, y_train)
Y_pred = lr.predict(X_test)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })

#submission.to_csv('../output/submission.csv', index=False)

