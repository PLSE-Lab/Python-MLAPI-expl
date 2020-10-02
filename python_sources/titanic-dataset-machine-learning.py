#!/usr/bin/env python
# coding: utf-8

# ## Import the relevant libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Setting the data 

# In[ ]:


test = pd.read_csv('../input/titanic/test.csv')
train = pd.read_csv('../input/titanic/train.csv')
gender = pd.read_csv('../input/titanic/gender_submission.csv')


# ## Analyzing the data

# In[ ]:


train.head(10)


# In[ ]:


test.head(10)


# In[ ]:


gender.head(10)


# In[ ]:


train.describe()


# In[ ]:


train.dtypes


# In[ ]:


train.info()


# In[ ]:


test.describe()


# In[ ]:


test.dtypes


# In[ ]:


test.info()


# ## Verify missing values

# In[ ]:


train.isna().sum()


# In[ ]:


test.isna().sum()


# In[ ]:


gender.isna().sum()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False)


# ## Exploratory analysis

# In[ ]:


sns.barplot(x = 'Sex', y = 'Survived', data = train)


# In[ ]:


train_Sex_Survived = train[['Sex','Survived']].groupby(['Sex'])
train_Sex_Survived.mean().sort_values(by = 'Sex', ascending = False)


# In[ ]:


sns.barplot(x = 'Pclass', y = 'Survived', data = train)


# In[ ]:


train_Pclass_Survived = train[['Pclass','Survived']].groupby(['Pclass'])
train_Pclass_Survived.mean().sort_values(by='Survived', ascending = False)


# In[ ]:


sns.barplot(x="Pclass", y="Survived", hue = "Sex", data=train)


# ## Dealing with missing values
# 
# #### Drop 
# 
# PassengerID
# 
# Name
# 
# Ticket
# 
# Cabin

# In[ ]:


columns = ['Name', 'Ticket', 'Cabin']
train.drop(columns, axis =1, inplace = True)
test.drop(columns, axis =1, inplace = True)


# In[ ]:


train.dropna(subset = ['Embarked'], how = 'all', inplace = True)


# In[ ]:


train['Age'] = train['Age'].fillna((train['Age'].mean()))
test['Age'] = test['Age'].fillna((train['Age'].mean()))
test['Fare'] = test['Fare'].fillna((train['Fare'].mean()))


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False)


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


PassengerId_train = train['PassengerId']
PassengerId_test = test['PassengerId']


# In[ ]:


train.drop('PassengerId', axis =1, inplace = True)
test.drop('PassengerId', axis =1, inplace = True)


# In[ ]:


train


# In[ ]:


test


# In[ ]:


test.info()


# In[ ]:


train.info()


# In[ ]:


test = test.assign(Survived = gender['Survived'])


# ## Assigning the dummies 

# In[ ]:


train_onehot = pd.get_dummies(train)
test_onehot = pd.get_dummies(test)


# In[ ]:


train_onehot


# In[ ]:


test_onehot


# ## Separate the features and targets

# In[ ]:


x_train = train_onehot.drop('Survived', axis =1)
y_train = train_onehot['Survived']


# In[ ]:


x_test = test_onehot.drop('Survived', axis =1)
y_test = test_onehot['Survived']


# ## Prediction Model
# 
# => Logistic Regression
# 
# => KNN or k-Nearest Neighbors
# 
# => Support Vector Machines
# 
# => Naive Bayes Classifier
# 
# => Decision Tree
# 
# => Random Forest
# 
# => Perceptron
# 
# => Artificial Neural Network
# 
# => RVM or Relevance Vector Machine

# ## Import different mathematical modules

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# ## Logistic Regression

# In[ ]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
score_log = logreg.score(x_test,y_test)
score_log


# ## Manually checking the method

# In[ ]:


y_pred


# In[ ]:


y_pred == y_test


# In[ ]:


np.sum((y_pred==y_test))


# In[ ]:


y_pred.shape[0]


# In[ ]:


accuracy = np.sum((y_pred == y_test))/y_pred.shape[0]
accuracy


# ## Support Vector Machines

# In[ ]:


svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
score_svc = svc.score(x_test,y_test)
score_svc


# ## KNN

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
score_knn = knn.score(x_test,y_test)
score_knn


# ## Gaussian Naive Bayes

# In[ ]:


gNB = GaussianNB()
gNB.fit(x_train,y_train)
y_pred = gNB.predict(x_test)
score_gNB = gNB.score(x_test,y_test)
score_gNB


# ## Percpetron

# In[ ]:


perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_test)
score_perceptron = perceptron.score(x_test,y_test)
score_perceptron


# ## Linear SVC

# In[ ]:


linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_test)
score_linear_svc = linear_svc.score(x_test,y_test)
score_linear_svc


# ## Stochastic Gradient Descent

# In[ ]:


sgd = SGDClassifier()
sgd.fit(x_train,y_train)
y_pred = sgd.predict(x_test)
score_sgd = sgd.score(x_test,y_test)
score_sgd


# ## Decision Tree

# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train,y_train)
y_pred = decision_tree.predict(x_test)
score_decision_tree = decision_tree.score(x_test,y_test)
score_decision_tree


# ## Random Forest

# In[ ]:


random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
score_random_forest = random_forest.score(x_test,y_test)
score_random_forest


# # Summary

# In[ ]:


models = pd.DataFrame({
    'Model' : ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Descent', 'Linear SVC', 'Decision Tree'], 
    'Score' : [score_svc, score_knn, score_log, score_random_forest, score_gNB, score_perceptron,
               score_sgd, score_linear_svc, score_decision_tree]})
models.sort_values(by = 'Score', ascending = False)


# # Conclusion
# 
# Logistic Regression model provides the best accuracy among all. Now inserting PassengerId column back.

# In[ ]:


logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
score_log = logreg.score(x_test,y_test)
score_log


# In[ ]:


train = train.assign(PassengerId = PassengerId_train)
test = test.assign(PassengerId = PassengerId_test)


# In[ ]:


submission = pd.DataFrame({"PassengerId": test["PassengerId"],
                          "Survived": y_pred})


# In[ ]:


submission.info()


# In[ ]:


submission.to_csv('submission.csv',index = False)


# In[ ]:




