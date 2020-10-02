#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster

# Inspired by the following kernel(s):
# * https://www.kaggle.com/startupsci/titanic-data-science-solutions

# ## 1. Import Necessary Python Packages

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## 2. Open Dataset

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_combined = pd.concat([df_train, df_test], sort=False)


# ## 2.1. Browse through the training data

# In[ ]:


df_train.head()


# In[ ]:


df_train.describe()


# In[ ]:


df_test.head()


# ## 2.2. Browse through the test data

# In[ ]:


df_test.describe()


# ## 3. Feature Analysis

# ## 3.1. Survival rate by Pclass

# Is there any correlation between Pclass and survival rate?

# In[ ]:


df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by="Survived", ascending=False)


# ## 3.2. Survival rate by Sex (Gender)

# Is there any correlation between Parch and survival rate?

# In[ ]:


df_train[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by="Survived", ascending=False)


# ## 3.3. Survival rate by SibSp

# Is there any correlation between SibSp and survival rate?

# In[ ]:


df_train[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by="Survived", ascending=False)


# ## 3.4. Survival rate by Parch

# Is there any relationship between Parch and survival rate?

# In[ ]:


df_train[['Parch','Survived']].groupby(['Parch'], as_index=False).maean().sort_values(by="Survived", ascending=False)


# # 4. Feature Engineering

# ## 4.1. Survival rate by Title

# # 4.1.1. Obtain Title from Name

# In[ ]:


df_train['Title'] = df_train['Name'].str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(df_train['Title'], df_train['Sex'])


# In[ ]:


df_test['Title'] = df_test['Name'].str.extract('([A-Za-z]+)\.', expand=False)


# # 4.1.2. Consolidate Title

# Consolidate Title with modern sounding ones otherwise classify as Rare

# In[ ]:


df_train['Title'] = df_train['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Jonkheer', 'Sir', 'Rev', 'Dona'], 'Rare')
df_train['Title'] = df_train['Title'].replace(['Lady', 'Countess', 'Mme'], 'Mrs')
df_train['Title'] = df_train['Title'].replace(['Ms', 'Mlle'], 'Miss')


# In[ ]:


df_test['Title'] = df_test['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Jonkheer', 'Sir', 'Rev', 'Dona'], 'Rare')
df_test['Title'] = df_test['Title'].replace(['Lady', 'Countess', 'Mme'], 'Mrs')
df_test['Title'] = df_test['Title'].replace(['Ms', 'Mlle'], 'Miss')


# # 4.1.3. Check survival rate

# In[ ]:


df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# ## 4.2. Survival rate by Family Size

# In[ ]:


df_train['FamSize'] = df_train['Parch'] + df_train['SibSp'] + 1 
df_test['FamSize'] = df_test['Parch'] + df_test['SibSp'] + 1


# In[ ]:


df_train.head()


# In[ ]:


df_train[['FamSize','Survived']].groupby(['FamSize'], as_index=False).mean().sort_values(by="Survived", ascending=False)


# ## 4.3. Survival rate by being a Sole or Group Traveller

# Does travelling alone or in a group matter when it comes to survival?

# In[ ]:


df_train['IsAlone'] = df_train["FamSize"].map(lambda s: 1 if s==1 else 0)
df_test['IsAlone'] = df_test["FamSize"].map(lambda s: 1 if s==1 else 0)


# In[ ]:


df_train[['IsAlone','Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by="Survived", ascending=False)


# ## 4.4. Survival rate by Embarked Port

# Does port of embarkation matter when it comes to survival?

# In[ ]:


df_train[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by="Survived", ascending=False)


# ## 4.5. Categorize Fare

# In[ ]:


df_train['Fare'] = df_train['Fare'].fillna(df_train['Fare'].median())
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())


# In[ ]:


df_train['CategoricalFare'] = pd.qcut(df_train['Fare'], 10)
# df_test['CategoricalFare'] = pd.qcut(df_test['Fare'], 10)


# In[ ]:


print (df_train[['CategoricalFare','Survived']].groupby(['CategoricalFare'], as_index=False).mean().sort_values(by="Survived", ascending=False))


# ## 4.6. Categorize Age

# In[ ]:


age_median = df_combined['Age'].median()

age_median


# In[ ]:


age_mean = df_combined['Age'].mean()

age_mean


# In[ ]:


age_std = df_combined['Age'].std()

age_std


# In[ ]:


age_null_count = df_train['Age'].isnull().sum()

age_null_count


# In[ ]:


df_train['Age'] = df_train['Age'].fillna(age_median)
df_train['CategoricalAge'] = pd.cut(df_train['Age'], 8)

print (df_train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())


# In[ ]:


df_test['Age'] = df_test['Age'].fillna(age_median)
# df_test['CategoricalAge'] = pd.cut(df_test['Age'], 8) # no 'cutting' on test data


# # 5. Data Cleaning

# ## 5.1. Map Title to Ordinal

# In[ ]:


df_train['Title'] = df_train['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 0}).astype(int)


# In[ ]:


df_train.head()


# In[ ]:


df_test['Title'] = df_test['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 0}).astype(int)


# In[ ]:


df_test.head()


# ## 5.2. Map Sex (Gender) to Ordinal

# In[ ]:


df_train['Sex'] = df_train['Sex'].map({'female' : 0, 'male' : 1}).astype(int)
df_train.head()


# In[ ]:


df_test['Sex'] = df_test['Sex'].map({'female' : 0, 'male' : 1}).astype(int)
df_test.head()


# # 5.3. Map Embarked to Ordinal

# In[ ]:


df_train.loc[df_train['Embarked'] == 'Empty']
df_train['Embarked'] = df_train['Embarked'].fillna("Empty")

df_train['Embarked'] = df_train['Embarked'].map({'S' : 0, 'C' : 1, 'Q' : 2, 'Empty' : 0}, na_action='ignore').astype(int)

df_train.head()


# In[ ]:


df_test['Embarked'] = df_test['Embarked'].fillna("Empty")
df_test.loc[df_test['Embarked'] == 'Empty']
df_test['Embarked'] = df_test['Embarked'].map({'S' : 0, 'C' : 1, 'Q' : 2, 'Empty' : 0}, na_action='ignore').astype(int)

df_test.head()


# ## 5.4. Map CategoricalAge to Ordinal

# In[ ]:


df_train['CategoricalAge'] = 0

df_train.loc[df_train['Age']  <= 10.368, 'CategoricalAge'] = 0
df_train.loc[(df_train['Age']  > 10.368) & (df_train['Age'] <= 20.315), 'CategoricalAge'] = 1
df_train.loc[(df_train['Age']  > 20.315) & (df_train['Age'] <= 30.263), 'CategoricalAge'] = 2
df_train.loc[(df_train['Age']  > 30.263) & (df_train['Age'] <= 40.21), 'CategoricalAge'] = 3
df_train.loc[(df_train['Age']  > 40.21) & (df_train['Age'] <= 50.158), 'CategoricalAge'] = 4
df_train.loc[(df_train['Age']  > 50.158) & (df_train['Age'] <= 60.105), 'CategoricalAge'] = 5
df_train.loc[(df_train['Age']  > 60.105) & (df_train['Age'] <= 70.052), 'CategoricalAge'] = 6
df_train.loc[df_train['Age']  > 70.052, 'CategoricalAge'] = 7

df_train.head()


# In[ ]:


df_test['CategoricalAge'] = 0

df_test.loc[df_test['Age']  <= 10.368, 'CategoricalAge'] = 0
df_test.loc[(df_test['Age']  > 10.368) & (df_test['Age'] <= 20.315), 'CategoricalAge'] = 1
df_test.loc[(df_test['Age']  > 20.315) & (df_test['Age'] <= 30.263), 'CategoricalAge'] = 2
df_test.loc[(df_test['Age']  > 30.263) & (df_test['Age'] <= 40.21), 'CategoricalAge'] = 3
df_test.loc[(df_test['Age']  > 40.21) & (df_test['Age'] <= 50.158), 'CategoricalAge'] = 4
df_test.loc[(df_test['Age']  > 50.158) & (df_test['Age'] <= 60.105), 'CategoricalAge'] = 5
df_test.loc[(df_test['Age']  > 60.105) & (df_test['Age'] <= 70.052), 'CategoricalAge'] = 6
df_test.loc[df_test['Age']  > 70.052, 'CategoricalAge'] = 7

df_test.head()


# ## 5.5. Map Categorical Fare to Ordinal

# In[ ]:


df_train['CategoricalFare'] = 0

df_train.loc[df_train['Fare'] <= 7.55, 'CategoricalFare'] = 0
df_train.loc[(df_train['Fare'] > 7.55) & (df_train['Fare'] <= 7.854 ), 'CategoricalFare'] = 1
df_train.loc[(df_train['Fare'] > 7.854) & (df_train['Fare'] <= 8.05 ), 'CategoricalFare'] = 2
df_train.loc[(df_train['Fare'] > 8.05) & (df_train['Fare'] <= 10.5 ), 'CategoricalFare'] = 3
df_train.loc[(df_train['Fare'] > 10.5) & (df_train['Fare'] <= 14.454 ), 'CategoricalFare'] = 4
df_train.loc[(df_train['Fare'] > 14.454) & (df_train['Fare'] <= 21.679 ), 'CategoricalFare'] = 5
df_train.loc[(df_train['Fare'] > 21.679) & (df_train['Fare'] <= 27.0 ), 'CategoricalFare'] = 6
df_train.loc[(df_train['Fare'] > 27.0) & (df_train['Fare'] <= 39.688 ), 'CategoricalFare'] = 7
df_train.loc[(df_train['Fare'] > 39.688) & (df_train['Fare'] <= 77.958 ), 'CategoricalFare'] = 8
df_train.loc[df_train['Fare'] > 77.958, 'CategoricalFare'] = 9

df_train.head()


# In[ ]:


df_test['CategoricalFare'] = 0

df_test.loc[df_test['Fare'] <= 7.55, 'CategoricalFare'] = 0
df_test.loc[(df_test['Fare'] > 7.55) & (df_test['Fare'] <= 7.854 ), 'CategoricalFare'] = 1
df_test.loc[(df_test['Fare'] > 7.854) & (df_test['Fare'] <= 8.05 ), 'CategoricalFare'] = 2
df_test.loc[(df_test['Fare'] > 8.05) & (df_test['Fare'] <= 10.5 ), 'CategoricalFare'] = 3
df_test.loc[(df_test['Fare'] > 10.5) & (df_test['Fare'] <= 14.454 ), 'CategoricalFare'] = 4
df_test.loc[(df_test['Fare'] > 14.454) & (df_test['Fare'] <= 21.679 ), 'CategoricalFare'] = 5
df_test.loc[(df_test['Fare'] > 21.679) & (df_test['Fare'] <= 27.0 ), 'CategoricalFare'] = 6
df_test.loc[(df_test['Fare'] > 27.0) & (df_test['Fare'] <= 39.688 ), 'CategoricalFare'] = 7
df_test.loc[(df_test['Fare'] > 39.688) & (df_test['Fare'] <= 77.958 ), 'CategoricalFare'] = 8
df_test.loc[df_test['Fare'] > 77.958, 'CategoricalFare'] = 9

df_test.head()


# # 6. Feature Selection

# In[ ]:


drop_elements = ['Name', 'Ticket', 'Cabin', 'Age', 'Fare', 'Parch', 'SibSp', 'IsAlone']
df_train = df_train.drop(drop_elements, axis = 1)
df_test = df_test.drop(drop_elements, axis = 1)


# In[ ]:


df_train.head()


# # 7. Data Scaling

# ## 7.1. Prepare Data for Scaling

# In[ ]:


train = df_train.copy()
train = train.drop('PassengerId', axis=1)

train.head()


# In[ ]:


test = df_test.copy()
test = test.drop('PassengerId', axis=1)

test.head()


# ## 7.2. Import Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler


# ## 7.2.1. Consolidate Data

# In[ ]:


X_train = train.drop('Survived', axis=1).astype(float)
y_train = df_train['Survived']

X_test = test.astype(float)


# ## 7.3. Perform Scaling

# In[ ]:


scaler = StandardScaler()
scaler.fit(X_train)


# ## 7.4. Check Data shape for sanity

# In[ ]:


X_train = scaler.transform(X_train)


# In[ ]:


X_test = scaler.transform(X_test)


# In[ ]:


print ("X_train shape: ", str(X_train.shape))


# In[ ]:


print ("y_train shape: ", str(y_train.shape))


# In[ ]:


print ("X_test shape: ", str(X_test.shape))


# # 8. Classification

# ## 8.1. Import Models

# In[ ]:


# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


# ## 8.2. Training (fit) and Inference (predict)

# ### 8.2.1. Logistic Regression

# In[ ]:


logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, y_train) * 100, 3)

acc_log


# ### 8.2.2. Support Vector Machine (SVM)

# In[ ]:


svc = SVC(gamma='scale')
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100, 3)

acc_svc


# ### 8.2.3. k-Nearest Neighbor (kNN)

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 3)

acc_knn


# ### 8.2.4. Gaussian Naive Bayes

# In[ ]:


gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 3)

acc_gaussian


# ### 8.2.5. Perceptron

# In[ ]:


perceptron = Perceptron(max_iter=1000, tol=0.001)
perceptron.fit(X_train, y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 3)

acc_perceptron


# ### 8.2.6. Linear SVC

# In[ ]:


linear_svc = LinearSVC(max_iter=10000)
linear_svc.fit(X_train, y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 3)

acc_linear_svc


# ### 8.2.7. Stochastic Gradient Descent (SGD)

# In[ ]:


sgd = SGDClassifier(max_iter=10000, tol=0.001)
sgd.fit(X_train, y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, y_train) * 100, 3)

acc_sgd


# ### 8.2.8. Decision Tree

# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 3)

acc_decision_tree


# ### 8.2.9. Random Forest

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 3)

acc_random_forest


# ### 8.2.10. Multi Layer Perceptron (Neural Networks)

# In[ ]:


MLP_clf = MLPClassifier(activation='tanh', solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(50,50), random_state=1, warm_start=True)
MLP_clf.fit(X_train, y_train)
Y_pred = MLP_clf.predict(X_test)
acc_MLP_clf = round(MLP_clf.score(X_train, y_train) * 100, 3)

acc_MLP_clf


# ### 8.2.11. AdaBoost Classifier

# In[ ]:


adaboost = AdaBoostClassifier()
adaboost.fit(X_train, y_train)
Y_pred = adaboost.predict(X_test)
acc_adaboost = round(adaboost.score(X_train, y_train) * 100, 3)

acc_adaboost


# ### 8.2.12. Gradient Boosting Classifier

# In[ ]:


gboost = GradientBoostingClassifier()
gboost.fit(X_train, y_train)
Y_pred = gboost.predict(X_test)
acc_gboost = round(gboost.score(X_train, y_train) * 100, 3)

acc_gboost


# ## 8.3. Consolidate Score from each model

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'MLPClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree, acc_MLP_clf, acc_adaboost, acc_gboost]})
models.sort_values(by='Score', ascending=False)


# ## 9. Submission

# ## 9.1. Use the best classifier and prepare for Y_pred

# In[ ]:


Y_pred = random_forest.predict(X_test)


# ## 9.2. Prepare submission.csv file

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)


# In[ ]:




