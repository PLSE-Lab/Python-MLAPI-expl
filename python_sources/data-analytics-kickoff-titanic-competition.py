# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# data analysis and wrangling
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

age_Condition = 16
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]

# Removed 'Ticket' and 'Cabin' as they are not useful for predction
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

# Adding Title column
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# Integer values for Title column
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# Removed 'Ticket' and 'Cabin' as they are not useful for predction
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
    
# Random values for NULL Age for dataset in combine:
for dataset in combine:
    average_age   = dataset["Age"].mean()
    std_age      = dataset["Age"].std()
    count_nan_age = dataset["Age"].isnull().sum()
    
    random_age = np.random.randint(average_age - std_age, average_age + std_age, size = count_nan_age)
    
    dataset["Age"][np.isnan(dataset["Age"])] = random_age

    dataset['Age'] = dataset['Age'].astype(int)

# FamilySize 
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']

#Family and Age <= age_Condition
for dataset in combine:
    dataset.loc[dataset['FamilySize'] == 0, 'Family'] = 0
    dataset.loc[dataset['FamilySize'] > 0, 'Family'] = 1
    dataset.loc[(dataset['Age'] <= age_Condition) & (dataset['Family'] == 0), 'Family'] = 2
    dataset.loc[(dataset['Age'] <= age_Condition) & (dataset['Family'] == 1), 'Family'] = 3
    dataset.loc[(dataset['Title'] == 4) & (dataset['Family'] == 0), 'Family'] = 2
    dataset.loc[(dataset['Title'] == 4) & (dataset['Family'] == 1), 'Family'] = 3
    
    dataset['Family'] = dataset['Family'].astype(int)
    
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

#Mode for Embarked
freq_port = train_df.Embarked.dropna().mode()[0]

# Replace NULL Embarked with mode
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# Integer for Embarked
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# Considering kids < 16 years in Embarked column
for dataset in combine:
    dataset.loc[ dataset['Age'] <= age_Condition, 'Embarked'] = 3
    dataset.loc[ dataset['Title'] == 4, 'Embarked'] = 3

# Replace Fare NULL with median
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

# Range for Fare
for dataset in combine:
    dataset['Fare'] = dataset['Fare'].astype(int)

# Integer values for Sex
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
# Considering kids < 16 years in Age column
for dataset in combine:
    dataset.loc[ dataset['Age'] <= age_Condition, 'Sex'] = 2
    dataset.loc[ dataset['Title'] == 4, 'Sex'] = 2

Dataset_train = train_df.drop("Survived", axis=1)
Prediction_train = train_df["Survived"]
Dataset_test  = test_df.drop("PassengerId", axis=1).copy()

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(Dataset_train, Prediction_train)
random_forest_prediction = random_forest.predict(Dataset_test)
random_forest_score = random_forest.score(Dataset_train, Prediction_train) * 100

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(Dataset_train, Prediction_train)
decision_tree_prediction = decision_tree.predict(Dataset_test)
decision_tree_score = decision_tree.score(Dataset_train, Prediction_train) * 100

# Support Vectore Machine
svc = SVC()
svc.fit(Dataset_train, Prediction_train)
svc_prediction = svc.predict(Dataset_test)
svc_score = svc.score(Dataset_train, Prediction_train) * 100

# k-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(Dataset_train, Prediction_train)
knn_prediction = knn.predict(Dataset_test)
knn_score = knn.score(Dataset_train, Prediction_train) * 100

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(Dataset_train, Prediction_train)
sgd_prediction = sgd.predict(Dataset_test)
sgd_score = sgd.score(Dataset_train, Prediction_train) * 100

# Perceptron
perceptron = Perceptron()
perceptron.fit(Dataset_train, Prediction_train)
perceptron_prediction = perceptron.predict(Dataset_test)
perceptron_score = perceptron.score(Dataset_train, Prediction_train) * 100

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(Dataset_train, Prediction_train)
gaussian_prediction = gaussian.predict(Dataset_test)
gaussian_score = gaussian.score(Dataset_train, Prediction_train) * 100

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(Dataset_train, Prediction_train)
logreg_prediction = logreg.predict(Dataset_test)
logreg_score = logreg.score(Dataset_train, Prediction_train) * 100

# Model comparison 
models = pd.DataFrame({
    'Model': ['Random Forest','Decision Tree', 'Support Vectore Machine', 
                'k-Nearest Neighbors', 'Stochastic Gradient Descent', 'Perceptron', 
                'Gaussian Naive Bayes', 'Logistic Regression'],
    'Score': [random_forest_score, decision_tree_score, svc_score, 
                knn_score, sgd_score, perceptron_score, 
                gaussian_score, logreg_score]})
models.sort_values(by='Score', ascending=False)

print(models);

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": random_forest_prediction
    })
    
submission.to_csv('submission.csv', index=False)