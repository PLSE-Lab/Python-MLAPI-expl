import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

#predict missing age values

def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    age_mapping = {'Unknown': 1, 'Baby': 2, 'Child': 3, 'Teenager': 4, 'Student': 5, 'Young Adult': 6, 'Adult': 7,
                   'Senior': 8}
    df.Age = df.Age.map(age_mapping)
    return df
train_data = simplify_ages(train_data)
test_data = simplify_ages(test_data)


#Convert sex categorical variable to numeric
train_data['Sex'] = train_data['Sex'].map({'female' : 0,'male':1})

# Filling Embarked missing values

if ((train_data[train_data["Embarked"] == "S"].shape[0]) >= (train_data[train_data["Embarked"] == "C"].shape[0])) and\
        ((train_data[train_data["Embarked"] == "S"].shape[0]) >= (train_data[train_data["Embarked"] == "Q"].shape[0])):
    train_data = train_data.fillna({'Embarked': 'S'})
elif ((train_data[train_data["Embarked"] == "Q"].shape[0]) >= (train_data[train_data["Embarked"] == "C"].shape[0])) and\
        ((train_data[train_data["Embarked"] == "Q"].shape[0]) >= (train_data[train_data["Embarked"] == "S"].shape[0])):
        train_data = train_data.fillna({'Embarked': 'Q'})
else:
    train_data = train_data.fillna({'Embarked': 'C'})

train_data['Embarked'] = train_data['Embarked'].map({'S':1,'Q':2,'C':3})

train_data.drop(['Cabin','Name','PassengerId','Ticket'],1,inplace=True)

test_data = test_data.fillna({'Fare':test_data.Fare.mean()})


#Splitting and testing the moodel
from sklearn.model_selection import train_test_split

predictors = train_data.drop(['Survived'], axis=1)
target = train_data["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.20, random_state = 6)


#Testing with different models


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)

#print(acc_logreg)

# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
#print(acc_svc)


# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
#print(acc_linear_svc)


# Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
#print(acc_perceptron)


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
#print(acc_decisiontree)


# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
#print(acc_randomforest)


# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
#print(acc_knn)


# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
#print(acc_sgd)


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
#print(acc_gbk)


#Let's compare the accuracies of each model!

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC',
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg,
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})

print(models.sort_values(by='Score', ascending=False))

test_data.drop(['Cabin','Name','Ticket'],1,inplace=True)
test_data['Sex'] = test_data['Sex'].map({'female':0,'male':1})
test_data['Embarked'] = test_data['Embarked'].map({'S':1,'Q':2,'C':3})

# We use logistic Regression with accuracy of 82%
predict = logreg.predict(test_data.drop('PassengerId', axis=1))
submission = pd.DataFrame({'PassengerId':test_data['PassengerId'],
                           'Survived' : predict})
submission.to_csv('submission.csv',index=False)
