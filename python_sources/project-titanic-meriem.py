# %% [code]
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 17:53:16 2020

@author: Meriem
"""
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
#ignore warnings
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#get a list of the features within the dataset
print(train.columns)
#check for any other unusable values
print(pd.isnull(train).sum())
#sort the ages into logical categories
train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()
#we'll start off by dropping the Cabin feature since not a lot more useful information can be extracted from it.
train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)
#we can also drop the Ticket feature since it's unlikely to yield any useful information
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)
#now we need to fill in the missing values in the Embarked feature
print("Number of people embarking in Southampton (S):")
southampton = train[train["Embarked"] == "S"].shape[0]
print(southampton)

print("Number of people embarking in Cherbourg (C):")
cherbourg = train[train["Embarked"] == "C"].shape[0]
print(cherbourg)

print("Number of people embarking in Queenstown (Q):")
queenstown = train[train["Embarked"] == "Q"].shape[0]
print(queenstown)
#create a combined group of both datasets
combine = [train, test]

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])
#map each of the title groups to a numerical value
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()
# fill missing age with mode age group for each title
mr_age = train[train["Title"] == 1]["AgeGroup"].mode() #Young Adult
miss_age = train[train["Title"] == 2]["AgeGroup"].mode() #Student
mrs_age = train[train["Title"] == 3]["AgeGroup"].mode() #Adult
master_age = train[train["Title"] == 4]["AgeGroup"].mode() #Baby
royal_age = train[train["Title"] == 5]["AgeGroup"].mode() #Adult
rare_age = train[train["Title"] == 6]["AgeGroup"].mode() #Adult

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}
train = train.fillna({"Age": train["Title"].map(age_title_mapping)})

test = test.fillna({"Age": test["Title"].map(age_title_mapping)})

#map each Age value to a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

train.head()

#dropping the Age feature for now, might change
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)
#drop the name feature since it contains no more useful information.
train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)
#Sex Feature
#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()
#map each Embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)
#fill in missing Fare value in test set based on mean fare for that Pclass 
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x] #Pclass = 3
        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)
        
#map Fare values into groups of numerical values
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])

#drop Fare values
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)
#check train data
print(train.head(5).to_string())
#=============================================================================================================
#Make train data from train with age information
train_survived = train
modifiedFlights = train_survived.dropna()
#Make test data from train without age information
null_columns=train.columns[train.isnull().any()]
x_train_survived = modifiedFlights.drop(['Survived'], axis = 1)
y_train_survived = modifiedFlights["Survived"]
x_test_survived= train[train.isnull().any(axis=1)]
x_test_survived = x_test_survived.drop(['Survived'], axis = 1)
from sklearn.model_selection import train_test_split
predictors = x_train_survived
target = y_train_survived
x_trainage, x_valage, y_trainage, y_valage = train_test_split(predictors, target, test_size = 0.1, random_state = 0)
#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

gbk = GradientBoostingClassifier()
gbk.fit(x_trainage, y_trainage)
y_predage = gbk.predict(x_valage)
acc_gbkage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_gbkage)
#Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

sgd = SGDClassifier()
sgd.fit(x_trainage, y_trainage)
y_predage = sgd.predict(x_valage)
acc_sgdage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_sgdage)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_trainage, y_trainage)
y_predage = knn.predict(x_valage)
acc_knnage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_knnage)
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_trainage, y_trainage)
y_predage = randomforest.predict(x_valage)
acc_randomforestage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_randomforestage)
#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_trainage, y_trainage)
y_predage = decisiontree.predict(x_valage)
acc_decisiontreeage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_decisiontreeage)
#Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_trainage, y_trainage)
y_predage = perceptron.predict(x_valage)
acc_perceptronage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_perceptronage)
#Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_trainage, y_trainage)
y_predage = linear_svc.predict(x_valage)
acc_linear_svcage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_linear_svcage)
#Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_trainage, y_trainage)
y_predage = svc.predict(x_valage)
acc_svcage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_svcage)
#Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_trainage, y_trainage)
y_predage = logreg.predict(x_valage)
acc_logregage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_logregage)
#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_trainage, y_trainage)
y_predage = gaussian.predict(x_valage)
acc_gaussianage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_gaussianage)
models = pd.DataFrame({
    'Model': ['Gradient Boosting Classifier','Stochastic Gradient Descent','KNN','Random Forest','Decision Tree', 'Perceptron','Linear SVC','Support Vector Machines', 
              'Logistic Regression','Naive Bayes',  
              ],
    'Score': [acc_gbkage, acc_sgdage, acc_knnage, acc_randomforestage, 
              acc_decisiontreeage, acc_perceptronage, acc_linear_svcage, acc_svcage, acc_logregage, 
               acc_gaussianage]})
print(models.sort_values(by='Score', ascending=False))
