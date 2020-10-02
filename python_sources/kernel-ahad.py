import numpy as np 
import pandas as pd 
import seaborn as sns
import random as rnd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import os

#****************************INPUT***********************************************************

train_data=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")
example_data=pd.read_csv("../input/gender_submission.csv")
combine_data=[train_data,test_data]

#Droping Unused features

train_data = train_data.drop(['Name','PassengerId','Ticket', 'Cabin','Fare'], axis=1)
test_data = test_data.drop(['Name','Ticket', 'Cabin','Fare'], axis=1)
combine = [train_data, test_data]


#converting into numerical values

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


#"Filling vacant values "*************inspired from tutorial******************************

guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)


#*********************************  Creating Age_Group (New Feature) *****************************************


train_data['AgeBand'] = pd.cut(train_data['Age'], 5)

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_data = train_data.drop(['AgeBand'], axis=1)

combine = [train_data, test_data]


#********************************** NEW_FEATURE -family size****************************************************************
for dataset in combine:
    dataset['family'] = dataset['SibSp'] + dataset['Parch'] + 1



train_data = train_data.drop(['Parch', 'SibSp', 'family'], axis=1)
test_data = test_data.drop(['Parch', 'SibSp', 'family'], axis=1)
combine = [train_data, test_data]


#**************************************************Finding most frequent Embarked and filling vacant values*********************

freq_port = train_data.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
    
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
combine = [train_data, test_data]

#************************************ Training Data*********************************************************************************
X_train = train_data.drop("Survived", axis=1)
Y_train = train_data["Survived"]
X_test  = test_data.drop("PassengerId", axis=1).copy()


#****************************************applying various model*********************************************************
'''
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
accuracy= 78+

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
accuracy coming 78+'''



dec_tree = DecisionTreeClassifier()
dec_tree.fit(X_train, Y_train)
Y_pred = dec_tree.predict(X_test)
acc_dec_tree = round(dec_tree.score(X_train, Y_train) * 100, 2)
#accuracy coming out to be 82.72

submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": Y_pred
    })
    
#*******************************convert to csv***************************************************************

submission.to_csv('submission.csv',index=False)
