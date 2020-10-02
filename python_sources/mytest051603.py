# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import AdaBoostClassifier

os.getcwd()
# read in training set
trainData = pd.read_csv('../input/train.csv', header = 0)

# Assign class labels to an array

# change sex variable to ints
trainData['Sex'] = trainData['Sex'].map({'female':0, 'male':1}).astype('int')
# change embarked variable to ints
trainData['Embarked'] = trainData['Embarked'].map({'C':0, 'Q':1, 'S':2})
# drop irrelevant columns 
trainData = trainData.drop(['Cabin', 'Name', 'Ticket'], axis = 1)

   

# create empty array which will hold mean ages for each gender of each class
mean_age = trainData['Age'].dropna().mean()
  
# create a new column to modify ages of missing values                                  
trainData['modAge'] = trainData['Age']
# populate new column
trainData.loc[(trainData.Age.isnull()),'modAge'] = mean_age
                       
                       

# drop the age column
trainData = trainData.drop(['Age'], axis = 1)





# drop na's from embarked (2 of them)
trainData = trainData.dropna()
# feature engineer family size
trainData['FamilySize'] = trainData['SibSp'] + trainData['Parch']
# drop sibsp and parch
trainData = trainData.drop(['Parch', 'SibSp'], axis = 1)
# remove passengerID but keep for reference if needed
trainPassenger = trainData['PassengerId']
trainData = trainData.drop(['PassengerId'], axis = 1)
Survived = trainData['Survived']
trainData = trainData.drop(['Survived'], axis = 1)

# fit ensemble method to train data
# set classifier
ada = AdaBoostClassifier(n_estimators = 100)
# fit model
model = ada.fit(trainData, Survived)

# read in test data
testData = pd.read_csv('../input/test.csv', header = 0)

# change sex variable to int
testData['Sex'] = testData['Sex'].map({'female':0, 'male':1}).astype('int')
# change embarked variable to ints
testData['Embarked'] = testData['Embarked'].map({'C':0, 'Q':1, 'S':2}).astype('int')
# drop irrelevant columns 
testData = testData.drop(['Cabin', 'Name', 'Ticket'], axis = 1)



# create a new column to modify ages of missing values                                  
testData['modAge'] = testData['Age']
# populate new column
testData.loc[(testData.Age.isnull()),'modAge'] = mean_age
                        


# drop age column
testData = testData.drop(['Age'], axis = 1)





# feature engineer family size
testData['FamilySize'] = testData['Parch'] + testData['SibSp']
# drop sibsp and parch
testData = testData.drop(['Parch', 'SibSp'], axis = 1)
# save passengerid for submit
testPassenger = testData['PassengerId']
# drop passengerid 
testData = testData.drop(['PassengerId'], axis = 1)
# fill na for fare as mean of column data
mean_fare = trainData[(trainData['Pclass'] == 3)]['Fare'].mean()
testData['Fare'].fillna(mean_fare,inplace = True)

# predict test data
predModel = ada.predict(testData)

# merge passengerid with predicted model array
submit = testData
submit['PassengerId'] = testPassenger
submit['Survived'] = predModel
submit = submit.drop(['Pclass', 'Sex', 'Fare', 'Embarked', 'modAge', 'FamilySize'], axis = 1)
# write to file
submit.to_csv('submit.csv')
