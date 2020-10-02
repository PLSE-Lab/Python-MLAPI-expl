# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 22:05:40 2015

@author: Yang Jiao
"""

import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

traindf = pd.read_csv('train.csv', header=0)
testdf = pd.read_csv('test.csv', header=0)

# Convert gender to integer value
traindf['Gender'] = traindf['Sex'].map({'male':1, 'female':0}).astype(int)
testdf['Gender'] = testdf['Sex'].map({'male':1, 'female':0}).astype(int)

# Fill in mission Embarked value
if len(traindf.Embarked[ traindf.Embarked.isnull() ]) > 0:
    traindf.Embarked[ traindf.Embarked.isnull() ] = \
    traindf.Embarked.dropna().mode().values
    
if len(testdf.Embarked[ testdf.Embarked.isnull() ]) > 0:
    testdf.Embarked[ testdf.Embarked.isnull() ] = \
    testdf.Embarked.dropna().mode().values
    
# Convert Embarked to integer value
Ports = list(enumerate(np.unique(traindf['Embarked'])))    
Ports_dict = { name : i for i, name in Ports }              
traindf.Embarked = traindf.Embarked.map( lambda x: Ports_dict[x]).astype(int) 

Ports = list(enumerate(np.unique(testdf['Embarked'])))    
Ports_dict = { name : i for i, name in Ports }              
testdf.Embarked = testdf.Embarked.map( lambda x: Ports_dict[x]).astype(int)    

## Add a column for missing age value
#testdf['AgePredicted'] = testdf['Age']
#traindf['AgePredicted'] = traindf['Age']

## Fill in missing age with median value with regard to their class
#if len(traindf.Age[ traindf.Age.isnull() ]) > 0:
#    median_ages_train = np.zeros((2,3))
#    for i in range(0, 2):
#        for j in range(0, 3):
#            median_ages_train[i,j] = traindf[(traindf['Gender'] == i) & \
#              (traindf['Pclass'] == j+1)]['Age'].dropna().median()
#    for i in range(0, 2):
#        for j in range(0, 3):
#            traindf.loc[ (traindf.Age.isnull()) & \
#                (traindf.Gender == i) & (traindf.Pclass == j+1),\
#                'AgePredicted'] = median_ages_train[i,j]
#
#if len(testdf.Age[ testdf.Age.isnull() ]) > 0:
#    median_ages_test = np.zeros((2,3))
#    for i in range(0, 2):
#        for j in range(0, 3):
#            median_ages_test[i,j] = testdf[(testdf['Gender'] == i) & \
#              (testdf['Pclass'] == j+1)]['Age'].dropna().median()
#    for i in range(0, 2):
#        for j in range(0, 3):
#            testdf.loc[ (testdf.Age.isnull()) & \
#                (testdf.Gender == i) & (testdf.Pclass == j+1),\
#                'AgePredicted'] = median_ages_test[i,j]

median_age = traindf['Age'].dropna().median()
if len(traindf.Age[ traindf.Age.isnull() ]) > 0:
    traindf.loc[ (traindf.Age.isnull()), 'Age'] = median_age
  
median_age = testdf['Age'].dropna().median()
if len(testdf.Age[ testdf.Age.isnull() ]) > 0:
    testdf.loc[ (testdf.Age.isnull()), 'Age'] = median_age  
  

# Fill in all missing Fare with median value
if len(testdf.Fare[ testdf.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              
        median_fare[f] = testdf[ testdf.Pclass == f+1 ]['Fare'].\
        dropna().median()
    for f in range(0,3):                                              
        testdf.loc[ (testdf.Fare.isnull()) & (testdf.Pclass == f+1 ), 'Fare']\
        = median_fare[f]
        
if len(traindf.Fare[ traindf.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              
        median_fare[f] = traindf[ traindf.Pclass == f+1 ]['Fare'].\
        dropna().median()
    for f in range(0,3):                                              
        traindf.loc[ (traindf.Fare.isnull()) & (traindf.Pclass == f+1 ), 'Fare']\
        = median_fare[f]

# So we add a ceiling
fare_max = 40
fare_bracket_size = 10
# Set max fare
traindf[ traindf.Fare >= fare_max]['Fare'] = fare_max - 1.0
testdf[testdf.Fare >= fare_max]['Fare'] = fare_max - 1.0

#traindf['FareBin'] = traindf['Fare']
#testdf['FareBin'] = traindf['Fare']
#
#for i in range(0, int(fare_max/fare_bracket_size) - 1):
#        traindf.loc[(traindf.Fare >= float(fare_bracket_size *i)) &\
#    (traindf.Fare < float(fare_bracket_size * (i+1))) ]['FareBin'] = i
#    
#for i in range(0, int(fare_max/fare_bracket_size) - 1):
#        testdf.loc[(testdf.Fare >= float(fare_bracket_size *i)) &\
#    (testdf.Fare < float(fare_bracket_size * (i+1)))]['FareBin'] = i


## Add a column to tell if the age was null
#testdf['AgeIsNull'] = pd.isnull(testdf.Age).astype(int)
#traindf['AgeIsNull'] = pd.isnull(traindf.Age).astype(int)

## Add family attribute to each passenger
#testdf['FamilySize'] = testdf['SibSp'] + testdf['Parch']
#traindf['FamilySize'] = traindf['SibSp'] + traindf['Parch']

## Add class and age factore
#testdf['Age*Class'] = testdf.AgePredicted * testdf.Pclass
#traindf['Age*Class'] = traindf.AgePredicted * traindf.Pclass

# Get passenger id before dropping the column
ids = testdf['PassengerId'].values

#Drop unused columns  
traindf = traindf.drop(['Name','Sex','Ticket','Cabin','SibSp',\
'Parch','Age','Embarked',\
'PassengerId'], axis=1)  

testdf = testdf.drop(['Name','Sex','Ticket','Cabin','SibSp',\
'Parch','Age','Embarked',\
'PassengerId'], axis=1)  

# Convert back to a numpy array
train_data = traindf.values
test_data = testdf.values

print ('Training...')
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print ('Predicting...')
output = forest.predict(test_data).astype(int)


predictions_file = open("TitanicPredict_RandomForest8.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print ('Done.')
