# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 01:21:55 2016

@author: Michael McMullin
"""

import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

def clean_data(df):
    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    df.loc[df['Embarked'].isnull(), 'Embarked'] = 'S'
    df['EmbarkedIndex'] = df['Embarked'].map( {'C': 0, 'S': 1, 'Q': 2} ).astype(int)
  
    median_ages = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i,j] = df[(df['Gender'] == i) & \
                                  (df['Pclass'] == j+1)]['Age'].dropna().median()
                                 
    df['AgeFill'] = df['Age']
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
                    'AgeFill'] = median_ages[i,j]
    
    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
    
    # Feature Engineering
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['Age*Class'] = df.AgeFill * df.Pclass
    
    if len(df.Fare[ df.Fare.isnull() ]) > 0:
        median_fare = np.zeros(3)
        for f in range(0,3):                                              # loop 0 to 2
            median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()
        for f in range(0,3):                                              # loop 0 to 2
            df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]
    return df
    


# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('../input/train.csv', header=0)
df = clean_data(df)

# Drop unneeded columns
df = df.drop(['PassengerId', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked'], axis=1)


# Take the same decision trees and run it on the test data
df_test = pd.read_csv('../input/test.csv', header=0)
df_test = clean_data(df_test)

# Drop unneeded columns
ids = df_test['PassengerId'].values
df_test = df_test.drop(['PassengerId', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked'], axis=1)

# Convert to array
train_data = df.values
test_data = df_test.values


print ('Training...')
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print ('Predicting...')
output = forest.predict(test_data).astype(int)


predictions_file = open("myfirstforest.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print ('Done.')