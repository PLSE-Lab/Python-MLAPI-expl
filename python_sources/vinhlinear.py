# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import csv as csv
from sklearn.linear_model import LinearRegression

df = pd.read_csv('../input/train.csv')
#load test data
td= pd.read_csv('../input/test.csv');
#convert Gender into binary
df['Gender'] = df['Sex'].map( {'female':0, 'male':1} ).astype(int)
td['Gender'] = df['Sex'].map( {'female':0, 'male':1} ).astype(int)
#find median age depend 
median_ages = np.zeros((2,3))
for i in range(0,2): #Gender
    for j in range(0,3): #Pclass
        median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()
        
df['AgeFill'] = df['Age']
td['AgeFill'] = td['Age']

#fill missing age
for i in range(0,2):
    for j in range(0,3):
        df.loc[(df['Age'].isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]
        td.loc[(td['Age'].isnull()) & (td.Gender == i) & (td.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

#df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
#td['AgeIsNull'] = pd.isnull(td.Age).astype(int)

#df['FamilySize'] = df['SibSp'] * df['Parch']
#td['FamilySize'] = td['SibSp'] * td['Parch']

#df['Age*Pclass'] = df['AgeFill'] * df['Pclass']
#td['Age*Pclass'] = td['AgeFill'] * td['Pclass']
df = df.drop(['Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked', 'SibSp', 'Parch', 'Fare'], axis=1)
td = td.drop(['Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked', 'SibSp', 'Parch', 'Fare'], axis=1)
#td.loc[(td['Fare'].isnull()), 'Fare'] = td['Fare'].median()

#random forest
train_data = df.values
linear = LinearRegression()
linear = linear.fit(train_data[0::,2::], train_data[0::,1])
test_data = td.values
output = linear.predict(test_data[0:,1:])
fileout = open('genderclassmodel.csv', 'w', newline='')
prediction_file = csv.writer(fileout)
prediction_file.writerow(['PassengerId', 'Survived'])
for i in range(0,len(output)):
    prediction_file.writerow([int(test_data[i,0]), int(output[i])])
    
fileout.close()