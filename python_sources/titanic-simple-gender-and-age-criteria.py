# Script to practice with the Kaggle Titanic challenge
# Uses simple criteria (female = survived and age<=18 survived)
# Script added to Kaggle by MHardin on 2016-07-26

import pandas as pd
import numpy as np

datadir = '../input/'

df = pd.read_csv(datadir + 'train.csv', header=0) # training data
td = pd.read_csv(datadir + 'test.csv', header=0)  # test data

# Reassign 'Sex' column to ints (from strings)
df['Gender']  = df['Sex'].map( {'female':0, 'male':1} ).astype(int)
td['Gender']  = td['Sex'].map( {'female':0, 'male':1} ).astype(int)

# Estimate ages for passengers with null values
df['AgeFill'] = df['Age']
td['AgeFill'] = td['Age']

median_age = np.zeros((2, 3))

# Get median ages based on gender and passenger class
for i in range(0, 2):
    for j in range(0, 3):
        median_age[i, j] = df[ (df['Gender'] == i) & (df['Pclass'] == j+1) ]\
                           ['Age'].dropna().median()

# Fill in median ages where 'Age' value is null
for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df['Gender']==i) & (df['Pclass']==j+1) & (df['Age'].isnull())\
               , 'AgeFill' ] = median_age[i, j]
        td.loc[ (td['Gender']==i) & (td['Pclass']==j+1) & (td['Age'].isnull())\
               , 'AgeFill' ] = median_age[i, j]

# Add column to indicate whether 'Age' is null
df['AgeIsNull'] = pd.isnull(df['Age']).astype(int)
td['AgeIsNull'] = pd.isnull(td['Age']).astype(int)

# Calculate survival based on one assumptions:
# Passengers adhered to "Women and children first" principle
survival = np.zeros(len(td['PassengerId'])).astype(int)

for i in range(len(td['PassengerId'])):
    if td['Gender'][i] == 0:
        survival[i] = 1
    elif td['Age'][i] <= 18:
        survival[i] = 1
    else:
        survival[i] = 0

# Dict to be saved as pandas.DataFrame
outdict = {'PassengerId': td['PassengerId'], 'Survived': survival}

output = pd.DataFrame(data=outdict)

output.to_csv('titanic_survival_simple.csv', index=False)