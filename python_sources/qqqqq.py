import numpy as np
import pandas as pd
import pylab as P

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head(2))

#print("\n\nSummary statistics of training data")
#print(train.describe())

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)

#print(type(train))
#print(train.dtypes)
#train.info()
#print(train.describe())

#print(train['Age'][0:10])
#print(train.Age.median())
#print(train[['Sex', 'Pclass', 'Age']])
#print(train[train.Age.isnull()][['Sex', 'Pclass', 'Age', 'Survived']])

#for i in range(1,4):
#    print(i, len(train[(train.Sex=="male") & (train.Pclass==i)]))

#print(train[0:5][:])

#train['Gender']=4
#train['Gender'] = train['Sex'].map(lambda x: x[0].upper())
train['Gender'] = train['Sex'].map({'female': 0, 'male': 1}).astype(int)
#print(train.head())

medianAges = np.zeros((2,3))

for i in range(0, 2):
    for j in range(0, 3):
        medianAges[i, j] = train[(train['Gender']==i) & (train['Pclass']==j+1)]['Age'].dropna().median()
        
#print(medianAges)

train['AgeFill'] = train['Age']

#print(train[train['Age'].isnull()][['Gender', 'Pclass', 'Age', 'AgeFill']].head(10))

for i in range(2):
    for j in range(3):
        train.loc[(train['Age'].isnull()) & (train['Gender']==i) & (train['Pclass']==j+1), 'AgeFill']=medianAges[i][j]

train['AgeIsNull'] = pd.isnull(train['Age']).astype(int)

train['FamilySize'] = train['SibSp'] + train['Parch']

train['Age*Class'] = train['AgeFill']*train['Pclass']

#print(train[train['Age'].isnull()][['Gender', 'Pclass', 'Age', 'AgeFill', 'AgeIsNull', 'SibSp', 'Parch', 'FamilySize', 'Age*Class']].head(10))

#print(train.dtypes[train.dtypes.map(lambda x: x=="object")])

train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'PassengerId'], axis=1)

train_array = train.values
print(train_array)

