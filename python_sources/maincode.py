import pandas as pd
import numpy as np
import pylab as P
import csv

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('../input/train.csv', header=0)

###print(df.head(3))
###print(df.tail(3))
###print(type(df))
###print(df.dtypes)
###print(df.info())
###print(df.describe())

###print(df['Age'][0:10])
###print(df.Age[0:10])
###print(type(df['Age']))

###print(df['Age'].mean())

###print(df[ ['Sex', 'Pclass', 'Age'] ])

###print(df[df['Age'] > 60])

###print(df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']])

###print(df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']])

for i in range(1,4):
    print(i, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) ]))

#df['Age'].hist()
#P.show()

#df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
#P.show()

df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

###print(df.head(3))

median_ages = np.zeros((2,3))

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()

print(median_ages)

df['AgeFill'] = df['Age']
###print(df.head())
###print(df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10))

for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),'AgeFill'] = median_ages[i,j]

###print(df[['Gender','Pclass','Age','AgeFill']].head(10))

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
###print(df[['Gender','Pclass','Age','AgeFill','AgeIsNull']].head(10))

df['FamilySize'] = df['SibSp'] + df['Parch']

df['Age*Class'] = df.AgeFill * df.Pclass

###df['Age*Class'].hist()
###P.show()

###print(df.dtypes)

###print(df.dtypes[df.dtypes.map(lambda x: x=='object')])

df = df.drop(['PassengerId', 'Age', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 
df = df.dropna()

print(df.dtypes)

train_data = df.values
###print(train_data)

# TEST DATA
# Load the test file into a dataframe
test_df = pd.read_csv('../input/test.csv', header=0)

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
###if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
###    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
###test_df.Embarked = test_df.Embarked.map(lambda x: Ports_dict[x]).astype(int)

# All the ages with no data -> make the median of all Ages
test_df['AgeFill'] = test_df['Age']
for i in range(0, 2):
    for j in range(0, 3):
        test_df.loc[ (test_df.Age.isnull()) & (test_df.Gender == i) & (test_df.Pclass == j+1),'AgeFill'] = median_ages[i,j]

# All the missing Fares -> assume median of their respective class
###if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
###    median_fare = np.zeros(3)
###    for f in range(0,3):                                              # loop 0 to 2
###        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
###    for f in range(0,3):                                              # loop 0 to 2
###        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

test_df['AgeIsNull'] = pd.isnull(test_df.Age).astype(int)
###print(df[['Gender','Pclass','Age','AgeFill','AgeIsNull']].head(10))

test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']

test_df['Age*Class'] = test_df.AgeFill * test_df.Pclass

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['PassengerId', 'Age', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
print(test_df[test_df.isnull().any(1)])

test_df.loc[ test_df.Fare.isnull(), 'Fare'] = 0
print(test_df[test_df.isnull().any(1)])

test_data = test_df.values

# ---prediction---
# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

# Create the random forest object which will include all the parameters for the fit
# n_estimators is The number of trees in the forest.
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,1::],train_data[0::,0])

output = forest.predict(test_data).astype(int)

predictions_file = open("myfirstforest.csv", "w", newline='')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()