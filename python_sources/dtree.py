import numpy as np
import pandas as pd
# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
all_data = train.append(test)

# (2) Create the submission file with passengerIDs from the test file
submission = pd.DataFrame({"PassengerId": test['PassengerId'], "Survived": pd.Series(dtype='int32')})

# Prepare the data
# set up gender
train['Gender'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test['Gender'] = test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
all_data['Gender'] = all_data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# fix missing ages
# find median ages
median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = all_data[(all_data['Gender'] == i) & (all_data['Pclass'] == j+1)]['Age'].dropna().median()
print(median_ages)
# create ageFill column
train['AgeFill'] = train['Age']
test['AgeFill'] = test['Age']
# Fill in missing ages
for i in range(0, 2):
    for j in range(0, 3):
        test.loc[ (test.Age.isnull()) & (test.Gender == i) & (test.Pclass == j+1), 'AgeFill'] = median_ages[i,j]
        train.loc[ (train.Age.isnull()) & (train.Gender == i) & (train.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

# Fill in missing fare
median_fare = all_data['Fare'].dropna().median()
print("Median fare" + str(median_fare))
test.loc[test.Fare.isnull(), 'Fare'] = median_fare

#print(train[train['AgeFill'].isnull()])
#print(test[test['AgeFill'].isnull()])

# Create FamilySize
train['FamilySize'] = train['SibSp'] + train['Parch']
test['FamilySize'] = test['SibSp'] + train['Parch']

#drop unused colums
train = train.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)
test = test.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1) 

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())
#print(test.describe())
#print(all_data.describe())

#print(test.describe())
#print(test[test['Fare'].isnull()])
#test = test.dropna()
#print(test.describe())

train_data = train.values
test_data = test.values

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,1::],train_data[0::,0])

# Take the same decision trees and run it on the test data
output = forest.predict(test_data)

#print(output)
submission.Survived = output

print(submission)

# (4) Create final submission file
submission.to_csv("submission.csv", index=False)