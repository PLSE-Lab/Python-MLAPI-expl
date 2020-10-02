import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#Print you can execute arbitrary python code
#train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
#test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)

df = pd.read_csv('../input/train.csv', header=0)

# Clean 'Sex' column by adding a new column, 'Gender', and mapping female to 0 and male to 1

df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

# First replacing NA values with 'C' as an embark location as fare price seems to suggest this.
# Clean 'Embarked' column by adding a new column, 'EmbarkLoc', and mapping C to 0, Q to 1 and S to 2. 

df['Embarked'].fillna(value='C', inplace='True')

df['EmbarkLoc'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)

# Clean 'Age' column by first finding the median age for each gender within each passenger class. 
# Use these values to fill in the NA in the newly created 'AgeFill' column.
# Finally, create a new column 'AgeIsNull' that tracks whether or not we filled in this column with a 'guessed' value.

median_ages = np.zeros((2,3))

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & \
                              (df['Pclass'] == j+1)]['Age'].dropna().median()

df['AgeFill'] = df['Age']

for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]     

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

# Check if Fare for having a value in every entry, if not, recreate the above steps, but with Fare instead, but, not bothering
# with telling the algorithm if this value was guessed because this only happens once. 


if len(df.Fare[ df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]
        
# Create a new feature 'FamilySize' which adds the number of sibilings, parents, children and spouses together

df['FamilySize'] = df['SibSp'] + df['Parch']

# Artificially create new feature 'Age*Class', amplifying the effects of having a lower class or higher age

df['Age*Class'] = df.AgeFill * df.Pclass

# Drop all columns which are not numerical, as algorithm cannot deal with these

df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)

train_data = df.values


df = pd.read_csv('../input/test.csv', header=0)

# Clean 'Sex' column by adding a new column, 'Gender', and mapping female to 0 and male to 1

df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

# First replacing NA values with 'C' as an embark location as fare price seems to suggest this.
# Clean 'Embarked' column by adding a new column, 'EmbarkLoc', and mapping C to 0, Q to 1 and S to 2. 

df['Embarked'].fillna(value='C', inplace='True')

df['EmbarkLoc'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)

# Clean 'Age' column by first finding the median age for each gender within each passenger class. 
# Use these values to fill in the NA in the newly created 'AgeFill' column.
# Finally, create a new column 'AgeIsNull' that tracks whether or not we filled in this column with a 'guessed' value.

median_ages = np.zeros((2,3))

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & \
                              (df['Pclass'] == j+1)]['Age'].dropna().median()

df['AgeFill'] = df['Age']

for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]     

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

# Check if Fare for having a value in every entry, if not, recreate the above steps, but with Fare instead, but, not bothering
# with telling the algorithm if this value was guessed because this only happens once. 


if len(df.Fare[ df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]
        
# Create a new feature 'FamilySize' which adds the number of sibilings, parents, children and spouses together

df['FamilySize'] = df['SibSp'] + df['Parch']

# Artificially create new feature 'Age*Class', amplifying the effects of having a lower class or higher age

df['Age*Class'] = df.AgeFill * df.Pclass

# Drop all columns which are not numerical, as algorithm cannot deal with these

df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)

test_data = df.values


# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,1::],train_data[0::,0])

# Take the same decision trees and run it on the test data
output = forest.predict(test_data)