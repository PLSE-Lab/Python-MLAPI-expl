import numpy as np
import pandas as pd
import csv as csv
import pylab as P

# PREPARE TRAINING DATA

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('../input/train.csv', header=0)

df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

df['Embarked'].fillna('0', inplace=True)
df['Boarded'] = df['Embarked'].map( {'C': 1, 'Q': 2, 'S': 3, '0': 3 } ).astype(int)

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

df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass

df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Age','PassengerId'], axis=1) 

train_data = df.values



# PREPARE TESTING DATA

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('../input/test.csv', header=0)

df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

df['Embarked'].fillna('0', inplace=True)
df['Boarded'] = df['Embarked'].map( {'C': 1, 'Q': 2, 'S': 3, '0': 3 } ).astype(int)

df['AgeFill'] = df['Age']
for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass

passenger_id = df['PassengerId'].astype(int)
print(passenger_id)

df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Age','PassengerId'], axis=1) 

print(df[ df['Fare'].isnull() ][['Gender','Pclass','AgeFill','Fare']].head(5))

fare_median = df['Fare'].dropna().median()
print('fare median '+str(fare_median))
df.loc[ df.Fare.isnull(),'Fare' ] = fare_median

test_data = df.values

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100, oob_score = True)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,1::],train_data[0::,0])

#df_test = df.drop(['Survived'],axis=1)
#test_data = df_test.values

# Take the same decision trees and run it on the test data
output = forest.predict(test_data)

print(output)


prediction_file = open("randomforestmodel_py.csv", "w")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])
index = 0;
for row in output:       # For each row in test.csv
    prediction_file_object.writerow([passenger_id[index],row])  
    index = index+1
prediction_file.close()





