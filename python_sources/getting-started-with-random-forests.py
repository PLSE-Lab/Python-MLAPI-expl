import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv as csv
from sklearn.ensemble import RandomForestClassifier

def prepare_data(df):
    # create Gender column with 0 and 1 values
    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # create Embarked column with 0, 1, 2 values
    df['Embarked'] = df['Embarked'].fillna('0')
    df['EmbarkedML'] = df['Embarked'].map( {'0': 0, 'S': 1, 'C': 2, 'Q': 3} ).astype(int)

    # there is NaN in test data
    df['Fare'] = df['Fare'].fillna(0.)
    
    # create AgeFill column with median values of age when passenger hasn't known age
    median_ages = np.zeros((2,3))

    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j + 1)]['Age'].dropna().median()

    df['AgeFill'] = df['Age']

    # create AgeIsNull column
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j + 1), 'AgeFill'] = median_ages[i,j]

    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

    # create FamilySize column
    df['FamilySize'] = df['SibSp'] + df['Parch']

    # create Age*Class column
    df['Age*Class'] = df.AgeFill * df.Pclass

    # drop unnecessary columns
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'PassengerId'], axis=1)

    print(df.head())
    print()
    
    return df.values


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train_df = pd.read_csv('../input/train.csv', header=0)
test_df = pd.read_csv('../input/test.csv', header=0)

# remember ids before drop
ids = test_df['PassengerId'].values

print("Info about datasets:")
print(train_df.info())
print()

print(test_df.info())
print()

# prepare training and test data
train_data = prepare_data(train_df)
test_data = prepare_data(test_df)

#np.set_printoptions(threshold=np.nan)

#print(train_data)
#print(test_data)


# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])

# Take the same decision trees and run it on the test data
output = forest.predict(test_data).astype(int)

print(output)

predictions_file = open("myfirstforest.csv", "wt")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()