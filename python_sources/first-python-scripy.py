import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)

# convert all strings to integer classifiers
# fill the massing values of the data and make it complete

# female = 0, male = 1
train['Gender'] = train['Sex'].map({'female':0, 'male':1}).astype(int)

if len(train.Embarked[train.Embarked.isnull()]) > 0:
    train.Embarked[ train.Embarked.isnull() ] = train.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train.Embarked = train.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

median_age = train['Age'].dropna().median()
if len(train.Age[ train.Age.isnull() ]) > 0:
    train.loc[ (train.Age.isnull()), 'Age'] = median_age

# remove the name column, cabin, ticket, ana sex
train = train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

# deal with the test data the same as the train data
test['Gender'] = test['Sex'].map({'female':0, 'male':1}).astype(int)
if len(test.Embarked[ test.Embarked.isnull() ]) > 0:
    test.Embarked[ test.Embarked.isnull() ] = test.Embarked.dropna().mode().values
test.Embarked = test.Embarked.map( lambda x: Ports_dict[x]).astype(int)

median_age = test['Age'].dropna().median()
if len(test.Age[ test.Age.isnull()] ) > 0:
    test.loc[ (test.Age.isnull()), 'Age'] = median_age

# fill in the fare data
if len(test.Fare[ test.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):
        median_fare[f] = test[ test.Pclass == f+1]['Fare'].dropna().median()
    for f in range(0,3):
        test.loc[ (test.Fare.isnull()) & (test.Pclass == f+1), 'Fare'] = median_fare[f]

# save the ids of test data
ids = test['PassengerId'].values
# remove useless columns
test = test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

train = train.drop('Fare', axis=1)
test = test.drop('Fare', axis=1)


# start training model
train_data = train.values
test_data = test.values

print ('Training...')
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])

print ('Predcting...')
output = forest.predict(test_data).astype(int)

result = {'PassengerId':ids, 'Survived':output}
df = pd.DataFrame(result)
df.to_csv('result.csv', index=False)

print ('Done.')






