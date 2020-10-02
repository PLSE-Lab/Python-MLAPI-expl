import csv
import pandas
import numpy
import scipy
pandas.options.mode.chained_assignment = None
from sklearn.ensemble import GradientBoostingClassifier

# initialize path to folder and training/testing sets
train = pandas.read_csv("../input/train.csv", dtype={"Age": numpy.float64})
test = pandas.read_csv("../input/train.csv", dtype={"Age": numpy.float64})

# output file
outFile = open("gbtPredictions.csv", "wb")

# TRAINING DATA
# clean the data for genders
train['Gender'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test['Gender'] = test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# clean the data for Embarkation port
if len(train.Embarked[train.Embarked.isnull()]) > 0:
	train.Embarked[ train.Embarked.isnull() ] = train.Embarked.dropna().mode().values
train['Embarked'] = train.Embarked.map( {'C': 0, 'Q': 1, 'S': 2}).astype(int)

# clean the data for missing Age values
median_ages = numpy.zeros((2,3))
for i in range(0,2):
	for j in range(0,3):
		median_ages[i,j] = train[(train['Gender'] == i) &
			(train['Pclass'] == j+1)]['Age'].dropna().median()

for i in range(0,2):
	for j in range(0,3):
		train.loc[ (train['Age'].isnull()) & (train['Gender'] == i) & (train['Pclass'] == j+1),
			'Age'] = median_ages[i,j]

train = train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)



# REPEAT FOR TEST DATA
if len(test.Embarked[test.Embarked.isnull()]) > 0:
	test.Embarked[ test.Embarked.isnull() ] = test.Embarked.dropna().mode().values
test['Embarked'] = test.Embarked.map( {'C': 0, 'Q': 1, 'S': 2}).astype(int)
median_ages = numpy.zeros((2,3))
for i in range(0,2):
	for j in range(0,3):
		median_ages[i,j] = test[(test['Gender'] == i) &
			(test['Pclass'] == j+1)]['Age'].dropna().median()
for i in range(0,2):
	for j in range(0,3):
		test.loc[ (test['Age'].isnull()) & (test['Gender'] == i) & (test['Pclass'] == j+1),
			'Age'] = median_ages[i,j]

# clean the fares values of the test set
if len(test.Fare[ test.Fare.isnull() ]) > 0:
    median_fares = numpy.zeros(3)
    for fare in range(0,3):                                              # loop 0 to 2
        median_fares[fare] = test[ test.Pclass == fare+1 ]['Fare'].dropna().median()
    for fare in range(0,3):                                              # loop 0 to 2
        test.loc[ (test.Fare.isnull()) & (test.Pclass == fare+1 ), 'Fare'] = median_fares[fare]

passengerIds = test.PassengerId.values
test = test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

# Begin setting up gradient boosted regression tree

trainValues = train.values
testValues = test.values

# fit the gradient boosting model
gbt = GradientBoostingClassifier(n_estimators = 200)
gbt = gbt.fit(trainValues[0::,1::],trainValues[0::,0])

# predict the test data
predictions = gbt.predict(testValues).astype(int)

# write predictions to a file
outFile_writer = csv.writer(outFile)
outFile_writer.writerow(["PassengerId", "Survived"])
outFile_writer.writerows(zip(passengerIds,predictions))
outFile.close()