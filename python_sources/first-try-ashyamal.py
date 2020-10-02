import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
#print(train.describe())

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)

train['Gender'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test['Gender'] = test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


#print(train)
median_ages = np.zeros((2,3))
median_test_ages = np.zeros((2,3))

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = train[(train['Gender'] == i) & \
                              (train['Pclass'] == j+1)]['Age'].dropna().median()
                              
                              
for i in range(0, 2):
    for j in range(0, 3):
        median_test_ages[i,j] = test[(test['Gender'] == i) & \
                              (test['Pclass'] == j+1)]['Age'].dropna().median()                              
#print(median_ages)
train['AgeFill'] = train['Age']
test['AgeFill'] = test['Age']



for i in range(0, 2):
    for j in range(0, 3):
        train.loc[ (train.Age.isnull()) & (train.Gender == i) & (train.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]
                
                
for i in range(0, 2):
    for j in range(0, 3):
        test.loc[ (test.Age.isnull()) & (test.Gender == i) & (test.Pclass == j+1),\
                'AgeFill'] = median_test_ages[i,j]
                
#print(train['Age'],train['AgeFill'])

train['AgeIsNull'] = pd.isnull(train.Age).astype(int)
test['AgeIsNull'] = pd.isnull(test.Age).astype(int)

train['FamilySize'] = train['SibSp'] + train['Parch']
test['FamilySize'] = test['SibSp'] + test['Parch']

train['Age*Class'] = train.AgeFill * train.Pclass
test['Age*Class'] = test.AgeFill * test.Pclass

train = train.drop(['Name', 'Sex','Age', 'Ticket', 'Cabin', 'Embarked'], axis=1)

cleaned = train.dropna()
print(cleaned.dtypes)

train_data = cleaned.values   #as_matrix()
print(train_data[0::,1::])
print(train_data[0::,0])

forest = RandomForestClassifier(n_estimators = 100)
forest.fit(train_data[0::,1::],train_data[0::,0])

#classifierAlgo=LogisticRegression()
#classifierFit=classifierAlgo.fit(train_data[0::,1::],train_data[0::,0])


test_data=test.drop(['Name','Sex','Age','Ticket','Cabin','Embarked'],axis=1).dropna()

#print(test_data)


output = forest.predict(test_data).astype(int)
#output = classifierAlgo.predict(test_data)

#output=output.DataFrame()

#print(len(output))
#np.savetxt("output.csv", output, delimiter=",")

submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": output
    })
submission.to_csv('titanic.csv', index=False)
