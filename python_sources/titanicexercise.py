import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv" )
test = pd.read_csv("../input/test.csv" )

#Data cleaning

#age
train['Gender'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = train[(train['Gender'] == i) & \
                              (train['Pclass'] == j+1)]['Age'].dropna().median()
                              
for i in range(0, 2):
    for j in range(0, 3):
        train.loc[ (train.Age.isnull()) & (train.Gender == i) & (train.Pclass == j+1),\
                'Age'] = median_ages[i,j]
train['Age']=train['Age'].astype(int)

test['Gender'] = test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = test[(test['Gender'] == i) & \
                              (test['Pclass'] == j+1)]['Age'].dropna().median()
                              
for i in range(0, 2):
    for j in range(0, 3):
        test.loc[ (test.Age.isnull()) & (test.Gender == i) & (test.Pclass == j+1),\
                'Age'] = median_ages[i,j]
test['Age']=test['Age'].astype(int)

# feature engineering
train['FamilySize'] = train['SibSp'] + train['Parch']
test['FamilySize'] = test['SibSp'] + test['Parch']

# drop redundent features
train = train.drop(['PassengerId','Name','Sex','SibSp','Parch','Cabin','Embarked','Ticket','Fare'],axis=1)
test = test.drop(['Name','Sex','SibSp','Parch','Cabin','Embarked','Ticket','Fare'],axis=1)

train_data = train.values
test_data = test.values
print(test.info())
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0::,1::],train_data[0::,0])
output = forest.predict(test_data[0::,1::])

# submit the output
submission=pd.DataFrame()
submission["PassengerId"] = test["PassengerId"]
submission["Survived"] = output
submission.to_csv("Submission.csv", index = False)
