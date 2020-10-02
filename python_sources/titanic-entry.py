# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import GradientBoostingClassifier #Use gradient boosting classifier from sklearn for model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

#Read training and testdata from csv files
trainDF = pd.read_csv('../input/train.csv')
testDF = pd.read_csv('../input/test.csv')

#print(len(trainDF.loc[(trainDF['Sex'] == 'female') & (trainDF['Survived'] == 0)]))
#trainDF.head()

#Passenger ID Nominal Attribute - All Passenger Labels Reassigned No Diff in Outcome
#Survived Target Attribute

#Adding new attribute 
trainDF['FamilySize'] = trainDF['SibSp'] + trainDF['Parch'] + 1
testDF['FamilySize'] = testDF['SibSp'] + testDF['Parch'] + 1
#Family Size Aggregates SibSp and Parch attributes
#Use aggregated Var to create binary isAlone attribute
trainDF['isAlone'] = 0
trainDF.loc[trainDF['FamilySize'] == 1, 'isAlone'] = 1
testDF['isAlone'] = 0
testDF.loc[testDF['FamilySize'] == 1, 'isAlone'] = 1

#Fill in Missing Values in Embarked Attribute with Most Frequent Occuring Value
trainDF['Embarked'] = trainDF['Embarked'].fillna(trainDF['Embarked'].mode()[0])
testDF['Embarked'] = testDF['Embarked'].fillna(testDF['Embarked'].mode()[0])

#Fill in Missing Values in Fare Attribute with Median Value
trainDF['Fare'] = trainDF['Fare'].fillna(trainDF['Fare'].median())
testDF['Fare'] = testDF['Fare'].fillna(testDF['Fare'].median())

#Generate random age values for missing values in Age
ageAvgTrain = trainDF['Age'].mean()
ageStdTrain = trainDF['Age'].std()
ageNullCountTrain = trainDF['Age'].isnull().sum()

ageNullRandListTrain = np.random.randint(ageAvgTrain - ageAvgTrain, ageAvgTrain + ageStdTrain, size = ageNullCountTrain)
trainDF['Age'][np.isnan(trainDF['Age'])] = ageNullRandListTrain
trainDF['Age'] = trainDF['Age'].astype(int)

ageAvgTest = testDF['Age'].mean()
ageStdTest = testDF['Age'].std()
ageNullCountTest = testDF['Age'].isnull().sum()

ageNullRandListTest = np.random.randint(ageAvgTest - ageAvgTest, ageAvgTest + ageStdTest, size = ageNullCountTest)
testDF['Age'][np.isnan(testDF['Age'])] = ageNullRandListTest
testDF['Age'] = testDF['Age'].astype(int)

titlesTrain = pd.DataFrame(trainDF.apply(lambda x: x.Name.split(",")[1].split(".")[0], axis=1), columns=['Title'])
#print(pd.Categorical(titlesTrain.Title))
trainDF = trainDF.join(titlesTrain)

titlesTest = pd.DataFrame(testDF.apply(lambda x: x.Name.split(",")[1].split(".")[0], axis=1), columns=['Title'])
#print(pd.Categorical(titlesTest.Title))
testDF = testDF.join(titlesTest)
#print(trainDF.columns)
trainDF['Title'] = trainDF['Title'].str.strip()
#print(trainDF)
trainDF['Title'] = trainDF['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major',\
                                    'Rev','Sir','Jonkheer','Dona'],'Rare')
trainDF['Title'] = trainDF['Title'].replace('Mlle', 'Miss')
trainDF['Title'] = trainDF['Title'].replace('Ms', 'Miss')
trainDF['Title'] = trainDF['Title'].replace('Mme', 'Mrs')

testDF['Title'] = testDF['Title'].str.strip()
testDF['Title'] = testDF['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major',\
                                   'Rev','Sir','Jonkheer','Dona'],'Rare')
testDF['Title'] = testDF['Title'].replace('Mlle', 'Miss')
testDF['Title'] = testDF['Title'].replace('Ms', 'Miss')
testDF['Title'] = testDF['Title'].replace('Mme', 'Mrs')

trainDF.loc[trainDF['Sex'] == "male",'Sex'] = 0
trainDF.loc[trainDF['Sex'] == "female",'Sex'] = 1
testDF.loc[testDF['Sex'] == "male",'Sex'] = 0
testDF.loc[testDF['Sex'] == "female",'Sex'] = 1
trainDF['Sex'] = trainDF['Sex'].astype(int)
testDF['Sex'] = testDF['Sex'].astype(int)

trainDF['Embarked'] = trainDF['Embarked'].map({'S': 0, 'C' : 1, 'Q' : 2}).astype(int)
testDF['Embarked'] = testDF['Embarked'].map({'S': 0, 'C' : 1, 'Q' : 2}).astype(int)



titleMapping = {"Mr" : 1, "Miss" : 2, "Mrs" : 3, "Master" : 4, "Rare" : 5}
#print
trainDF['Title'] = trainDF['Title'].map(titleMapping)
trainDF['Title'] = trainDF['Title'].fillna(0)
testDF['Title'] = testDF['Title'].map(titleMapping)
testDF['Title'] = testDF['Title'].fillna(0)

trainDF.loc[trainDF['Fare'] <= 7.91, 'Fare'] = 0
trainDF.loc[(trainDF['Fare'] > 7.91) & (trainDF['Fare'] <= 14.454), 'Fare'] = 1
trainDF.loc[(trainDF['Fare'] > 14.454) & (trainDF['Fare'] <= 31), 'Fare'] = 2
trainDF.loc[(trainDF['Fare'] > 31), 'Fare'] = 3
trainDF['Fare'] = trainDF['Fare'].astype(float)

testDF.loc[testDF['Fare'] <= 7.91, 'Fare'] = 0
testDF.loc[(testDF['Fare'] > 7.91) & (testDF['Fare'] <= 14.454), 'Fare'] = 1
testDF.loc[(testDF['Fare'] > 14.454) & (testDF['Fare'] <= 31), 'Fare'] = 2
testDF.loc[(trainDF['Fare'] > 31), 'Fare'] = 3
testDF['Fare'] = testDF['Fare'].astype(float)

trainDF.loc[trainDF['Age'] <= 16, 'Age'] = 0
trainDF.loc[(trainDF['Age'] > 16) & (trainDF['Age'] <= 32), 'Age'] = 1
trainDF.loc[(trainDF['Age'] > 32) & (trainDF['Fare'] <= 48), 'Age'] = 2
trainDF.loc[(trainDF['Age'] > 48) & (trainDF['Fare'] <= 64), 'Age'] = 3
trainDF.loc[(trainDF['Age'] > 64), 'Age'] = 4
trainDF['Age'] = trainDF['Age'].astype(float)

testDF.loc[testDF['Age'] <= 16, 'Age'] = 0
testDF.loc[(testDF['Age'] > 16) & (testDF['Age'] <= 32), 'Age'] = 1
testDF.loc[(testDF['Age'] > 32) & (testDF['Fare'] <= 48), 'Age'] = 2
testDF.loc[(trainDF['Age'] > 48) & (testDF['Fare'] <= 64), 'Age'] = 3
testDF.loc[(trainDF['Age'] > 64), 'Age'] = 4
testDF['Age'] = testDF['Age'].astype(float)
#print(testDF['PassengerId'])
dropElements = {'PassengerId','Name','Ticket','Cabin','SibSp','Parch','FamilySize'}
trainDF = trainDF.drop(dropElements, axis = 1)
finalDataPassID = testDF['PassengerId']
testDF = testDF.drop(dropElements, axis = 1)
#print(trainDF.columns)
#print(testDF.columns)
# Any results you write to the current directory are saved as output.

y = trainDF['Survived'].values
X = (trainDF.drop('Survived', axis = 1)).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

gb = GradientBoostingClassifier(learning_rate = 0.1, n_estimators = 30)
gb.fit(X_train, y_train)
predictions = gb.predict(X_test)

print("Accuracy is ",accuracy_score(y_test, predictions))

finalDataToPredict = testDF.values
results = gb.predict(finalDataToPredict)



results = pd.DataFrame(results)
results.columns = ['Survived']
#print(results)
gender_submission = pd.concat([finalDataPassID,results], axis = 1)
#print(gender_submission.columns)
gender_submission.to_csv('gender_submission.csv', index = False)

#results = pd.DataFrame([finalDataPassID, results], columns = ['PassengerId','Survived'])

#print(finalDataPassID)


