import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


trainData = pd.read_csv('../input/train.csv')
testData = pd.read_csv('../input/test.csv')

## START HERE
trainData['Gender'] = trainData['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
testData['Gender'] = testData['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


""" Column selection """
trainDataSet = pd.DataFrame()
trainDataSet =  trainData.loc[:,['Gender','Pclass','Age','Fare','SibSp','Parch','Survived']]
trainDataSet = trainDataSet.fillna(trainDataSet.mean())

testDataSet = pd.DataFrame()
testDataSet = testData.loc[:,['Gender','Pclass','Age','Fare','SibSp','Parch','Survived']]
testDataSet = testDataSet.fillna(testDataSet.mean())

xTrain = trainDataSet.loc[:,['Gender','Pclass','Age','Fare','SibSp','Parch']]
yTrain = trainDataSet['Survived']
xTest = testDataSet.loc[:,['Gender','Pclass','Age','Fare','SibSp','Parch']]


""" Create model """
model = RandomForestClassifier(n_estimators =100, min_samples_leaf= 15)
model = model.fit(xTrain.as_matrix(), np.ravel(yTrain))
testData['Survived'] = model.predict(xTest)

## FINISH HERE

submission = pd.DataFrame()
submission = testData.loc[:,['PassengerId','Survived']]



print(submission)

submission.to_csv('submission.csv',index=False)