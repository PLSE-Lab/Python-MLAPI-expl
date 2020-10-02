import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

feature=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

train.loc[train['Sex']=='male','Sex']=1
train.loc[train['Sex']=='female','Sex']=2
train.loc[train['Embarked']=='C','Embarked']=0
train.loc[train['Embarked']=='S','Embarked']=1
train.loc[train['Embarked']=='Q','Embarked']=2
train['Age']=train['Age'].fillna(train['Age'].mean())
train['Embarked']=train['Embarked'].fillna(1)

test.loc[test['Sex']=='male','Sex']=1
test.loc[test['Sex']=='female','Sex']=2
test.loc[test['Embarked']=='C','Embarked']=0
test.loc[test['Embarked']=='S','Embarked']=1
test.loc[test['Embarked']=='Q','Embarked']=2
test['Age']=test['Age'].fillna(test['Age'].mean())
test['Embarked']=test['Embarked'].fillna(1)
test['Fare']=test['Fare'].fillna(test['Fare'].mean())

model=LogisticRegression()
mytrainer=train[feature]
testfeature=test[feature]

mylabel=train['Survived']
model.fit(mytrainer,mylabel)
prediction=model.predict(testfeature)
final=pd.DataFrame(test['PassengerId'])
final['Survived']=prediction
#print(final.head(5))
#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())

#Any files you save will be available in the output tab below
final.to_csv('copy_of_the_training_data.csv', index=False)