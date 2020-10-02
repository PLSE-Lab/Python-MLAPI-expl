import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

#get some overview of the data
titanic=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')

#fill in meidan of Age to the absent data of Age
#Change 'male' to 0, and 'female' to 1
print(titanic.Sex.unique())
titanic.Age=titanic.Age.fillna(titanic.Age.median())
titanic.loc[titanic.Sex=='male','Sex']=0
titanic.loc[titanic.Sex=='female','Sex']=1

#change Embarked entries to numbers
print(titanic.Embarked.unique())
titanic.Embarked=titanic.Embarked.fillna('S')
Embarked_dummies=pd.get_dummies(titanic.Embarked,prefix='Embarked').iloc[:,1:]
titanic=pd.concat([titanic,Embarked_dummies],axis=1)

predictors=['Sex','Age','Pclass','Fare','Embarked_Q','Embarked_S','Parch','SibSp']

#RandomForest training
model=RandomForestRegressor(n_estimators=1000,oob_score=True,random_state=42)
train_predictors=titanic[predictors]
train_target=titanic['Survived']
model.fit(train_predictors,train_target)
predicted=model.predict(train_predictors)
predicted[predicted>0.5]=1
predicted[predicted<=0.5]=0
count=0
for i in range(len(predicted)):
    if predicted[i]==titanic['Survived'][i]:
        count +=1
accuracy=float(count)/len(predicted)

test.Age=test.Age.fillna(titanic.Age.median())
test.Fare=test.Fare.fillna(titanic.Fare.median())
test.loc[test.Sex=='male','Sex']=0
test.loc[test.Sex=='female','Sex']=1
test.Embarked=test.Embarked.fillna('S')
Embarked_dummies=pd.get_dummies(test.Embarked,prefix='Embarked').iloc[:,1:]
test=pd.concat([test,Embarked_dummies],axis=1)
model=RandomForestRegressor(random_state=1)

model=RandomForestRegressor(n_estimators=1000,oob_score=True,random_state=42)
model.fit(titanic[predictors],titanic['Survived'])
predictions=model.predict(test[predictors])
submission=pd.DataFrame({'PassengerID':test['PassengerId'],'Survived':predictions})



