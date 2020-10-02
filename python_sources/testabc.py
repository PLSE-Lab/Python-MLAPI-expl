import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

## search missing data
train.describe() ## some data are missing on Age column
test.describe()  ## Age and Fare columns have missing data

## insert insert value
train['Age'] = train['Age'].fillna( train['Age'].median() )
test['Age'] = test['Age'].fillna( test['Age'].median() )
test['Fare'] = test['Fare'].fillna( test['Fare'].median() )

## convert sex column into integer
train.loc[ train['Sex'] == 'male', "Sex" ] = 0
train.loc[ train['Sex'] == 'female', "Sex" ] = 1
test.loc[ test['Sex'] == 'male', "Sex" ] = 0
test.loc[ test['Sex'] == 'female', "Sex" ] = 1

## convert embarked into integer
train['Embarked'] = train['Embarked'].fillna( 0 )
train.loc[ train['Embarked'] == 'S', "Embarked" ] = 0
train.loc[ train['Embarked'] == 'C', "Embarked" ] = 1
train.loc[ train['Embarked'] == 'Q', "Embarked" ] = 2
test['Embarked'] = test['Embarked'].fillna( 0 )
test.loc[ test['Embarked'] == 'S', "Embarked" ] = 0
test.loc[ test['Embarked'] == 'C', "Embarked" ] = 1
test.loc[ test['Embarked'] == 'Q', "Embarked" ] = 2

## logistic regression model
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
logit = LogisticRegression(random_state=1)
logit.fit( train[predictors], train['Survived'] )
predictions = logit.predict(test[predictors])

submission = pd.DataFrame({
    "PassengerId": test['PassengerId'],
    "Survived": predictions
})
submission.to_csv("kaggle.csv")

 
 