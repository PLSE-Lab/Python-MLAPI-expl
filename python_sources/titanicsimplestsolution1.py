import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv("../input/train.csv")#, dtype={"Age": np.float64},)
test = pd.read_csv("../input/train.csv")#, dtype={"Age": np.float64},)
train = train.drop(['Name','Ticket','Embarked', 'Cabin'], axis=1)
train['Sex'] = train['Sex'].map({'male':1,'female':0})
test['Sex'] = test['Sex'].map({'male':1,'female':0})
test    = test.drop(['Name','Ticket','Embarked','Cabin'], axis=1,)
train['Age'].fillna(train['Age'].median(),inplace = True)
test['Age'].fillna(train['Age'].median(),inplace = True)
test['Fare'].fillna(train['Fare'].mean(),inplace = True)
test.describe()
X_Train = train.drop("Survived", axis = 1)
Y_Train = train["Survived"]
X_Test = test.drop("PassengerId",axis = 1)
#Logistic Regression
LogReg = LogisticRegression()
LogReg.fit(X_Train,Y_Train)
print (LogReg.score(X_Train,Y_Train))
Y_Pred = LogReg.predict(X_Test)
#Random Forest
random_Forest = RandomForestClassifier(n_estimators = 100)
random_Forest.fit(X_Train,Y_Train)
print (random_Forest.score(X_Train,Y_Train))
Y_Predict = random_Forest.predict(X_Test)
#printing to csc file
Submission = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived":Y_Predict})
Submission.to_csv("Titanic_output.csv",index=False)