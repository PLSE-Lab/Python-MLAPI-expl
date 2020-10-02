
"""
Created on Fri Mar 17 21:26:23 2017

@author: OLAGUNJU
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



#cleaning the train data
train.Embarked = train['Embarked'].fillna('S')
train.Age = train['Age'].fillna(np.mean(train.Age))
train.Age[train.Age <= 18 ] = 1
train.Age[train.Age > 18 ] = 0           
train['Sex'][train['Sex'] == 'male'] = 0  
train['Sex'][train['Sex'] == 'female'] = 1 
train.Embarked[train.Embarked == 'S'] = 1 
train.Embarked[train.Embarked == 'C'] = 2 
train.Embarked[train.Embarked == 'Q'] = 3 
              
#cleaning the test data
test.Embarked = test['Embarked'].fillna('S')
test.Age = test['Age'].fillna(np.mean(test.Age))
test.Fare = test['Fare'].fillna(np.mean(test.Fare))
test.Age[test.Age <= 18 ] = 1
test.Age[test.Age > 18 ] = 0           
test['Sex'][test['Sex'] == 'male'] = 0  
test['Sex'][test['Sex'] == 'female'] = 1 
test.Embarked[test.Embarked == 'S'] = 1 
test.Embarked[test.Embarked == 'C'] = 2 
test.Embarked[test.Embarked == 'Q'] = 3 

#Spliting the data into features(X) and targets(y)             
y_train = train.Survived.values
X_train = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].values
               
#y_test = gender.Survived.values
X_test = test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].values               
         

#linear regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
ylog_pred = logreg.predict(X_test)

solution = pd.DataFrame({'Survived':np.array(ylog_pred)}, index=test.PassengerId)
solution.to_csv('copy_of_solution.csv')

#Print to standard output, and see the results in the "log" section below after running your script
