import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn import svm, neighbors
from sklearn.preprocessing import Imputer
from patsy import dmatrix, dmatrices
import sklearn.ensemble as ske

clf = svm.SVC()


#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
train.drop(['Name', 'Ticket','Cabin'], 1, inplace=True)
test.drop(['Name','Ticket','Cabin'], 1, inplace=True)


train['Sex'] = train['Sex'].map({'male':0, 'female':1}).astype(int)
test['Sex'] = test['Sex'].map({'male':0, 'female':1}).astype(int) 

train['Child'] = train['Age'].apply(lambda x: 1 if x < 18 else 0)
test['Child'] = test['Age'].apply(lambda x: 1 if x < 18 else 0) 

train['Embarked'] = [0 if embarked == 'C' else 1 if embarked == 'Q' 
                else 2 if embarked == 'S' else np.NaN for embarked in train['Embarked']]
test['Embarked'] = [0 if embarked == 'C' else 1 if embarked == 'Q' 
                else 2 if embarked == 'S' else np.NaN for embarked in test['Embarked']]

#See the proportions of ppl survived by categocigal variables
train['Survived'].groupby(train['Sex']).mean()      #o
train['Survived'].groupby(train['Pclass']).mean()   #o
train['Survived'].groupby(train['SibSp']).mean()    #maybe?
train['Survived'].groupby(train['Parch']).mean()    #maybe?
train['Survived'].groupby(train['Embarked']).mean() #o
train['Survived'].groupby(train['Child']).mean() #o




imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
y = train['Survived']
#X = train[['Sex', 'Pclass','SibSp','Parch','Embarked','Child']]
X = train[['Sex', 'Pclass','Parch','Embarked','Child']]
imp.fit(X)
X = imp.transform(X)
y = train['Survived']

test_pred = test[['Sex', 'Pclass','Parch','Embarked','Child']]
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(test_pred)
test_pred = imp.transform(test_pred)

clf.fit(X, y)
y_pred = clf.predict(test_pred)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_pred})
submission.to_csv('titanic.csv', index=False)

