# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import PolynomialFeatures
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from sklearn.ensemble import RandomForestClassifier
# Any results you write to the current directory are saved as output.

data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#data.columns
#data.describe()

columns= ['PassengerId','Pclass','Sex','Age', 'SibSp','Parch', 'Fare','Survived','Embarked']
features= ['Pclass','Sex','Age', 'SibSp','Parch', 'Fare','Embarked']
#data[features].groupby('Embarked')['Embarked'].count()
#features= ['Sex','Embarked']
#from sklearn.preprocessing import OneHotEncoder
#enc = OneHotEncoder(handle_unknown='ignore')
#df.fillna({'a':0, 'b':0}, inplace=True)
#df=data[columns].fillna({'Age':data['Age'].mean(),'Fare':data['Fare'].mean(),'Embarked':'S'})
#df=df.fillna(0)
#X=df[features]
df=data[columns].fillna({'Embarked':'S'})
X=pd.get_dummies(df[features],columns=['Sex','Embarked','Pclass'], prefix='col')
#X=pd.get_dummies(X,columns=['Parch'], prefix='Parch')
#new_data = X.copy()

# make new columns indicating what will be imputed
#cols_with_missing = (col for col in new_data.columns 
 #                                if new_data[col].isnull().any())
#for col in cols_with_missing:
#    new_data[col + '_was_missing'] = new_data[col].isnull()
#new_data.columns
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
X = my_imputer.fit_transform(X)
poly = PolynomialFeatures(3)
X=poly.fit_transform(X)
#print(poly.get_feature_names())
#poly
#X=X[features]
X=pd.DataFrame(X)
X.columns=poly.get_feature_names()
#['Age', 'SibSp', 'Parch','Fare', 'col_female', 'col_male', 'col_C', 'col_Q','col_S', 'col_1', 'col_2', 'col_3']
#X.head()       
#X1=pd.get_dummies(X)    
#enc.fit(X)
#X=enc.transform(X).toarray()
#X_new = pd.concat([X,pd.get_dummies(X,columns=['Sex','Embarked'], prefix='col')],axis=1).drop(['Sex','Embarked','col_0'],axis=1)
#X = pd.get_dummies(X,columns=['Sex','Embarked'], prefix='col')
#X_new = pd.get_dummies(X,columns=['Sex','Embarked'], prefix='col')
#print(df)
from sklearn.svm import SVC, LinearSVC
svc = LinearSVC()
y=df.Survived
#print(X.columns)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
#n=[10,20,30,40,50,60,70,80,90,100,200]
n=[50]
for n_estim in n:
    #model=RandomForestClassifier(n_estimators=n_estim)
    model=LogisticRegression()
    #model=xgb.XGBClassifier(learning_rate=0.05)
    #model=LogisticRegression()
    #print(X.columns)
    model.fit(X,y)
    #scores = cross_val_score(model,X,y,cv=5)
    #print("estim:%f Accuracy: %0.2f (+/- %0.2f)" % (n_estim,scores.mean(), scores.std() * 2))
    #print(scores)
    
columns_test= ['PassengerId','Pclass','Sex','Age', 'SibSp','Parch', 'Fare','Embarked']
df=test[columns_test]
test_X=test[features]
test_X = pd.get_dummies(test_X,columns=['Sex','Embarked','Pclass'], prefix='col')
#df=test[columns_test].fillna({'Age':data['Age'].mean(),'Fare':data['Fare'].mean()})
#df=df.fillna(0)
#test_X.columns
test_X = my_imputer.fit_transform(test_X)
test_X=poly.fit_transform(test_X)
#X=X[features]
test_X=pd.DataFrame(test_X)
test_X.columns=poly.get_feature_names()
#['Age', 'SibSp', 'Parch', 'Fare', 'col_female','col_male', 'col_C', 'col_Q', 'col_S', 'col_1', 'col_2', 'col_3']
       

#test_y=test.survived
#test_X = pd.concat([test_X,pd.get_dummies(test_X,columns=['Sex','Embarked'], prefix='col')],axis=1).drop(['Sex','Embarked'],axis=1)

#print(test_X.columns)
model.predict(test_X)

my_submission = pd.DataFrame({'PassengerId': df.PassengerId, 'Survived': model.predict(test_X)})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)