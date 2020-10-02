# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 08:14:54 2018
Titanic project
@author: zg
"""
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score,accuracy_score

'''
1.import data
'''
trainData=pd.read_csv('../input/train.csv')
testData=pd.read_csv('../input/test.csv')
gender_submission=pd.read_csv('../input/gender_submission.csv')
'''
2.Missing data analysis :Age Cabin Embarked
'''
# 1) first I using zero to replace the age nan,
train=trainData.copy()

test=testData.copy()
train['Age']=train['Age'].fillna(train["Age"].median())
test['Age']=test['Age'].fillna(test["Age"].median())
# 2) exclusion Cabin
# 3) Embarked just only one missing data ,and according to the Cabin,I think  
#    PassengerId=62 Embarked=C,
train['Embarked']=train['Embarked'].fillna('C')
# 4) test Fare use mean to replace
test['Fare']=test['Fare'].fillna(test["Fare"].median())
#temp=list(train.loc[:,'Embarked'])
#print(np.unique(temp))
#Embarked= C Q S 

'''
3.data precessing
'''
#female=0 male=1
train.loc[train['Sex']=='female','Sex']=0
train.loc[train['Sex']=='male','Sex']=1
test.loc[test['Sex']=='female','Sex']=0
test.loc[test['Sex']=='male','Sex']=1
#Embarked C=1 Q=2 S=3
train.loc[train['Embarked']=='C','Embarked']=1
train.loc[train['Embarked']=='Q','Embarked']=2
train.loc[train['Embarked']=='S','Embarked']=3
test.loc[test['Embarked']=='C','Embarked']=1
test.loc[test['Embarked']=='Q','Embarked']=2
test.loc[test['Embarked']=='S','Embarked']=3
#select features and Normalization
train_x=train.loc[:,['Pclass','Age','Sex','SibSp','Parch','Fare','Embarked']]
train_y=train['Survived']

test_x=test.loc[:,['Pclass','Age','Sex','SibSp','Parch','Fare','Embarked']]
test_y=gender_submission['Survived']

min_max_scaler = preprocessing.MaxAbsScaler()
train_x_minmax = min_max_scaler.fit_transform(train_x)
test_x_minmax = min_max_scaler.fit_transform(test_x)
'''
4.train model
'''
# using Logistic regression
#clf=LogisticRegression(random_state=1,solver='liblinear')
clf=AdaBoostClassifier()
clf.fit(train_x_minmax,train_y)
test_predict=clf.predict(test_x_minmax)
#result
print("Accuracy=",accuracy_score(test_y,test_predict))
print("AUC=",roc_auc_score(test_y, test_predict))
#submission
submission = pd.DataFrame({
        "PassengerId": testData["PassengerId"],
        "Survived": test_predict
    })
submission.to_csv('titanic.csv', index=False)