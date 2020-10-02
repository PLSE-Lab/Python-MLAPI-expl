# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns# data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics as mt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



data=pd.read_csv('../input/train.csv')

data['Embarked']=data['Embarked'].fillna(value='s')
data=data.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)


lb3=LabelEncoder()
data['Embarked']=lb3.fit_transform(data['Embarked'])

lb4=LabelEncoder()
data['Sex']=lb4.fit_transform(data['Sex'])


data.isnull().any()
non_null_data=data[data.Age.notnull()]
dat=non_null_data
dat.info()
x1=dat.drop(['Survived'],axis =1)
y1=dat['Survived']



X_train1, X_test1, y_train1, y_test1 = train_test_split( x1, y1, test_size=0.30, random_state=0)




svr=SVC()
para = [
  {'C': [.001, 0.01,0.1,1,10], 'kernel': ['linear']},
  {'C': [1, 10,50, 100, 1000], 'gamma': [0.1, 10,50,1], 'kernel': ['rbf']},
 ]


gs=GridSearchCV(estimator=svr,param_grid=para,n_jobs=-1)
gs.fit(X_train1,y_train1)
print(gs.best_params_)


svc1=SVC(kernel='linear',C=.1)
svc1.fit(X_train1,y_train1)
y_pred=svc1.predict(X_test1)



print('confusion matrix for linear svc',mt.confusion_matrix(y_test1, y_pred))

print('precision score for linear svm',mt.precision_score(y_test1, y_pred))
print('recall score for linear svm',mt.recall_score(y_test1, y_pred))
print('f1 score for linear svm',mt.f1_score(y_test1, y_pred))
print('roc_auc_ score for linear svm',mt.roc_auc_score(y_test1, y_pred))
print('accuracy score for linear svm',mt.accuracy_score(y_test1, y_pred))


model = XGBClassifier()
model.fit(X_train1, y_train1)

y_pred=model.predict(X_test1)

from sklearn import metrics as mt

print('confusion matrix for xgboost',mt.confusion_matrix(y_test1, y_pred))

print('precision score for xgboost',mt.precision_score(y_test1, y_pred))
print('recall score for xgboost',mt.recall_score(y_test1, y_pred))
print('f1 score for xgboost',mt.f1_score(y_test1, y_pred))
print('roc_auc_ score for xgboost',mt.roc_auc_score(y_test1, y_pred))
print('accuracy score for xgboost',mt.accuracy_score(y_test1, y_pred))






# Any results you write to the current directory are saved as output.