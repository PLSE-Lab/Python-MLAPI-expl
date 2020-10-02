import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
from xgboost import XGBClassifier
import xgboost as xgb
test = pd.read_csv("../input/criminal_test.csv")
train= pd.read_csv("../input/criminal_train.csv")
featuers = train.columns[train.columns != "Criminal"]
X_train = train[featuers]
y_train = train['Criminal']
list = ['PERID']
x = train.drop(list,axis=1)
y = test.drop(list,axis=1)
list1 = ['TROUBUND']
x = x.drop(list1,axis=1)
y = y.drop(list1,axis=1)
x = x.drop('Criminal',axis=1)
model = XGBClassifier()
model.fit(x,y_train)
predicted = model.predict(y)
sub2 = pd.DataFrame({'PERID':test.PERID, 'Criminal':predicted})
sub2 = sub2[['PERID', 'Criminal']]
sub2.to_csv('xgboost_edit.csv', index=False)