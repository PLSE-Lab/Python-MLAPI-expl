import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
from xgboost import XGBClassifier
import xgboost as xgb
test = pd.read_csv("../input/criminal_test.csv")
train= pd.read_csv("../input/criminal_train.csv")


#Import libraries:
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

target = train["Criminal"]

train= train[train["IIHHSIZ2"]>=0.75]
train= train[train["PRXRETRY"]>=1.8]
train= train[train["PRXRETRY"]>=0] 
train= train[train["HLNVREF"]>=6.5]
train = train[train["CELLNOTCL"]<=98]

list1 = ['PERID']
x = train.drop(list1,axis = 1)

features = x.columns[x.columns!="Criminal"]
y_test = test.drop(list1,axis =1)
X_train = x[features]
y_train = x['Criminal']
y_wala = test.drop(list1,axis=1)

xgb4 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=20000,
 max_depth=3,
 min_child_weight=4,
 gamma=0.4,
 subsample=0.65,
 colsample_bytree=0.65,
 reg_alpha=1,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
 
xgb4.fit(X_train,y_train)
predicted = xgb4.predict(y_wala)
sub2 = pd.DataFrame({'PERID':test.PERID, 'Criminal':predicted})
sub2 = sub2[['PERID', 'Criminal']]
sub2.to_csv('xgb_updatedji.csv', index=False)

X_train.describe()