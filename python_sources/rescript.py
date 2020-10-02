import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train_1.csv')
test = pd.read_csv('../input/test_1.csv')
t2=train
t2=t2.dropna()
t2=t2.reset_index()
rep={'Gender' : {'Male' : 1 , 'Female' : 0},'Married' : {'Yes' : 1 , 'No' : 0},'Education' : {'Graduate' : 1 , 'Not Graduate' : 0},'Self_Employed' : {'Yes' : 1 , 'No' : 0},'Dependents' : {'0' : 0 , '1' : 1 , '2' : 2 , '3+' : 3},'Property_Area' : {'Rural' : 0 , 'Semiurban' : 1 , 'Urban' : 2}, 'Loan_Status' : {'Y' : 1 , 'N' : 0}}
t2=t2.replace(rep)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
tr_x=t2[[ 'Dependents', 'Education', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']]
tr_y=t2['Loan_Status']
import statistics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
model = RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=4)
#model = GaussianNB()
acc1=[]
# X is the feature set and y is the target
for train_index, test_index in rkf.split(t2):
     X_train, X_test = tr_x.loc[train_index], tr_x.loc[test_index]
     y_train, y_test = tr_y.loc[train_index], tr_y.loc[test_index]
     model.fit(X_train,y_train)
     pred = model.predict(X_test)
     acc = accuracy_score(y_test,pred)
     #print(acc)
     acc1.append(acc)
print('Average = ' ,statistics.mean(acc1))