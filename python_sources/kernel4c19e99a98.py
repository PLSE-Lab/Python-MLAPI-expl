# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import pandas as pd

pp = pd.read_csv("../input/breast-cancer.csv")
pp.head(5)
print(pp)

pp.diagnosis[pp.diagnosis=='M'] = 1
pp.diagnosis[pp.diagnosis=='B'] = 0

data=pp.drop(["id"],axis=1)
pp.dropna(axis='columns')
data.head(5)
data2=data.dropna(axis='columns')
data2.head(5)


from sklearn import svm 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

from sklearn.model_selection import train_test_split
Y = data2.diagnosis 
print(data2)
data3 = data2
X = data3.drop(['diagnosis'],axis=1)
print(X)
from sklearn import model_selection

data2.diagnosis=data2.diagnosis.astype('int')
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, data2.diagnosis, test_size=0.4, random_state=7)

model.fit(X_train,Y_train)
model.predict(X_test)
model.score(X_test,Y_test)
model.predict_proba(X_test)

from sklearn import svm 
model1 = svm.SVC(gamma='scale')
model1.fit(X_train,Y_train)
model1.predict(X_test)
model1.score(X_test,Y_test)
#model1.predict_proba(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection

RFmodel2 = RandomForestClassifier(n_estimators = 20)
RFmodel2.fit(X_train,Y_train)
RFmodel2.predict(X_test)
RFmodel2.score(X_test,Y_test)
#0.9649 accuracy 




