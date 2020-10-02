# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/voice.csv')
# changing males->1, females->0
num = LabelEncoder()
df['label'] = num.fit_transform(df['label'].astype('str'))
data = df.iloc[:,:20]
label = df.iloc[:,-1]

# scaling values between -1 and 1
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

x_train,x_test,y_train,y_test=train_test_split(data,label,test_size=0.2)

#LOGISTIC REGRESSION
logic = LogisticRegression(C=1e5)
logic.fit(x_train,y_train)
print ('Logistic Regression accuracy: %f' % logic.score(x_test,y_test))


#STOCHASTIC GRADIENT
sgd = SGDClassifier(loss='hinge',penalty="l1")
sgd.fit(x_train,y_train)
print ("Stochastic Gradient accuracy: %f" % sgd.score(x_test,y_test))

#SVM
svm = SVC()
svm.fit(x_train,y_train)
print ("Support Vector Machines accuracy: %f" % svm.score(x_test,y_test))

#RANDOM FOREST
forest = RandomForestClassifier(n_estimators=100)
forest.fit(x_train,y_train)
print ("Random Forest accuracy: %f" % forest.score(x_test,y_test))
