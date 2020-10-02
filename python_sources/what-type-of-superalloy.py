# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/selestedsuperalloy.csv')
print(data.head(3))

typelabels, typelevels = pd.factorize(data['type'])
data['type']=(typelabels)



from sklearn.preprocessing import scale
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

scaler = MinMaxScaler()
data[['Cr','Co','Mo','W','Ta','Nb','Al','Ti','Fe','C','B','Zr','Re','Hf','Ru'
]] = scaler.fit_transform(data[['Cr','Co','Mo','W','Ta','Nb','Al','Ti','Fe','C','B','Zr','Re','Hf','Ru'
]])


X=data.iloc[:,2:]
y=data.iloc[:,1:2]
#print('X:')
print(X.shape)
#print('y:')
y=np.ravel(y)
print(y.shape)
svc = svm.SVC(kernel='linear')
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=4)
svc.fit(X_train, y_train)
y_pred=svc.predict(X_test)
print(accuracy_score(y_test, y_pred))
