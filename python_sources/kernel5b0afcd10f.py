# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
concrete = pd.read_csv('../input/concrete.csv')
concrete.columns



concrete.strength.min()
concrete.strength.max()
#converting Sales column to categorical
concrete_dummy=pd.DataFrame(pd.cut(concrete.strength,bins=[2,25,50,85],labels=['low','medium','high'],right=True))
concrete_dummy
concrete_dummy=pd.concat([concrete_dummy,pd.get_dummies(concrete.loc[:,concrete.columns.difference(['strength'])])],axis=1)
concrete_dummy

from sklearn.model_selection import train_test_split
train,test = train_test_split(concrete_dummy,test_size = 0.3,random_state=0)
trainX = train.drop(["strength"],axis=1)
trainY = pd.DataFrame(train.strength) 
testX = test.drop(["strength"],axis=1)
testY = pd.DataFrame(test.strength) 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# to normalise the data we use this scaler
scaler.fit(trainX)

trainX = scaler.transform(trainX)
testX = scaler.transform(testX)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes = (30,30))
mlp.fit(trainX,trainY)

pred_train = mlp.predict(trainX)
pred_train
pred_test = mlp.predict(testX)
pred_test

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(testY,pred_test))

print(confusion_matrix(trainY,pred_train))




# Any results you write to the current directory are saved as output.