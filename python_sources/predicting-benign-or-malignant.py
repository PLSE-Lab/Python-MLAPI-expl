# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data=pd.read_csv('../input/data.csv')
data.head()
data.columns
#Diagnosis is our response variable
data['diagnosis'].unique()
data.isnull().sum()
#Drops unnecessary columns
data=data.drop(['Unnamed: 32','id'],axis=1)
data.dtypes
#drops response variable from cleaned set
x=data.drop(['diagnosis'],axis=1)
y=data['diagnosis']
from sklearn import preprocessing
#Scaling the cleaned set
x=preprocessing.scale(x)
from sklearn.model_selection import train_test_split
#Splitting the set into test/train
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=10)
train_x.shape
np.shape(train_x[1])
#Runs decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(train_x,train_y)
pred=dt.predict(test_x)
from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(pred,test_y)
accuracy_score(pred,test_y)
#Runs Random Forest algorithm
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(train_x,train_y)
pred1=rf.predict(test_x)
confusion_matrix(pred1,test_y)
accuracy_score(pred1,test_y)
from sklearn.preprocessing import LabelEncoder
#Convert the String type variable of training label to integer so that they can be mapped to keras
encode=LabelEncoder()
y_label_train=encode.fit_transform(train_y)
y_label_test=encode.fit_transform(test_y)
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
model=Sequential()
model.add(Dense(128,activation="relu",input_dim=np.shape(train_x)[1]))
model.add(Dropout(0.25))
model.add(Dense(32,activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1,activation="sigmoid"))
#applying stochastic gradient descent as optimizer 
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_x,y_label_train,batch_size=10,epochs=10)
pred2=model.predict(test_x)
#applying naive bayes model
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(train_x,train_y)
pred3=nb.predict(test_x)
confusion_matrix(pred3,test_y)
accuracy_score(pred3,test_y)
#DEcision Tree-90%
#Random Forest-97%
#Deep learning with keras-97.5%
#Naive Bayes-95


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.