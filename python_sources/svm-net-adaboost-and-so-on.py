# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

df=pd.read_csv("../input/voice.csv")
print(df.shape)

x_list=df.drop("label",axis=1).values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_list)
x_list = scaler.transform(x_list)
from sklearn.preprocessing import LabelEncoder
gender_encoder = LabelEncoder()
y_list = gender_encoder.fit_transform(df["label"].values)
#y_list=[1 if x=="male" else 0 for x in df["label"].values]
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_list, y_list, test_size=0.2)
from sklearn.svm import SVC
clf=SVC()
clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)
from sklearn import metrics
print("SVM:",metrics.accuracy_score(y_test, y_pred))
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),algorithm="SAMME",n_estimators=200, learning_rate=0.8)
ada.fit(x_train, y_train)
y_pred=ada.predict(x_test)
print("ADA:",metrics.accuracy_score(y_test, y_pred))
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("LR:",metrics.accuracy_score(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier
rdf=RandomForestClassifier()
rdf.fit(x_train,y_train)
y_pred=rdf.predict(x_test)
print("RDF:",metrics.accuracy_score(y_test, y_pred))

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
def conv1To2(lst):
    return np.array([[0,1] if l==1 else [1,0] for l in lst])
y_train=conv1To2(y_train)
y_test=conv1To2(y_test)
#y_train=lb.fit_transform(y_train)
#y_test=lb.fit_transform(y_test)

model = Sequential()
model.add(Dense(15,input_dim=20, init='uniform'))
model.add(Activation('relu'))
model.add(Dense(10, init='uniform'))
model.add(Activation('relu'))
model.add(Dense(2, init='uniform'))
model.add(Activation('softmax'))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='mean_squared_error', optimizer=sgd, class_mode="binary")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])  
model.fit(x_train, y_train, nb_epoch=200, batch_size=200,verbose=0)
#print("Net:",model.evaluate(x_test, y_test))
y_pred=model.predict_proba(x_test)
print(y_pred)
#print("Net:",model.evaluate(y_test, y_pred))

