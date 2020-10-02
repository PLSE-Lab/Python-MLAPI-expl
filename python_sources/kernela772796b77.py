# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("../input/mushrooms.csv")
x=df.iloc[:,1:].values
y=pd.DataFrame(df.iloc[:,0].values)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
l1=LabelEncoder()
x[:,0]=l1.fit_transform(x[:,0])
l2=LabelEncoder()
x[:,1]=l2.fit_transform(x[:,1])
l3=LabelEncoder()
x[:,2]=l3.fit_transform(x[:,2])
l4=LabelEncoder()
x[:,3]=l4.fit_transform(x[:,3]) 
l5=LabelEncoder()
x[:,4]=l5.fit_transform(x[:,4]) 
l6=LabelEncoder()
x[:,5]=l6.fit_transform(x[:,5]) 
l7=LabelEncoder()
x[:,6]=l7.fit_transform(x[:,6]) 
l8=LabelEncoder()
x[:,7]=l8.fit_transform(x[:,7])
l9=LabelEncoder()
x[:,8]=l9.fit_transform(x[:,8])
l10=LabelEncoder()
x[:,9]=l10.fit_transform(x[:,9])
l11=LabelEncoder()
x[:,10]=l11.fit_transform(x[:,10])
l12=LabelEncoder()
x[:,11]=l12.fit_transform(x[:,11])
l13=LabelEncoder()
x[:,12]=l13.fit_transform(x[:,12])
l14=LabelEncoder()
x[:,13]=l14.fit_transform(x[:,13]) 
l15=LabelEncoder()
x[:,14]=l15.fit_transform(x[:,14]) 
l16=LabelEncoder()
x[:,15]=l16.fit_transform(x[:,15]) 
l17=LabelEncoder()
x[:,16]=l17.fit_transform(x[:,16]) 
l18=LabelEncoder()
x[:,17]=l18.fit_transform(x[:,17])
l19=LabelEncoder()
x[:,18]=l19.fit_transform(x[:,18])
l20=LabelEncoder()
x[:,19]=l20.fit_transform(x[:,19])
l21=LabelEncoder()
x[:,20]=l21.fit_transform(x[:,20])
l22=LabelEncoder()
x[:,21]=l22.fit_transform(x[:,21])
ly=LabelEncoder()
y=ly.fit_transform(y)

onehotencoder1=OneHotEncoder(categorical_features=[0])
x=onehotencoder1.fit_transform(x).toarray()
x=x[:,1:]

onehotencoder2=OneHotEncoder(categorical_features=[5])
x=onehotencoder2.fit_transform(x).toarray()
x=x[:,1:]

onehotencoder3=OneHotEncoder(categorical_features=[8])
x=onehotencoder3.fit_transform(x).toarray()
x=x[:,1:]

onehotencoder4=OneHotEncoder(categorical_features=[18])
x=onehotencoder4.fit_transform(x).toarray()
x=x[:,1:]

onehotencoder5=OneHotEncoder(categorical_features=[29])
x=onehotencoder5.fit_transform(x).toarray()
x=x[:,1:]

onehotencoder6=OneHotEncoder(categorical_features=[41])
x=onehotencoder6.fit_transform(x).toarray()
x=x[:,1:]

onehotencoder7=OneHotEncoder(categorical_features=[45])
x=onehotencoder7.fit_transform(x).toarray()
x=x[:,1:]

onehotencoder8=OneHotEncoder(categorical_features=[48])
x=onehotencoder8.fit_transform(x).toarray()
x=x[:,1:]

onehotencoder9=OneHotEncoder(categorical_features=[51])
x=onehotencoder9.fit_transform(x).toarray()
x=x[:,1:]

onehotencoder10=OneHotEncoder(categorical_features=[59])
x=onehotencoder10.fit_transform(x).toarray()
x=x[:,1:]

onehotencoder11=OneHotEncoder(categorical_features=[68])
x=onehotencoder11.fit_transform(x).toarray()
x=x[:,1:]

onehotencoder12=OneHotEncoder(categorical_features=[72])
x=onehotencoder12.fit_transform(x).toarray()
x=x[:,1:]

onehotencoder13=OneHotEncoder(categorical_features=[76])
x=onehotencoder13.fit_transform(x).toarray()
x=x[:,1:]

onehotencoder14=OneHotEncoder(categorical_features=[84])
x=onehotencoder14.fit_transform(x).toarray()
x=x[:,1:]

onehotencoder15=OneHotEncoder(categorical_features=[89])
x=onehotencoder15.fit_transform(x).toarray()
x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

import keras
from keras.layers.core import Dense,Flatten,Dropout
from keras.models import Sequential
from keras.optimizers import Adam

model=Sequential()
model.add(Dense(units=48,input_dim=95,kernel_initializer="glorot_uniform",activation="relu"))
model.add(Dropout(rate=0.1))
model.add(Dense(units=24,kernel_initializer="glorot_uniform",activation="relu"))
model.add(Dropout(rate=0.1))
model.add(Dense(units=12,kernel_initializer="glorot_uniform",activation="relu"))
model.add(Dropout(rate=0.1))
model.add(Dense(units=6,kernel_initializer="glorot_uniform",activation="relu"))
model.add(Dropout(rate=0.1))
model.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))

model.compile(optimizer=Adam(lr=0.0001),loss="binary_crossentropy",metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=24,epochs=50,validation_split=0.1,verbose=2,shuffle=True)

y_pred=model.predict(x_test)
y_pred=(y_pred>0.5)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
