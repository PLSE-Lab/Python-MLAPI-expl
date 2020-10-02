# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
from sklearn.utils import resample

from sklearn.preprocessing import StandardScaler

data=pd.read_csv("../input/train.csv")

data.columns


data.isnull()
data.isnull().sum()

a=data.values[:,100]
b=data.values[:,101]
c=data.values[:,102]

data1=np.vstack((a,b,c))
data1=data1.T

for i in range(data1.shape[1]):
    p=[]
    q=[]
    t=[]
    r=data1[:,i]
    p=r[r!='yes']
    q=p[p!='no']
    t=data1[:,i]
    t[t=='yes']=np.max(q)
    t[t=='no']=0
    data1[:,i]=t

data.drop(['Id','idhogar','dependency','edjefe','edjefa'],axis=1,inplace=True)

data_train=np.hstack((data1,data))

data_train.shape
imputer=Imputer()

data2=imputer.fit_transform(data_train)

data_train=pd.DataFrame(data2)

A=0
B=0
C=0
D=0

for i in range(data_train.shape[0]):
    if (data_train.values[i,data_train.shape[1]-1]==1):
        A=A+1
    elif (data_train.values[i,data_train.shape[1]-1]==2):
        B=B+1
    elif (data_train.values[i,data_train.shape[1]-1]==3):
        C=C+1
    elif (data_train.values[i,data_train.shape[1]-1]==4):
        D=D+1
print(A,B,C,D)

df_train=data_train.sample(frac=1).reset_index(drop=True)

#scaler=StandardScaler()
#df_train=scaler.fit_transform(df_train)
#df_train=pd.DataFrame(df_train)

train=df_train.values

train1=train[train[:,train.shape[1]-1]==1]
train2=train[train[:,train.shape[1]-1]==2]
train3=train[train[:,train.shape[1]-1]==3]
train4=train[train[:,train.shape[1]-1]==4]

train1=resample(train1,replace=True,n_samples=D,random_state=0)
train2=resample(train2,replace=True,n_samples=D,random_state=0)
train3=resample(train3,replace=True,n_samples=D,random_state=0)

train_new=np.vstack((train1,train2,train3,train4))


#Overall classifier (Random Forest)
X=train_new[:,0:train_new.shape[1]-1]

Y=train_new[:,train_new.shape[1]-1]

cls=RandomForestClassifier()
cls.fit(X,Y)

pred=cls.predict(X)


CM=confusion_matrix(Y_test,pred_test)
print("Accuracy of class 1:", CM[0,0]/(CM[0,0]+CM[0,1]+CM[0,2]+CM[0,3]))
print("Accuracy of class 2:", CM[1,1]/(CM[1,0]+CM[1,1]+CM[1,2]+CM[1,3]))
print("Accuracy of class 3:", CM[2,2]/(CM[2,0]+CM[2,1]+CM[2,2]+CM[2,3]))
print("Accuracy of class 4:", CM[3,3]/(CM[3,0]+CM[3,1]+CM[3,2]+CM[3,3]))

#preparing test for prediction
data_test=pd.read_csv("../input/test.csv")

d=data_test.values[:,100]
e=data_test.values[:,101]
f=data_test.values[:,102]

df1=np.vstack((d,e,f))
df1=df1.T

for i in range(df1.shape[1]):
    p=[]
    q=[]
    t=[]
    r=df1[:,i]
    p=r[r!='yes']
    q=p[p!='no']
    t=df1[:,i]
    t[t=='yes']=np.max(q)
    t[t=='no']=0
    df1[:,i]=t

data_test.drop(['Id','idhogar','dependency','edjefe','edjefa'],axis=1,inplace=True)

df_test=np.hstack((df1,data_test))

imputer=Imputer()

data_test=imputer.fit_transform(data_test)

data_test=pd.DataFrame(data_test)

X_test=data_test.values[:,0:data_test.shape[1]-1]

Y_test=data_test.values[:,data_test.shape[1]-1]

pred_test=cls.predict(X_test)

pred_test=pd.DataFrame(pred_test)

pred_test.to_csv("prediction.csv")

