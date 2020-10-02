# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# In[2]:

data=pd.read_csv('ga_map_compact_all_58K.CSV', skiprows=[0])
data_iv_x= data.OpCode
data_dv_y = data['Cd-SSN']

print('Train columns with null values:\n', data_iv_x.isnull().sum())
print("-"*10)

print('Test/Validation columns with null values:\n', data_dv_y.isnull().sum())
print("-"*10)

data_iv_x.describe()
full_data= data.loc[:,['OpCode','Cd-SSN']]
full_data.describe()

full_data["OpCode"]=full_data["OpCode"].fillna(' Send authentication info')

full_data.describe()

#categorising data
OpCode = pd.get_dummies( full_data.OpCode , prefix='OpCode' )
OpCode.head()
OpCode.drop(['OpCode_Send authentication info'], axis=1, inplace=True)
OpCode.head()
full_data = full_data.join(OpCode)

full_data.describe()

#categorising data
Cd_SSN = pd.get_dummies( full_data["Cd-SSN"] , prefix='SSN' )
Cd_SSN.head()
Cd_SSN.drop(['SSN_Home location register'], axis=1, inplace=True)
Cd_SSN.head()
full_data = full_data.join(Cd_SSN)

full_data.drop(['OpCode'],axis=1, inplace=True)
full_data.drop(['Cd-SSN'],axis=1, inplace=True)

x = full_data.iloc[:,0:27].values
y = full_data.iloc[:,26:].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

y_train_1=y_train[:,0]
y_train_1=y_train_1.transpose()
y_train_1
y_test_1=y_test[:,0]
y_test_1.transpose()


y_train_2=y_train[:,1]
y_train_2=y_train_2.transpose()
y_train_2
y_test_2=y_test[:,1]
y_test_2.transpose()

y_train_3=y_train[:,2]
y_train_3=y_train_3.transpose()
y_train_3
y_test_3=y_test[:,2]
y_test_3.transpose()

y_train_4=y_train[:,3]
y_train_4=y_train_4.transpose()
y_train_4
y_test_4=y_test[:,3]
y_test_4.transpose()

y_train_5=y_train[:,4]
y_train_5=y_train_5.transpose()
y_train_5
y_test_5=y_test[:,4]
y_test_5.transpose()

y_train_6=y_train[:,5]
y_train_6=y_train_6.transpose()
y_train_6
y_test_6=y_test[:,5]
y_test_6.transpose()


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC

model_1 = SVC()
model_2 = SVC()
model_3 = SVC()
model_4 = SVC()
model_5 = SVC()
model_6 = SVC()

model_1.fit(X_train,y_train_1)
model_2.fit(X_train,y_train_2)
model_3.fit(X_train,y_train_3)
model_4.fit(X_train,y_train_4)
model_5.fit(X_train,y_train_5)
model_6.fit(X_train,y_train_6)

y_pred_1=model_1.predict(X_test)
y_pred_2=model_2.predict(X_test)
y_pred_3=model_3.predict(X_test)
y_pred_4=model_4.predict(X_test)
y_pred_5=model_5.predict(X_test)
y_pred_6=model_6.predict(X_test)


from sklearn.metrics import confusion_matrix
cm_1 = confusion_matrix(y_test_1, y_pred_1)
cm_2 = confusion_matrix(y_test_2, y_pred_2)
cm_3 = confusion_matrix(y_test_3, y_pred_3)
cm_4 = confusion_matrix(y_test_4, y_pred_4)
cm_5 = confusion_matrix(y_test_5, y_pred_5)
cm_6 = confusion_matrix(y_test_6, y_pred_6)
