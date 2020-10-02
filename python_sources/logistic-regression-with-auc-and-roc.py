# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
spine=pd.read_csv('../input/Dataset_spine.csv')
spine.head()
spine.columns
spine['Unnamed: 13'].unique()
spine.isnull().sum()
spine=spine.drop(['Unnamed: 13'],axis=1)
spine['Class_att'].unique()
spine.dtypes
dict1={"Abnormal":0,"Normal":1}
dict1
spine['Class_att']=spine['Class_att'].map(dict1)
spine['Class_att']
spine.describe()
x=spine.drop(["Class_att"],axis=1)
y=spine["Class_att"]
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
spine_scaled=sc.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(x_train,y_train)
pred=lr_model.predict(x_test)
pred[2]
lr_model.score(x_test,y_test)
y_test.iloc[1]
from sklearn.metrics import roc_auc_score,roc_curve

import scikitplot as skplt
#skplt.metrics.plot_roc_curve(pred, y_test)
#plt.show()
y_pred_proba = lr_model.predict_proba(x_test)[::,1]#probability estimates
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.