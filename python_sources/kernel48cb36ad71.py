# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,f1_score
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.linear_model import LogisticRegression
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]


pulsar=pd.read_csv('/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv')
pulsar1=pulsar.copy()
sns.pairplot(pulsar1)
corr=pulsar1.corr()
x=pulsar1[pulsar1.columns].drop(['target_class',' Standard deviation of the integrated profile',' Mean of the DM-SNR curve',' Excess kurtosis of the DM-SNR curve',' Skewness of the DM-SNR curve'],axis=1)
y=pulsar['target_class']
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


lg=LogisticRegression()
lg.fit(X_train,y_train)
pred=cross_val_predict(lg,X_train,y_train,cv=5)
matrix=confusion_matrix(y_train,pred)
precision=precision_score(y_train,pred)
f1score=f1_score(y_train,pred)
accuracy=accuracy_score(y_train,pred)

pred1=lg.predict(X_test)
matrix1=confusion_matrix(y_test,pred1)
precision1=precision_score(y_test,pred1)
f1score1=f1_score(y_test,pred1)
accuracy1=accuracy_score(y_test,pred1)

data=pd.DataFrame(pred1)
data.to_csv('/kaggle/working/pred.csv',index=None)
