# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.linear_model import SGDClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.linear_model import SGDClassifier

data=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
new_data=data.copy()
new_data['diagnosis']=new_data['diagnosis'].str.replace('M','0')
new_data['diagnosis']=new_data['diagnosis'].str.replace('B','1')
new_data['diagnosis']=new_data['diagnosis'].astype('int64')
new_data=new_data.drop(['id','Unnamed: 32'],axis=1)
column=new_data.columns
print(column)
#plt.scatter('perimeter_mean','diagnosis',data=new_data)
#sns.pairplot(new_data)
corr=new_data.corr()
x=new_data[new_data.columns.drop('diagnosis')].drop(['smoothness_mean','texture_se','smoothness_se','symmetry_se','symmetry_mean','fractal_dimension_mean'],axis=1)
y=new_data['diagnosis']
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
sgd_cls=SGDClassifier(random_state=42)
sgd_cls.fit(X_train,y_train)
#prior to prediction evaluation
y_pred=cross_val_predict(sgd_cls,X_train,y_train,cv=3)
matrix=confusion_matrix(y_train,y_pred)
precision=precision_score(y_train,y_pred)
f1=f1_score(y_train,y_pred)

#actual prediction
preds=sgd_cls.predict(X_test)
p_matrix=confusion_matrix(y_test,preds)
p_precision=precision_score(y_test,preds)
p_f1=f1_score(y_test,preds)
#save prediction to excel
pred_data=pd.DataFrame({'actual':y_test,'predicted':preds})
pred_data.to_excel('/kaggle/working/Pediction.xlsx',index=None)