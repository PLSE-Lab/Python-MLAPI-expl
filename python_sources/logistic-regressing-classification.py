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

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV


dataset = pd.read_csv("/kaggle/input/online-shoppers-intention/online_shoppers_intention.csv")


X = dataset.drop(['Revenue','Month'],axis=1)
y = dataset.iloc[:,-1].values
y = np.reshape(y,(12330,1))



label_encoder_X = LabelEncoder()
X['VisitorType'] = label_encoder_X.fit_transform(X['VisitorType'])

label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)
y = np.reshape(y,(12330,1))



imputer = SimpleImputer(strategy='median')
imputer = imputer.fit(X)
X=imputer.transform(X)


scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.20,random_state = 0)



logistic_regression = LogisticRegression(random_state=0,max_iter=50,penalty='l2')
logistic_regression.fit(X_train,y_train)
y_pred = logistic_regression.predict(X_test)


#grid_param={"penalty":['l1','l2'],
#            "max_iter":[50,80,100,150]}

#CV = GridSearchCV(estimator=logistic_regression,param_grid=grid_param,cv=2)
#CV.fit(X_train,y_train)
#CV.best_params_

cm = confusion_matrix(y_true= y_test, y_pred=y_pred)
acc_score = accuracy_score(y_true=y_test, y_pred = y_pred)


def predict(arr):
    if arr[15] == "Returning_Visitor":
        arr[15] = 2
    
    elif arr[15] == "New_Visitor":
        arr[15] = 0
    else:
        arr[15] = 1
        
    arr = np.asarray(arr)
    

    arr = np.delete(arr,[10,17])
    arr = np.reshape(arr,(1,16))
    scaler = StandardScaler()
    arr = scaler.fit_transform(arr)

    arr = np.reshape(arr,(1,16))
    pred = logistic_regression.predict(arr)
    if pred==0:
        return "false"
    else:
        return "true"

count_false=0
count_true=0
for i in range(0,11):
    pred = predict(dataset.iloc[i,:].values)
    print(pred)
    if pred == "false":
        count_false +=1
    elif pred == "true":
        count_true += 1