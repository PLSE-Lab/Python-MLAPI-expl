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

inp=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')

y_train=inp.loc[:,'Survived']
x_train=inp.drop(['Survived','Name','Ticket','Cabin'],axis=1)
x_test=test.drop(['Name','Ticket','Cabin'],axis=1)

from sklearn.impute import SimpleImputer
SImean=SimpleImputer(missing_values=np.nan,strategy='mean')
x_train.loc[:,['Age']]=SImean.fit_transform(x_train.loc[:,['Age']])
x_test.loc[:,['Age']]=SImean.fit_transform(x_test.loc[:,['Age']])
x_test.loc[:,['Fare']]=SImean.fit_transform(x_test.loc[:,['Fare']])

SImode=SimpleImputer(strategy='most_frequent')
x_train.loc[:,['Embarked']]=SImode.fit_transform(x_train.loc[:,['Embarked']])

x_train=pd.get_dummies(x_train,columns=['Pclass','Sex','Embarked'])
x_test=pd.get_dummies(x_test,columns=['Pclass','Sex','Embarked'])

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)

prediction=classifier.predict(x_test)

submission=test.loc[:,['PassengerId']]
submission['Survived']=prediction
