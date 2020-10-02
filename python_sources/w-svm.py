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

import numpy as np
import pandas as pd

train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

data = pd.DataFrame(pd.concat([train,test]))
data.reset_index()
data.to_csv("data.csv")
print(data.shape)

from sklearn.preprocessing import MinMaxScaler,StandardScaler,PolynomialFeatures

# 1.deal with missing values

age_means = data.groupby(['Sex', 'Pclass'])['Age']
data["Age"] = age_means.transform(lambda x: x.fillna(x.mean()))
Fare_mean = data.groupby(['Sex', 'Pclass'])['Fare']
data["Fare"]  = Fare_mean.transform(lambda x: x.fillna(x.mean()))
data["Embarked"] = data["Embarked"].fillna("S") #with motst frequent value

# 2.generate feature with intuition
data['Kid'] = data["Age"].apply(np.log1p)
data["Age"] = abs(data["Age"] - data["Age"].median())
data['Name_len'] = data['Name'].apply(len)

data['Title'] = data['Name'].str.split(', ').str[1]
data['Title'] = data['Title'].str.split('.').str[0]
data['Title'] = data['Title'].replace(['the Countess','Capt', 'Col', 'Don','Dona', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Rare')
data['Title'] = data['Title'].replace(['Lady','Mlle', 'Ms'], 'Mrs')
data['Title'] = data['Title'].replace(['Mme'], 'Miss')

# 3.generate dummy variables
data = pd.get_dummies(data,columns=['Sex','Embarked','Pclass','Title'], drop_first=True)
data.to_csv('data.csv')

# 4.feature convert
data['SibSp'] = data['SibSp'].apply(np.log1p)
data['Parch'] = data['Parch'].apply(np.log1p)
data['Fare'] = data['Fare'].apply(np.log1p)
data["Age"] = data['Age'].apply(np.log1p)

# 5.return data
ind_cols = ['Sex_male','Embarked_Q','Embarked_S','Pclass_2','Pclass_3','Title_Miss','Title_Mr','Title_Mrs','Title_Rare','Age','Kid','SibSp','Parch','Fare','Name_len']

print(data.isnull().sum())
X = data.loc[:,ind_cols].values
y = data.loc[:,'Survived'].values
print(data.tail())
sub = pd.DataFrame(test['PassengerId'])
print(sub)
# 6.feature scaling
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# 7.poly feature
poly = PolynomialFeatures(degree=1)
poly.fit(X)
X = poly.transform(X)
print("After poly progress X's shape is {}".format(X.shape))
# train data
X_train = X[:891,:]
y_train = y[:891]
X_test = X[891:,:]
print("X_test size is {}".format(X_test.shape))
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
xf = pd.DataFrame(X_train)
yf = pd.DataFrame(y_train)
clf_ = SVC(tol=1e-6,max_iter=10000,C=1)
clf_ = clf_.fit(X_train,y_train)
scores = cross_val_score(clf_,X_train,y_train,cv=10)
print("score is {}".format(scores.mean()))

## output submission
sub['Survived'] =  clf_.predict(X_test)
sub['Survived'] = sub['Survived'].apply(int)
sub.to_csv("gender_submission.csv",index=False)
