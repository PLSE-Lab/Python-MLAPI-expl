# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load DataSet
df=pd.read_csv('../input/titanicdataset-traincsv/train.csv')
#Processing missing values
df.isnull().sum()
df=df.drop(columns=['PassengerId','Cabin'])
df['Age'].mean()
df['Age'].fillna(df['Age'].mean(), inplace=True)
df=df.dropna()
#Converting categories into numerical values

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['Sex'] = LE.fit_transform(df['Sex'])
df['Embarked'] = LE.fit_transform(df['Embarked'])
#Train & Test Data

labeles=df['Survived']
features=df[['Pclass','Sex','SibSp','Parch','Fare','Embarked']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labeles, test_size=0.2,random_state=5 )

from sklearn import tree
clf=tree.DecisionTreeClassifier(min_samples_split=70)
clf.fit(X_train,y_train)
pred=clf.predict(X_test)
#accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)
from sklearn.metrics import accuracy_score
accur=accuracy_score(y_test,pred)
print(accur)