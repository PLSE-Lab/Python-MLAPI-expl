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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

test=pd.read_csv('/kaggle/input/summeranalytics2020/test.csv')
train=pd.read_csv('/kaggle/input/summeranalytics2020/train.csv')
ss=pd.read_csv('/kaggle/input/summeranalytics2020/Sample_submission.csv')

train.head()
train.describe()
corr = train.corr()
import seaborn as sns
sns.heatmap(corr,annot=True)
train.columns
train.drop(["Behaviour","EmployeeNumber"],axis=1,inplace=True)
train.head()



train["BusinessTravel"].value_counts()
train["Department"].value_counts()
train["EducationField"].value_counts()
train["JobRole"].value_counts()
le = LabelEncoder()


train["BusinessTravel"] = le.fit_transform(train["BusinessTravel"])

train["BusinessTravel"].value_counts()

train["Department"] = le.fit_transform(train["Department"])

train["Department"].value_counts()

train["EducationField"] = le.fit_transform(train["EducationField"])

train["JobRole"] = le.fit_transform(train["JobRole"])

train["Gender"] = le.fit_transform(train["Gender"])

train["MaritalStatus"] = le.fit_transform(train["MaritalStatus"])

train["OverTime"] = le.fit_transform(train["OverTime"])

train.iloc[0]



train.head()

XData = train.drop(["Attrition"],axis=1)
YData = train["Attrition"]



test.head()

test.drop(["Behaviour","EmployeeNumber"],axis=1,inplace=True)



test["BusinessTravel"] = le.fit_transform(test["BusinessTravel"])

test["Department"] = le.fit_transform(test["Department"])

test["EducationField"] = le.fit_transform(test["EducationField"])

test["JobRole"] = le.fit_transform(test["JobRole"])

test["Gender"] = le.fit_transform(test["Gender"])

test["MaritalStatus"] = le.fit_transform(test["MaritalStatus"])

test["OverTime"] = le.fit_transform(test["OverTime"])


train.head()
test.head()
X_train, X_test, y_train, y_test = train_test_split(XData, YData, test_size=0.33, random_state=42)
YData.value_counts()
from sklearn.neighbors import KNeighborsClassifier
model1 = KNeighborsClassifier(n_neighbors=3)
model1.fit(X_train,y_train)
ypred = model1.predict(X_test)
model1.score(X_test,y_test)
ypred[:5] , y_test[:5]
from sklearn.linear_model import LogisticRegression
model2 = LogisticRegression(solver='saga',max_iter=5000)
model2.fit(X_train,y_train)
ypred = model2.predict(X_test)
model2.score(X_test,y_test)
from sklearn import svm
model3 = svm.SVC(probability=True)
model3.fit(X_train,y_train)
ypred = model3.predict(X_test)
model3.score(X_test,y_test)
ypred[:5] , y_test[:5]
a=model3.predict_proba(test)[:,1]
b=a
ss=ss.drop(['Attrition'],axis=1)
ss['Attrition']=b
ss.to_csv('submission1.csv',index=False)
ss.head()