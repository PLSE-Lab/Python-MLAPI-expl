# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
files=[]
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
       files.append(os.path.join(dirname, filename))

submission=pd.read_csv(files[0])
test=pd.read_csv(files[1])
train=pd.read_csv(files[2])

y = train["Survived"]

X=train.drop(["Survived","Name","PassengerId","Ticket"], axis=1)
test=test.drop(["Name","PassengerId","Ticket"], axis=1)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder=LabelEncoder()


test["Sex"] =labelEncoder.fit_transform(test["Sex"].astype(str))
test["Cabin"] =labelEncoder.fit_transform(test["Cabin"].astype(str))
test["Embarked"] =labelEncoder.fit_transform(test["Embarked"].astype(str))


X["Sex"] =labelEncoder.fit_transform(X["Sex"].astype(str))
X["Cabin"] =labelEncoder.fit_transform(X["Cabin"].astype(str))
X["Embarked"] =labelEncoder.fit_transform(X["Embarked"].astype(str))



from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['Sex','Age','SibSp','Parch','Fare', 'Cabin', 'Embarked',]

features_log_minmax_transform = pd.DataFrame(data = X)
features_log_minmax_transform[numerical] = scaler.fit_transform(X[numerical])

scaler_test = MinMaxScaler() # default=(0, 1)


features_log_minmax_transform_test = pd.DataFrame(data = test)
features_log_minmax_transform_test[numerical] = scaler_test.fit_transform(test[numerical])


from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
X_imputed = my_imputer.fit_transform(features_log_minmax_transform)

my_imputer_test = SimpleImputer()
X_imputed_test = my_imputer_test.fit_transform(features_log_minmax_transform_test)


from sklearn.model_selection import train_test_split
# Set a random state.
X_train, X_test, y_train, y_test = train_test_split(X_imputed,y, test_size=0.25, random_state=42)


from sklearn.metrics import fbeta_score, make_scorer
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
accuracy= fbeta_score(y_test,prediction,beta=2)


print(accuracy)

submittion_predict=clf.predict(X_imputed_test)
print(submittion_predict)

import csv
row = ['PassengerId',' Survived']
with open('Submition.csv', 'r') as readFile:
reader = csv.reader(readFile)
lines = list(reader)
lines[2] = row
with open('Submition.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)
readFile.close()
writeFile.close()

