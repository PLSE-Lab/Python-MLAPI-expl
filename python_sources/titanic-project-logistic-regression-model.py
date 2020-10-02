# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import seaborn as sns

df_titanic_train = pd.read_csv("../input/train_data.csv")
df_titanic_test = pd.read_csv("../input/test_data.csv")

sns.countplot(df_titanic_train['Survived'], label="count")
plt.show()

sns.countplot(df_titanic_test['Survived'], label="count")
plt.show()

df_titanic_train.shape
df_titanic_train.keys()

#************** Dropping the columns  that are not needed *****************
df_titanic_train = df_titanic_train.drop(['Unnamed: 0','PassengerId'],axis =1)
df_titanic_test = df_titanic_test.drop(['Unnamed: 0','PassengerId'],axis =1)
df_titanic_train.keys()

plt.figure(figsize= (20,20))
sns.heatmap(df_titanic_train.corr(), annot=True)
plt.show()

#*********** Dropping the columns  that are not needed and prep data********
X_train = df_titanic_train.drop('Survived',axis=1).values
y_train = df_titanic_train['Survived'].values
X_test = df_titanic_test.drop('Survived',axis=1).values
y_test = df_titanic_test['Survived'].values


#************** Running the logistic regression model *****************
from sklearn.linear_model import LogisticRegression 
classifier_titanic = LogisticRegression(random_state =0)
classifier_titanic.fit(X_train,y_train)
y_predict = classifier_titanic.predict(X_test)

#************** Evaluating the model *****************
from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot = True)
plt.show()

print(classification_report(y_test,y_predict))