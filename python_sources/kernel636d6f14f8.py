# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as snp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, confusion_matrix
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')

# checking head of the test data
print(train_data.head())

# analysing some information from train data
cf.go_offline()
snp.set_style('whitegrid')
fig,axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
snp.countplot(x='Survived', data=train_data, hue='Sex', ax=axes[0][0])
snp.countplot(x='Survived', data=train_data, hue='Pclass', ax=axes[0][1])
snp.distplot(train_data['Age'], kde=False, bins=30, ax=axes[1][0])
snp.boxplot(x='Pclass', y='Age', data=train_data, ax=axes[1][1])

# assign dummies variable for male and Embarked as our model can read them as an input
sex = pd.get_dummies(data=train_data['Sex'], drop_first=True)
sex_test = pd.get_dummies(data=test_data['Sex'], drop_first=True)
embarked = pd.get_dummies(data=train_data['Embarked'], drop_first=True)
embarked_test = pd.get_dummies(data=test_data['Embarked'], drop_first=True)
train_data = pd.concat([train_data, sex, embarked], axis=1)
test_data = pd.concat([test_data, sex_test, embarked_test], axis=1)

# since we have semi-acceptance data as we have null values present.
# Here we will clear and refine data
# will drop cabin feature from train and test data as it contains many null values.
train_data.drop(['Cabin','Name', 'PassengerId', 'Sex', 'Embarked', 'Ticket'], axis=1, inplace=True)
test_data.drop(['Cabin','Name', 'PassengerId', 'Sex', 'Embarked', 'Ticket'], axis=1, inplace=True)

# will fill null values of feature by calculating average age of people resides in a Passenger class.
def mean_age(col):
	Age, Pclass= col[0], col[1]
	if pd.isnull(Age):
		if Pclass == 1:
			return cal_age(1)
		elif Pclass == 2:
			return cal_age(2)			
		else :
			return cal_age(3)
	else:
		return Age

def cal_age(num):
	return int(train_data[train_data['Pclass'] == num]['Age'].mean())

train_data['Age'] = train_data[['Age', 'Pclass']].apply(mean_age, axis=1)
test_data['Age'] = test_data[['Age', 'Pclass']].apply(mean_age, axis=1)

# testing and training a model
logmodel = LogisticRegression()
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
logmodel.fit(X_train,y_train)

# prediction
prediction = logmodel.predict(X_test)

# evaluation
# classification report
print(classification_report(y_test, prediction))

# confusion matrix
print(confusion_matrix(y_test, prediction))
plt.show()
