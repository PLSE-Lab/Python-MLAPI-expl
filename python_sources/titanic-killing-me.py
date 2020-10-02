# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.naive_bayes import GaussianNB
# Sample Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
raw = pd.read_csv('../input/train.csv',index_col= 'PassengerId')
train = raw[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]
#or
#raw.drop(['Name','Ticket'],inplace=1)
#train
#train['Survived'].value_counts().plot(kind='bar')
#train[train['Sex'] == "female"] 
#female = train['Sex'] == "female"
#embarked_southampton = train['Embarked'] == "S"
#train[female & embarked_southampton][:10]
#train[female & embarked_southampton][['Pclass']][:10]
#counts and grouping
#train[female & embarked_southampton]['Pclass'].value_counts()
#females who died by point of embarkation
#died = train['Survived'] == 0
#died_female = train[female & died]
#compare female deaths by point of embarking
#(died_female['Embarked'].value_counts()/train['Embarked'].value_counts()).plot(kind = 'bar')
#(train[died]['Sex'].value_counts()/train['Sex'].value_counts()).plot(kind='bar') 
#(train[died]['Pclass'].value_counts()/train['Pclass'].value_counts()).plot(kind='bar') 
#(train[died]['Age']).plot()
#(train[died]['Cabin'].value_counts()/train['Cabin'].value_counts()).plot(kind='bar')
# Any results you write to the current directory are saved as output.
#survived_counts = train.groupby(['Survived','Sex']).aggregate(['sum','count','mean']).plot(kind='bar',figsize=(15, 6))
#survived_counts.plot(kind='bar')
#na_values = ['NO CLUE', 'N/A', '0']
#requests = pd.read_csv('../data/311-service-requests.csv',na_values=na_values, dtype={'Incident Zip': str})
#describe all numerical values
#train.describe()
#Replace missing by mean
#train['Age']=train['Age'].fillna(train['Age'].mean()) 
##or
#cleaning numerical variables
train['Age'].fillna(train['Age'].mean(),inplace=True)
#train.describe()
#describe categorical columns
#train[train.columns[train.dtypes == 'object']].describe()

#cleaning categorical variables
categorical = train.columns[train.dtypes == 'object']
for column in categorical:
    train[column].fillna('Missing',inplace = True)
    
#use one-hot-encoding to code categorical into numeric variables which can be converted into numpy matrix and used with ski-kit learn.
train_encoded = pd.get_dummies(train)
train_encoded.describe()
#train[train.columns[train.dtypes == 'object']].describe()

#train.columns[train.dtypes == 'float']
label = train_encoded['Survived'].values
train_encoded.drop('Survived', axis=1,inplace=True)
train_array = train_encoded.values
#train_array.shape
classifier = GaussianNB()
classifier.fit(train_array,label)
print(classifier.score(train_array,label))
# make predictions
expected = label
predicted = classifier.predict(train_array)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
#without normalization - raw values
#print(metrics.accuracy_score(expected,predicted, normalize=False))
print(metrics.accuracy_score(expected,predicted))

#Decision Tree
# fit a CART model to the data
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn import linear_model
model = linear_model.LogisticRegression(C=1e5)
#model = svm.SVC()
#model = RandomForestClassifier(n_estimators=10)
#model = AdaBoostClassifier(n_estimators=100)
model.fit(train_array, label)
print(model)
# make predictions
expected = label
predicted = model.predict(train_array)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
#without normalization - raw values
#print(metrics.accuracy_score(expected,predicted, normalize=False))
print(metrics.accuracy_score(expected,predicted))


raw_test = pd.read_csv('../input/test.csv',index_col= 'PassengerId')
test = raw_test[['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]
test['Age'].fillna(test['Age'].mean(),inplace=True)
test['Fare'].fillna(test['Fare'].mean(),inplace=True)
test_categorical = test.columns[test.dtypes == 'object']
for column in test_categorical:
    test[column].fillna('Missing',inplace = True)
test_encoded = pd.get_dummies(test)
test[test.columns[test.dtypes == 'object']].describe()
test.describe()

#important since all categorical variable values may not be present in both the tables. One hot encoding can cause problems.
final_train, final_test = train_encoded.align(test_encoded,join='left', axis=1)

final_test.fillna(0,inplace = True)
test_array = final_test.values
model.fit(final_train.values, label)
test['Survived'] = model.predict(test_array)
#test
test['Survived'].to_csv('submission.csv',header=True, index_label='PassengerId')