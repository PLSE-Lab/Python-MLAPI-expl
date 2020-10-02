# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import feature_selection
#from sklearn.cross_validation import cross_val_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
titanic_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')

#print(titanic_train.info())
#titanic_train.head()
#titanic_train.shape

Y_train = titanic_train['Survived']
X_train = titanic_train.drop(['PassengerId', 'Name', 'Survived'], axis = 1)
X_train['Age'].fillna(X_train['Age'].mean(), inplace = True)
X_train.fillna('unknown', inplace = True)

X_test = titanic_test.drop(['PassengerId', 'Name'], axis = 1)
passangerId = np.array(titanic_test['PassengerId'])
X_test['Age'].fillna(X_test['Age'].mean(), inplace = True)
X_test.fillna('unknown', inplace = True)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.25, random_state = 33)
print(X_train.shape)

vec = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_val = vec.transform(X_val.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

#X_train.toarray()

dt = DecisionTreeClassifier(criterion = 'entropy')
dt.fit(X_train, Y_train)
#dt.score(X_val, Y_val)
Y_predict = dt.predict(X_test)
predictions = pd.DataFrame({'PassengerId': passangerId, 'Survived': Y_predict})
predictions.to_csv('submission.csv', index=False)

print('xxx')

fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile = 7)
X_train_fs = fs.fit_transform(X_train, Y_train)
dt.fit(X_train_fs, Y_train)
X_val_fs = fs.transform(X_val)
print(dt.score(X_val_fs, Y_val))

X_test_fs = fs.transform(X_test)
Y_predict_fs = dt.predict(X_test_fs)
#Y_predict_fs
predictions_fs = pd.DataFrame({'PassengerId': passangerId, 'Survived': Y_predict_fs})
predictions_fs.to_csv('submission_fs.csv', index=False)
