# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import csv
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print(train.head())

# convert character string values to numeric values
train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 0)

# missing values
train['Age'] = train['Age'].fillna(np.mean(train['Age']))
train['Fare'] = train['Fare'].fillna(np.mean(train['Fare']))

train = train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
pId = test['PassengerId']

X = train.drop('Survived', axis = 1)
y = train['Survived']
print(len(X))
print(len(y))
xdf = pd.DataFrame(data=X)
ydf = pd.DataFrame(data=y)
#print(xdf.iloc[[2]])

#splitting dataset
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
kf = KFold(n_splits=8)
kf.get_n_splits(X)
sel_acc = 0;
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = xdf.iloc[train_index], xdf.iloc[test_index]
    y_train, y_test = ydf.iloc[train_index], ydf.iloc[test_index]
    # instantiate a logistic regression model, and fit with X and y
    model = LogisticRegression()
    model = model.fit(X_train, y_train)
    
    # check the accuracy on the training set
    print('train accuracy = ')
    print(model.score(X_train, y_train))
    
    predicted = model.predict(X_test);
    print('validation accuracy = ')
    print (accuracy_score(y_test, predicted))
    if(accuracy_score(y_test, predicted) > sel_acc):
        sel_acc = accuracy_score(y_test, predicted)
        model_sel = model
            
print('==========================================\n')





#actual_prediction

# convert character string values to numeric values
test['Sex'] = test['Sex'].apply(lambda x: 1 if x == 'male' else 0)
# missing values
test['Age'] = test['Age'].fillna(np.mean(test['Age']))
test['Fare'] = test['Fare'].fillna(np.mean(test['Fare']))

test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

data_predict = model_sel.predict(test)
print(len(data_predict))


rows = ['']*(len(pId)+1)
out = open("output_1.csv","w")
out.write('PassengerId,Survived\n')

for num in range(0, len(pId)):
    rows[num] = "%d,%d\n"%(pId[num],data_predict[num])

out.writelines(rows)
out.close()