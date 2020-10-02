# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

'''import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))'''

def to_int_gender_col(val):
    '''return 1 for male 0 for female'''
    if val == 'male':
        return 1
    else:
        return 0
    
def to_int_embarked_col(val):
    '''return 1 for S 
                2 for C 
                3 for Q 
                0 for nan'''
    
    if val == 'S':
        return 1
    elif val == 'C':
        return 2
    elif val == 'Q':
        return 3
    else :
        return 0
    

def preprocess(df):
    df = df.drop(columns=['PassengerId','Name','Ticket','Cabin','Fare'])
    cols = df.columns
    df['Sex'] = df['Sex'].apply(to_int_gender_col)
    df['Embarked'] = df['Embarked'].apply(to_int_embarked_col)
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df))
    df.columns = cols
    return df

X = pd.read_csv('/kaggle/input/titanic/train.csv')
Y = X['Survived']
X = preprocess(X)
X = X.drop(columns='Survived')



'''
Y_pred = logReg(X_test)
print(Y_pred)
'''
#X.head()
logRegMod = LogisticRegression().fit(X,Y)



dTree = DecisionTreeClassifier()
dTreeMod = dTree.fit(X,Y)


X_test = pd.read_csv('/kaggle/input/titanic/test.csv')
ids = X_test['PassengerId']
X_test = preprocess(X_test)
y_pred = dTree.predict(X_test)

ans = pd.DataFrame(ids,columns=['PassengerId'])
ans['Survived'] = y_pred



#print('model built')
# Any results you write to the current directory are saved as output.
ans.to_csv('svm.csv',index=False)