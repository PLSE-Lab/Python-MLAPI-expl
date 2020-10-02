# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('../input/train.csv', header = 0, )
test  = pd.read_csv('../input/test.csv' , header = 0, )
Full_data =[train,test]
to_drop=['PassengerId', 'Ticket', 'Name', 'Cabin']
for Datatype in Full_data:
    Datatype =Datatype.drop(to_drop, axis = 1)
    Datatype['Sex'] = Datatype['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    Datatype['Embarked'] = Datatype['Embarked'].map( {'S': 0, 'C': 1, 'Q' : 2} ).astype(int)
    Datatype['Age'].fillna(Datatype['Age'].mean())
    Datatype[' Age_category']= pd.qcut(Datatype['Age'],5)
    Datatype['Fare_category']= pd.qcut(Datatype['Fare'],5)
    
    Datatype['Pclass']= Datatype['Pclass'].astype(int)
    Datatype['Parch']= Datatype['Pclass'].astype(int)
    Datatype['Sibsp']= Datatype['Pclass'].astype(int)
    
    Datatype=Datatype.drop(['Age','Fare'],axis = 1)
    
    
labels = train['Survived'].astype(int).values
train=train.drop('Survived', axis = 1)
train=train.values
test =test.values

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.