# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import time
import sys
import os 

DeprecationWarning('ignore')
warnings.filterwarnings('ignore',message="don't have warning")

df=pd.read_csv('/kaggle/input/titanic/train.csv')

train, test = train_test_split(df,test_size=0.2,random_state = 12)

def fill_age(df):
    mean =  29.67
    df['Age'].fillna(mean, inplace = True)

#%%
train.isnull().sum()

def fill_age(df):
    mean =  29.67
    df['Age'].fillna(mean, inplace = True)
    return df

def fill_embarked(df):
    df.Embarked.fillna('S',inplace = True)
    return(df)
def label_encode(df):
    from sklearn.preprocessing import LabelEncoder 
    label = LabelEncoder()
    df['Sex'] =  label.fit_transform(df['Sex'])
    df['Embarked'] = label.fit_transform(df['Embarked'])
    return df
def encode_feature(df):
    df = fill_age(df)
    df = fill_embarked(df)
    df = label_encode(df)
    return(df)

train = encode_feature(train)
test = encode_feature(test)

def x_and_y(df):
    x = df.drop(["Survived","PassengerId","Cabin","Name","Ticket"],axis=1)
    y = df["Survived"]
    return x,y
x_train,y_train = x_and_y(train)
x_test,y_test = x_and_y(test)

log_model = LogisticRegression()
log_model.fit(x_train,y_train)
prediction = log_model.predict(x_train)
score = accuracy_score(y_train,prediction)
print(score)