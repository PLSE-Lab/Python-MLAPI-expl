#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#print(check_output(["pwd"]).decode("utf8"))


# In[ ]:


train_df = pd.read_csv('../input/train.csv')


# In[ ]:


train_df.info()


# In[ ]:


model_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
model_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin_class', 'Cabin_num', 'Embarked']


# In[ ]:


#df = pd.DataFrame([{"a": 1, "b": 7}, {"a": 2,"b": 8}, {'a': 3, "b": 9}, {'a': 4, "b": 10}])
#df.insert(1,'c',0)
#df


# In[ ]:


def get_cabin_class(val):
    if val is not np.nan:
        return val.strip()[0]
    return val

def get_cabin_number(val):
    if val is not np.nan:
        #print(val.split())
        vs = val.split()
        for x in vs:
            v = x[1:].strip()
            if v != '':
                return int(v)
        return np.nan
    return val
    
#tdf = train_df['Cabin'][70:80].apply(get_cabin_number)
#for i in tdf:
#    print(i)


# In[ ]:


train_df['Cabin_class']=train_df['Cabin'].apply(get_cabin_class)
train_df['Cabin_num']=train_df['Cabin'].apply(get_cabin_number)


# In[ ]:


train_df


# In[ ]:


train_x = pd.get_dummies(train_df[model_cols], columns=['Pclass', 'Sex', 'Cabin_class', 'Embarked'])
train_y = train_df['Survived']


# In[ ]:


train_x['Age'].fillna(train_x['Age'].median(), inplace=True)
train_x['Fare'].fillna(train_x['Fare'].median(), inplace=True)
train_x['Cabin_num'].fillna(0, inplace=True)
#test_x['Age'].fillna(train_x['Age'].median(), inplace=True)


# In[ ]:


# Scale the features
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()

train_x[['Age','Fare','SibSp', 'Parch']] = scaler.fit_transform(train_x[['Age','Fare','SibSp', 'Parch']])


# In[ ]:


train_x.columns


# In[ ]:


from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss='log')
clf = clf.fit(train_x, train_y)


# In[ ]:


test_df = pd.read_csv('../input/test.csv')
test_df['Cabin_class']=test_df['Cabin'].apply(get_cabin_class)
test_df['Cabin_num']=test_df['Cabin'].apply(get_cabin_number)
test_x = pd.get_dummies(test_df[model_cols], columns=['Pclass', 'Sex', 'Cabin_class', 'Embarked'])
test_x['Age'].fillna(train_x['Age'].median(), inplace=True)
test_x['Fare'].fillna(train_x['Fare'].median(), inplace=True)
test_x['Cabin_num'].fillna(0, inplace=True)

#test_x[['Age','Fare','SibSp', 'Parch']] = scaler.transform(test_x[['Age','Fare','SibSp', 'Parch']])

#test_y = test_df['Survived']


# In[ ]:


#Select training columns from test df

train_cols = train_x.columns
test_cols_all = test_x.columns
test_cols = [x for x in train_cols if x in test_cols_all] #maintain train column order
test_xx = test_x[test_cols]

lc = len(train_x.columns)
j = 0
for i in range(lc):
    if train_cols[i] == test_cols[j]:
        j += 1
        continue
    else:
        test_xx.insert(i, train_cols[i], 0)


# In[ ]:


#test_cols


# In[ ]:


def cmp(a, b):
    return (a > b) - (a < b) 

cmp(test_xx.columns, train_x.columns)


# In[ ]:


test_pred = clf.predict(test_xx)


# In[ ]:


test_pred


# In[ ]:


feat_coef = list(zip(train_x.columns, clf.coef_[0]))
feat_coef.sort(key=lambda x: -x[1])


# In[ ]:


feat_coef


# In[ ]:


#train_df[['Sex', 'Survived']].groupby(['Sex', 'Survived']).size()


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": test_pred
    })

submission.to_csv('./submission.csv', index=False)


# In[ ]:


#print(check_output(["ls"]).decode("utf8"))


# In[ ]:




