#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train_df = pd.read_csv('../input/titanic/train.csv', index_col = 'PassengerId')
train_df


# In[ ]:


for col in train_df.columns:
    print(col, len(train_df[col].unique()))


# In[ ]:


train_df.drop(columns=['Name', 'Cabin', 'Ticket'], inplace=True)
train_df


# In[ ]:


age_mean = train_df.Age.mean()
train_df.Age.fillna(age_mean, inplace=True)
train_df.isna().sum()


# In[ ]:


target_col = 'Survived'
print(train_df[target_col].unique())
print(train_df[target_col].value_counts())
print(train_df.isna().sum())


# In[ ]:


from sklearn.preprocessing import LabelEncoder


sex_le =  LabelEncoder() 
train_df['Sex'] = sex_le.fit_transform(train_df['Sex'].astype(str))

embarked_le =  LabelEncoder() 
train_df['Embarked'] = embarked_le.fit_transform(train_df['Embarked'].astype(str))


# In[ ]:


y = train_df[target_col]
X = train_df.drop(columns=[target_col])


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[ ]:


para = list(range(100, 1001, 100))
print(para)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
results = {}
for n in para:
    print('para=', n)
    model = RandomForestClassifier(n_estimators=n)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accu = accuracy_score(y_true=y_test, y_pred=preds)
    f1 = f1_score(y_true=y_test, y_pred=preds, average='micro')
    print(classification_report(y_true=y_test, y_pred=preds))
    print('--------------------------')
    results[n] = f1


# In[ ]:


best_para = max(results, key=results.get)
print('best para', best_para)
print('value', results[best_para])


# In[ ]:


test_df = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')
test_df


# In[ ]:


test_df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
test_df


# In[ ]:


test_df['Sex'] = sex_le.transform(test_df['Sex'].astype(str))
test_df['Embarked'] = embarked_le.transform(test_df['Embarked'].astype(str))


# In[ ]:


final_model = RandomForestClassifier(n_estimators=best_para)
final_model.fit(X, y)


# In[ ]:


test_df.isna().sum()
test_df.Age.fillna(int(age_mean), inplace=True)
test_df.Fare.fillna(int(train_df.Fare.mean()), inplace=True)
train_df


# In[ ]:


predictions = final_model.predict(test_df)
predictions[:5]


# In[ ]:


predictions


# In[ ]:


sub_df = pd.DataFrame(data={
    'PassengerId': test_df.index,
    'Survived': predictions
})


# In[ ]:


sub_df.to_csv('submission.csv', index=False)

