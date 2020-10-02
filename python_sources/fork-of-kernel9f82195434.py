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


# Path of the file to read. We changed the directory structure to simplify submitting to a competition
train_path = '../input/titanic/train.csv'
titanic_data=pd.read_csv(train_path,index_col='PassengerId')


# In[ ]:


titanic_data


# In[ ]:


titanic_data.head()


# In[ ]:


titanic_data.columns


# In[ ]:


titanic_data.isna().sum()


# In[ ]:


titanic_data.Survived.mean()


# In[ ]:


titanic_data.drop(columns=['Name','Cabin','Ticket'],inplace=True)


# In[ ]:


titanic_data


# In[ ]:


age_mean=titanic_data.Age.mean()
titanic_data.fillna(age_mean,inplace=True)
titanic_data


# In[ ]:


titanic_data.isna().sum()


# In[ ]:


titanic_data.Embarked


# In[ ]:


titanic_data.Embarked.isna().sum()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
# Create target object and call it y
y=titanic_data.Survived

titanic_data.Sex.replace('male',0,inplace=True)
titanic_data.Sex.replace('female',1,inplace=True)
titanic_data.Embarked.replace('C',0,inplace=True)
titanic_data.Embarked.replace('S',1,inplace=True)
titanic_data.Embarked.replace('Q',2,inplace=True)

X=titanic_data.drop(columns=['Survived'])


# In[ ]:


titanic_data


# In[ ]:


X


# In[ ]:


X.head()


# In[ ]:


y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[ ]:


para = list(range(5, 30, 1))


# In[ ]:


results = {}
for i in para:
    dt = DecisionTreeClassifier(max_leaf_nodes=i, random_state=1)
    dt.fit(X_train, y_train)
    preds = dt.predict(X_test)
    acc = accuracy_score(y_true=y_test, y_pred=preds)
    f1 = f1_score(y_true=y_test, y_pred=preds)
    print(i)
    print(classification_report(y_true=y_test, y_pred=preds))
    print('-----------------------')
    results[i] = f1
    


# In[ ]:


results


# In[ ]:


max(results, key=results.get)


# In[ ]:


results[max(results, key=results.get)]


# In[ ]:


best_para = max(results, key=results.get)


# In[ ]:


final_model = DecisionTreeClassifier(max_leaf_nodes=best_para)
final_model.fit(X, y)


# In[ ]:


testpath = '../input/titanic/test.csv'
test_df = pd.read_csv(testpath, index_col='PassengerId')
test_df.drop(columns=['Name', 'Cabin', 'Ticket'], inplace=True)
test_df


# In[ ]:


test_df.isna().sum()


# In[ ]:


test_df.Age.fillna(age_mean, inplace=True)
test_df.Fare.fillna(titanic_data.Fare.mean(), inplace=True)


# In[ ]:


test_df


# In[ ]:


test_df.Sex.replace('male', 0, inplace=True)
test_df.Sex.replace('female', 1, inplace=True)
test_df.Embarked.replace('C', 0, inplace=True)
test_df.Embarked.replace('S', 1, inplace=True)
test_df.Embarked.replace('Q', 2, inplace=True)
test_df


# In[ ]:


preds = final_model.predict(test_df)


# In[ ]:


test_out = pd.DataFrame({
    'PassengerId': test_df.index, 
    'Survived': preds
})


# In[ ]:


test_out.to_csv('submission.csv', index=False)

