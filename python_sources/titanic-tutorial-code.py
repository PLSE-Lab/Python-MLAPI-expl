#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


dataset = pd.read_csv('/kaggle/input/titanic/train.csv')
testset = pd.read_csv("/kaggle/input/titanic/test.csv")

X_train = dataset.drop(['Survived','Name','Ticket','Cabin'],axis=1)
X_test = testset.drop(['Name','Ticket','Cabin'],axis=1)

y_train = dataset['Survived']


# In[ ]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(X_train[['Age']])
X_train['Age'] = imputer.transform(X_train[['Age']])

imputer.fit(X_test[['Age']])
X_test['Age'] = imputer.transform(X_test[['Age']])

imputer.fit(X_test[['Fare']])
X_test['Fare'] = imputer.transform(X_test[['Fare']])


imputer_text = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputer_text.fit(X_train[['Embarked']])
X_train['Embarked'] = imputer_text.transform(X_train[['Embarked']])

imputer_text.fit(X_test[['Embarked']])
X_test['Embarked'] = imputer_text.transform(X_test[['Embarked']])


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,2,7])], remainder='passthrough')
X_train = np.array(ct.fit_transform(X_train))
X_test = np.array(ct.fit_transform(X_test))


# In[ ]:


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


output = pd.DataFrame({'PassengerId': testset.PassengerId, 'Survived': y_pred})
output.to_csv('my_submission.csv', index=False)

