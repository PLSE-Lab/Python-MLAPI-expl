#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


test.shape


# In[ ]:


train.shape


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.columns.values


# In[ ]:


test.columns.values


# In[ ]:


train.describe()


# # Exploring Missing Values

# In[ ]:


train.BloodPressure.plot()


# In[ ]:


#drop outliers
train3 = train[(train[['BloodPressure']] != 0).all(axis=1)]
train3.BloodPressure.plot()
print(train3.shape)


# In[ ]:


train.shape


# In[ ]:


train.BMI.plot()


# In[ ]:


#drop outliers
from sklearn.preprocessing import Imputer
import numpy

imputer = Imputer()
train2 =train[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',]].replace(0, numpy.NaN)
train2["Outcome"]= train["Outcome"]
transformed_train = imputer.fit_transform(train2)
print(numpy.isnan(transformed_train).sum())
train2.BMI.plot()
print(train2.shape)


# In[ ]:


train2.describe()


# In[ ]:


#checking missing values
train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train['Outcome'].hist(bins = 20)
plt.show()


# In[ ]:


train.Outcome.unique()


# In[ ]:


train.Outcome.value_counts()


# **Grid Search**

# # predicting

# In[ ]:


X = train2.drop(['Outcome'], axis = 1)
from sklearn.preprocessing import Imputer
imputer = Imputer()
transformed_X = imputer.fit_transform(X)
X_with_null = train.drop(['Outcome'], axis = 1)
y = train2.Outcome
y_with_null = train.Outcome


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


y.head()


# In[ ]:


X.head()


# In[ ]:


train3.shape


# # Cross-Validation Score

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

X_validate = X[:200]
Y_validate = y[:200]
params = {'learning_rate':0.1}
model = XGBClassifier(n_estimators=500, **params)
kfold = KFold(n_splits=10, random_state=7)
print("dropped nulls")
print(cross_val_score(model, transformed_X, y,cv=kfold))


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

X_validate_null = X_with_null[:200]
Y_validate_null = y_with_null[:200]
params = {'learning_rate':0.1}
model = XGBClassifier(n_estimators=500, **params)
kfold = KFold(n_splits=10, random_state=7)
print("Raw data")
print(cross_val_score(model, X_validate_null, Y_validate_null,cv=kfold))


# In[ ]:


XX = pd.DataFrame(data=transformed_X)
XX.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
XX['Id'] = train['Id']
XX.head()


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
params = {'learning_rate':0.01}
#clf = GradientBoostingClassifier(n_estimators=290, **params)
clf = XGBClassifier(n_estimators=100, **params)
clf.fit(XX,y)


# In[ ]:


predicted = clf.predict(test)


# In[ ]:


print(predicted)
#from sklearn.metrics import accuracy_score
#accuracy_score(X[['Outcome']], predicted)


# In[ ]:


test.shape


# In[ ]:


predicted.shape


# In[ ]:


output = pd.DataFrame(predicted,columns = ['Outcome'])
test = pd.read_csv('../input/test.csv')
output['Id'] = test['Id']
output[['Id','Outcome']].to_csv('johnnybgood2.csv', index = False)
output.head()


# In[ ]:


get_ipython().system('ls -l *.csv')

