#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


dataset = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')


# # Preprocessing

# In[ ]:


dataset.shape


# In[ ]:


dataset.count()


# In[ ]:


dataset.isna().sum()


# In[ ]:


dataset.dropna(inplace=True)


# In[ ]:


dataset.isna().sum()


# In[ ]:


dataset.describe()


# In[ ]:


dataset['Dependents'].value_counts()


# In[ ]:


dataset['Dependents'] = dataset['Dependents'].map({'3+': 3, '1':1, '2':2, '0':0})
dataset['Dependents'].value_counts()


# In[ ]:


dataset.head()


# # Encoders

# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
le = LabelEncoder()
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [10])], remainder='passthrough')


# # Label Encoding

# In[ ]:


dataset['Gender'] = le.fit_transform(dataset['Gender'])
dataset['Married'] = le.fit_transform(dataset['Married'])
dataset['Education'] = le.fit_transform(dataset['Education'])
dataset['Self_Employed'] = le.fit_transform(dataset['Self_Employed'])
dataset['Loan_Status'] = le.fit_transform(dataset['Loan_Status'])


# In[ ]:


dataset.head()


# In[ ]:


X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# # Splitting into train and test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# # Scalling

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 5:9] = sc.fit_transform(X_train[:, 5:9])
X_test[:, 5:9] = sc.transform(X_test[:, 5:9])


# In[ ]:


X_train


# # Applying onehot encoder on categorical data

# In[ ]:


X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)


# In[ ]:


X_train


# # install catboost

# In[ ]:


get_ipython().system('pip install catboost')


# # Training

# In[ ]:


from catboost import CatBoostClassifier
classifier = CatBoostClassifier()
classifier.fit(X_train, y_train)


# # Prediction

# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# Let's check accuracy using cross validation

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# Ok. Here we got Accuracy: 80.19 % and Standard Deviation: 6.00 % in cross validation method
