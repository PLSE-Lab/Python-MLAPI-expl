#!/usr/bin/env python
# coding: utf-8

# # **TITANIC MACHINE LEARNING FROM DISASTERS MODEL**

# # Importing the libraries

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


# # Importing the training dataset

# In[ ]:


dataset_train = pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


dataset_train.head()


# # Creating a new type of variable from parch and sibsp

# In[ ]:


dataset_train['family_size'] = dataset_train['SibSp'] + dataset_train['Parch'] + 1 
dataset_train.head()


# In[ ]:


dataset_train[['family_size', 'Survived']].groupby(['family_size'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# # Creating another new type from family_size

# In[ ]:


dataset_train['alone'] = 0
dataset_train.loc[dataset_train['family_size'] == 1, 'alone'] = 1
dataset_train.head()


# In[ ]:


dataset_train[['alone', 'Survived']].groupby(['alone'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


X_train = dataset_train.iloc[:, [2,4,5,9,11,13]].values
y_train = dataset_train.iloc[:, 1].values


# # Checking Missing values in our training dataset
# 

# In[ ]:


print(dataset_train.isnull().sum())


# # Inserting new Values at the place of missing data in training set

# In[ ]:


# For Age
imputer_1 = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_1.fit(X_train[:, [2]])
X_train[:, [2]] = imputer_1.transform(X_train[:, [2]])

# For Embarked
imputer_2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer_2.fit(X_train[:, [4]])
X_train[:, [4]] = imputer_2.transform(X_train[:, [4]])


# # Encoding categorical data in training set

# In[ ]:


# Encoding P Class
ct_1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X_train = np.array(ct_1.fit_transform(X_train))
X_train = X_train[: ,1:]

# Encoding Embarked
ct_2 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')
X_train = np.array(ct_2.fit_transform(X_train))
X_train = X_train[: ,[0,1,3,4,5,6,7,8]]

# Encoding Gender
le_train = LabelEncoder()
X_train[:, 4] = le_train.fit_transform(X_train[:, 4])


# # Now Doing the above whole preprocessing on our test dataset

# # Importing the test dataset

# In[ ]:


dataset_test= pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


dataset_test.head()


# In[ ]:


dataset_test['family_size'] = dataset_test['SibSp'] + dataset_test['Parch'] + 1 
dataset_test.head()


# In[ ]:


dataset_test['alone'] = 0
dataset_test.loc[dataset_train['family_size'] == 1, 'alone'] = 1
dataset_test.head()


# In[ ]:


X_test = dataset_test.iloc[:, [1,3,4,8,10,12]].values


# # Checking Missing values in our test dataset

# In[ ]:


print(dataset_test.isnull().sum())


# # Inserting new Values at the place of missing data in test set

# In[ ]:


# For Age
imputer_3 = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_3.fit(X_test[:, [2]])
X_test[:, [2]] = imputer_3.transform(X_test[:, [2]])

# For Fare
imputer_4 = SimpleImputer(missing_values=np.nan, strategy='median')
imputer_4.fit(X_test[:, [3]])
X_test[:, [3]] = imputer_4.transform(X_test[:, [3]])


# # Encoding categorical data in test dataset

# In[ ]:


# Encoding P Class
ct_3 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X_test = np.array(ct_3.fit_transform(X_test))
X_test = X_test[: ,1:]

# Encoding Embarked
ct_4 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')
X_test = np.array(ct_4.fit_transform(X_test))
X_test = X_test[: ,[0,1,3,4,5,6,7,8]]

# Encoding Gender
le_test = LabelEncoder()
X_test[:, 4] = le_test.fit_transform(X_test[:, 4])


# # Splitting the dataset into the Training set and Test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_1, X_2, y_1, y_2 = train_test_split(X_train, y_train, test_size = 0.20)


# # Applying Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_1[:, [5,6]] = sc_X.fit_transform(X_1[:, [5,6]])
X_2[:, [5,6]] = sc_X.transform(X_2[:, [5,6]])
X_test[:, [5,6]] = sc_X.transform(X_test[:, [5,6]])


#  # Now traing our Machine learning model on training dataset and fitting it over test data set to predict survival
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier.fit(X_1,y_1)


# # Prediction for Training Set

# In[ ]:


y_pred_train = classifier.predict(X_2)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

print('Confusion Matrix :')
print(confusion_matrix(y_2, y_pred_train)) 
print('Accuracy Score :',accuracy_score(y_2, y_pred_train))
print('Report : ')
print(classification_report(y_2, y_pred_train))


# # Prediction for test Set

# In[ ]:


y_pred_test = classifier.predict(X_test)

output = pd.DataFrame({'PassengerId': dataset_test.PassengerId, 'Survived': y_pred_test})
output.to_csv('my_submission_4.csv', index=False)
print("Your submission was successfully saved!")

