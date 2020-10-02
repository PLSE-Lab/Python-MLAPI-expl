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


df = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')
df


# In[ ]:


['Survived',len (df['Survived'].unique())]


# In[ ]:


target_column = 'Survived'
print(df[target_column].unique())
print ('----------------------------------')
print(df[target_column].value_counts())
print ('----------------------------------')
print ('  Data Types        ')
print(df.dtypes)
print ('----------------------------------')
print ('         counts of Missing values')
print (df.isna().sum())
print ('----------------------------------')
print ('         Numbers of unique values')
print ([[col,len (df[col].unique())] for col in df.columns])


# Drop some unimportant Columns

# In[ ]:


drop_columns = ['Name','Ticket','Cabin']
df.drop (drop_columns, axis=1, inplace=True)
df


# Imputaion the missing value for column 'Age' (Mean)

# In[ ]:


imputed_columns = ['Age']
imputed_df = df[imputed_columns]  # >> as a DF
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer() #strategy='most_frequent')
imputed_df1 = pd.DataFrame(my_imputer.fit_transform(imputed_df))
# Imputation removed column names; put them back
imputed_df1.columns = imputed_df.columns
df.update(imputed_df1)
print ('The Counts of Missing Data equal:      ',df[imputed_columns].isna().sum())


# Imputaion the missing value for column 'Embarked' (strategy='most_frequent')

# In[ ]:


imputed_columns = ['Embarked']
imputed_df = df[imputed_columns]  # >> as a DF
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(strategy='most_frequent')
imputed_df1 = pd.DataFrame(my_imputer.fit_transform(imputed_df))
# Imputation removed column names; put them back
imputed_df1.columns = imputed_df.columns
df.update(imputed_df1)
print ('The Counts of Missing Data equal:      ',df[imputed_columns].isna().sum())


# In[ ]:


df.isna().sum()


# In[ ]:


print(df[target_column].value_counts())
print ('----------------------------------')
print ('         Data type        ')
print(df.dtypes)
print ('----------------------------------')
print ('         counts of Missing values')
print (df.isna().sum())


# # LableEncoder (Catigorical Variables)

# In[ ]:


lable_columns = ['Sex','Embarked']
label_df = df[lable_columns]
from sklearn.preprocessing import LabelEncoder
label_df = label_df.apply(LabelEncoder().fit_transform)
label_df


# ## Final Data Frame

# In[ ]:


df1 = df.drop (['Sex','Embarked'], axis=1)
final_df = pd.concat([df1,label_df], axis=1)
final_df


# In[ ]:


print ('         Data type        ')
print(final_df.dtypes)
print ('----------------------------------')
print ('         counts of Missing values')
print (final_df.isna().sum())


# Split the data

# In[ ]:


df = final_df
target_column = 'Survived'
y = df[target_column]
X = df.drop(columns=[target_column]) # FOR drop more than one column

# split the data

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)


# # Model 

# RandomForestClassifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
for n in range (2,11,2):
    forest_model = RandomForestClassifier(n_estimators=n,random_state=1)
    forest_model.fit(X_train, y_train)
    preds = forest_model.predict(X_valid)
    print(accuracy_score(y_valid, preds))


# XGBoost

# In[ ]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

for n in range (2,7,2):
    my_model = XGBClassifier(n_estimators=n, learning_rate=0.05, n_jobs=4)
    my_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)
    predictions = my_model.predict(X_valid)
    print(str(accuracy_score(predictions, y_valid)))


# Final Model
# 

# In[ ]:


final_model = XGBClassifier(n_estimators=2, learning_rate=0.05, n_jobs=4)
final_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)
preds = final_model.predict(X_valid)
print(accuracy_score(y_valid, preds))


# # Test Data

# In[ ]:


df_test = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')
df_test


# In[ ]:


print ('  Data Types        ')
print(df_test.dtypes)
print ('----------------------------------')
print ('         counts of Missing values')
print (df_test.isna().sum())
print ('----------------------------------')
print ('         Numbers of unique values')
print ([[col,len (df_test[col].unique())] for col in df_test.columns])


# Missing Values

# In[ ]:


print ('The Counts of Missing Data equal:\n\n',df_test.isna().sum())


# Drop some unimportant columns

# In[ ]:


drop_columns_test = ['Name','Ticket','Cabin']
df_test.drop (drop_columns_test, axis=1, inplace=True)
df_test


# Imputaion the missing value for column 'Age' , 'Fare'

# In[ ]:


df_test['Age'].fillna(value = df_test['Age'].mean(), inplace = True)
df_test['Fare'].fillna(value = df_test['Fare'].mean(), inplace = True)


# In[ ]:


df_test.isna().sum()


# LabelEncoder

# In[ ]:


lable_columns_test = ['Sex','Embarked']
label_df_test = df_test[lable_columns_test]
from sklearn.preprocessing import LabelEncoder
label_df_test = label_df_test.apply(LabelEncoder().fit_transform)
label_df_test


# In[ ]:


final_df_test = pd.concat([df_test.drop (lable_columns_test,axis=1),label_df_test],axis=1)
final_df_test


# In[ ]:


print ('         Data type        ')
print(final_df_test.dtypes)
print ('----------------------------------')
print ('         Counts of Missing values')
print (final_df_test.isna().sum())


# In[ ]:


preds_test = final_model.predict(final_df_test)
preds_test


# In[ ]:


# Save test predictions to file
output = pd.DataFrame({'PassengerId': final_df_test.index,
                       'Survived': preds_test})
output.to_csv('submission.csv', index=False)

