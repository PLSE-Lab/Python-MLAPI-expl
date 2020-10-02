#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# # Loading in the dataset

# In[ ]:


df_train = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")
df_test = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")


# In[ ]:


# Checking for missing value on the training dataset
df_train.isna().sum()


# In[ ]:


# Checking for missing value on the test dataset
df_test.isna().sum()


# * We happen to have a lot of missing data for both training and testing.
# * For the sake of simplicity I will just replace them with NA.
# * Later, however, we will need to impute the missing variables.

# In[ ]:


df_train.info()


# In[ ]:


#  Since we have both integers/floats and strings I will fill integers with np.nan and strings with "NA"
object
object_type = np.dtype("O")
df_train = df_train.apply(lambda x: x.fillna("NA") if x.dtype == object_type
                                       else x.fillna(-1))
df_test = df_test.apply(lambda x: x.fillna("NA") if x.dtype == object_type
                                       else x.fillna(-1))


# # Encoding Features
# 
# Now that we have dealt with missing values I will move on to encode the features

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


# For simplicity sake I will encode all categorical variables the same way but this is not the way we will do it for our actual submission
categorical_columns = df_train.select_dtypes("object")

for col in categorical_columns:
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(df_train[col])
    df_train[col] = encoded
    
# Same for the test
    
categorical_columns = df_test.select_dtypes("object")

for col in categorical_columns:
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(df_test[col])
    df_test[col] = encoded


# In[ ]:


df_train.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df_train.drop(["id","target"],axis=1)
y = df_train.target
X_train, X_test, y_train, y_test = train_test_split(X,y) # Splitting the data to train and test to see how we are doing


# In[ ]:


from sklearn.ensemble import RandomForestClassifier # Importing our model of choice


# In[ ]:


# Notice I'm not using cross validation here but it will be neccesary when we want to make a good submission
rf = RandomForestClassifier(n_jobs=-1)
rf.fit(X_train,y_train)


# In[ ]:


preds = rf.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(preds,y_test)) # We are not doing well here as we lack precision on target class 1


# In[ ]:


preds = rf.predict(df_test.drop("id",axis=1))


# # Making a submission
# 
# We want ids from y_test and the predictions but to make sure lets check the sample submission

# In[ ]:


sample = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')
sample.head()


# In[ ]:


submission = pd.DataFrame({
    "id" : df_test.id,
    "target" : preds
})
submission.head()


# We can then submit the prediction after committing the kernel by selecting it from the output files.

# In[ ]:


submission.to_csv("submission.csv",index=False)


# In[ ]:




