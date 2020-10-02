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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train_features = pd.read_csv('../input/train_features.csv')
df_train_labels = pd.read_csv("../input/train_labels.csv")
df_test_features = pd.read_csv("../input/test_features.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")


# **Inspectt the files**

# In[ ]:


df_train_labels.columns


# In[ ]:


df_train_features.shape


# In[ ]:


df_test_features.shape


# **Majority Class Baseline**
# 
# Using the mode of the target variable as the majority class baseline 
# 

# In[ ]:


import numpy as np
majority_class = df_train_labels['status_group'].mode()[0]
#print(majority_class)

y_pred = np.full(shape=df_train_labels['status_group'].shape, fill_value=majority_class)


# In[ ]:


df_train_labels.status_group.shape, y_pred.shape


# In[ ]:


all(y_pred==majority_class)


# In[ ]:


from sklearn.metrics import accuracy_score 
accuracy_score(df_train_labels['status_group'], y_pred)


# **Class imbalance**
# 
# It's import to check how the classes are represented. 

# In[ ]:


df_train_labels['status_group'].value_counts()


# In[ ]:


df_train_labels['status_group'].value_counts(normalize=True)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions


#let's import the warning before running any sophisticated methods
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


#  **classification report**

# In[ ]:


print(classification_report(df_train_labels['status_group'], y_pred) )


# Looking at the above report, it's a pretty bad prediction and recall. 

# **Logistic Regression only with Numeric Values**

# **Merge Training and Test datasets**
# 
# Using pandas concat function excluding 'axis=1' in the arguements. I'll notedown the last row of my train dataset to make the splitting easier (we want to retain the same train/test shape and observations as originally given). The last row of our df_train_features dataset is 59394.

# In[ ]:


#Let's merge train and test
full_df = pd.concat([df_train_features, df_test_features])


# In[ ]:


full_df.shape


# In[ ]:


#these were the number of rows in the orignal sets
59400 + 14358


# **a Bit of Cleaning**
# 
# before we run logistic regression, let's do little bit of cleaning of the data

# In[ ]:


full_df.head()


# In[ ]:


full_df.isna().sum()


# i'll take care of the misisng values in other columns now

# In[ ]:


#data['Native Country'] = data['Native Country'].fillna(data['Native Country'].mode()[0])
full_df['funder'] = full_df['funder'].fillna(full_df['funder'].mode()[0])


# In[ ]:


full_df['installer'] = full_df['installer'].fillna(full_df['installer'].mode()[0])


# In[ ]:


full_df['subvillage'] = full_df['subvillage'].fillna(full_df['subvillage'].mode()[0])


# In[ ]:


full_df['public_meeting'] = full_df['public_meeting'].fillna(full_df['public_meeting'].mode()[0])


# In[ ]:


full_df['scheme_management'] = full_df['scheme_management'].fillna(full_df['scheme_management'].mode()[0])


# In[ ]:


full_df['permit'] = full_df['permit'].fillna(full_df['permit'].mode()[0])


# In[ ]:


full_df.isna().sum()


# In[ ]:


full_df = full_df.drop(columns = 'scheme_name')


# **Spliting the dataset into orignal shape**
# 

# In[ ]:


#split the data back
X_cleaned = full_df[:-14358]
X_test_cleaned = full_df[-14358:]
y = df_train_labels['status_group']


# In[ ]:


X_cleaned.shape, X_test_cleaned.shape, y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y, test_size=0.25, random_state=42, shuffle=True)


# In[ ]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# **Logistic Regression with only Numeric Features**
# 
# The score isn't going to be super exciting but let's give a try before going into more complicated/sophisticated models

# In[ ]:


X_train.dtypes


# For now, we aren't touching X_test_cleaned dataframe. That's our actual test data. We are only using our training that we split furtherinto training and testing sets

# In[ ]:


X_train_numeric = X_train.select_dtypes(np.number)


# In[ ]:


X_test_numeric = X_test.select_dtypes(np.number)


# In[ ]:


#let's see how our model does here

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_numeric, y_train)
y_pred = model.predict(X_test_numeric)
accuracy_score(y_test, y_pred)


# As expected, we do not see a great improvement over our baseline model. 

# **Encoding Catergorical Features using dummyEncoder from sklearn to perform Logistic R**
# 
# Since our numeric only is not a significat improvment over the baseline model, let's one hot encode all catergorical without much cleaning. This encoding isn't a great choice for this kind of dataset (with lots of categories of categorical features so in some sense, it's a baseline encoding) as it encodes the catergories in ordianl fashion. 

# In[ ]:


from sklearn.preprocessing import LabelEncoder
def dummyEncode(df):
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        le = LabelEncoder()
        for feature in columnsToEncode:
            try:
                df[feature] = le.fit_transform(df[feature])
            except:
                print('Error encoding '+feature)
        return df


# In[ ]:


#encode our train df that we split from the full_df
cat_coded_df = dummyEncode(X_cleaned)


# In[ ]:


cat_coded_df.head()


# In[ ]:


#let's also encode out test set we split from full_df
X_cleaned_test = dummyEncode(X_test_cleaned)


# In[ ]:


X_cleaned_test.head()


# In[ ]:


#split our train set that we just encoded (and assigned into cat_coded_df)
#into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cat_coded_df, y, test_size=0.25, random_state=42, shuffle=True)


# In[ ]:


import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


# In[ ]:


#run multinomial logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='newton-cg', multi_class='multinomial')
model.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# **Logistic Regression with encoded categorical features and using StandardScaler**

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(StandardScaler(), 
                        LogisticRegression(solver='newton-cg', multi_class='multinomial'))

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# StandardScaler doesn't seem to have any impact on the scores. Infact, the score is a little worse now. 
