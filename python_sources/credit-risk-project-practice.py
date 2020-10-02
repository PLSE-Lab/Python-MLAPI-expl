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


# # Goal
# In this project I would like to explore the dataset of credit risk to find its insight. I will use the simple data manipulating and visualization skills in order to work on it. Then, decision tree algorithm will be implemented to see if I can predict whether a credit risk is good or bad based on the applicant's information.

# # First glimpse of the data

# In[ ]:


df = pd.read_csv('../input/german-credit-data-with-risk/german_credit_data.csv')

print('The dataset consists of {} entries and {} features'.format(df.shape[0], df.shape[1]))
df.head()


# In[ ]:


df.info()


# We can see here the dataset consists of numerous categorical features, which we will need to transform them so that ML model can work properly later.

# In[ ]:


df.drop(['Unnamed: 0'], axis=1, inplace=True)


# # Fill Null Value

# In[ ]:


df['Saving accounts'].fillna('none', inplace=True)
df['Checking account'].fillna('none', inplace=True)

df.isnull().sum()


# There are not many null values we need to fill here. Only little needed to be performed.

# # Visualization

# In[ ]:


import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(20, 15))

plt.subplot(3,2,1)
sns.countplot(df['Sex']);

plt.subplot(3,2,2)
sns.distplot(df['Age'], kde=False);

plt.subplot(3,2,3)
sns.countplot(df['Job']);

plt.subplot(3,2,4)
sns.countplot(df['Housing'], order=df['Housing'].value_counts().index);

plt.subplot(3,2,5)
sns.countplot(df['Saving accounts'], order=df['Saving accounts'].value_counts().index);

plt.subplot(3,2,6)
sns.countplot(df['Checking account'], order=df['Checking account'].value_counts().index);


# The charts are here to show the count and ditribution of the features.

# In[ ]:


sns.distplot(df['Credit amount'], kde=False);
plt.show()

sns.countplot(df['Purpose'], order=df['Purpose'].value_counts().index);
plt.xticks(rotation=45)
plt.show()

sns.countplot(df['Risk']);
plt.show()


# # Feature Engineering

# I decided to use LabelEncoder on Sex and Risk features because they are only binary categorical data, whereas if the number of categorical values is more than 2, one hot encoding or get dummies is preferred (for instance, the Purpose feature). However, although Checking and Saving account features are not binary, I did not apply get_dummies on them because their values are in hierarchy. I encoded them in order instead. 

# In[ ]:


from sklearn.preprocessing import LabelEncoder

new_df = df.copy()
# binary encoding
features_toencode = ['Sex', 'Risk']
for f in features_toencode:
    le = LabelEncoder()
    new_df[f] = le.fit_transform(df[f])

# encoder in order
new_df['Checking account'] = df['Checking account'].map({'none': 0, 'little': 1, 'moderate': 2, 'rich': 3})
new_df['Saving accounts'] = df['Saving accounts'].map({'none': 0, 'little': 1, 'moderate': 2, 'rich': 3, 'quite rich': 4})

# encode in dummies
new_df = pd.get_dummies(new_df)

new_df.head()


# # Classification

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = new_df.drop(['Risk'], axis=1)
y = new_df['Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

dt = DecisionTreeClassifier().fit(X_train, y_train)
train_pred = dt.predict(X_train)
test_pred = dt.predict(X_test)

print('Train set score:', accuracy_score(y_train, train_pred))
print('Test set score:', accuracy_score(y_test, test_pred))


# Apparently without tuning the hyperparameters, the decision tree model shows significant overfitting.

# In[ ]:


from sklearn.model_selection import GridSearchCV

params = {'max_depth': range(1,10,2), 'min_samples_split' : range(2,200,10)}

model = GridSearchCV(DecisionTreeClassifier(random_state=123), params, n_jobs=3, cv=3)
model.fit(X_train, y_train)

model.best_estimator_


# In[ ]:


print(model.score(X_train, y_train))
print(model.score(X_test, y_test))


# After several tries, the test set score is more or less around 0.71 approximately. I think it is because we only have 1000 entries in this dataset, which are far from enough for ML model to be in its full capacity. 
# 
# If you have any thought on how to improve the model or this whole project itself, feel free to comment below. As a beginner, I'm eager to improve my data science skill!
