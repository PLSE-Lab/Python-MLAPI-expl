#!/usr/bin/env python
# coding: utf-8

# **Problem Statement**
# 
# * To determine which classification algorithm will work well on the dataset.
# * Which features are more useful in this task.

# Here **class** is our **target variable**. We will use various classifiaction techniques to determine which one performs well.

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


# In[ ]:


# Importing libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# In[ ]:


# Importing machine learning algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


# In[ ]:


# Importing dataset
mushroom = pd.read_csv("../input/mushroom-classification/mushrooms.csv")


# In[ ]:


# Getting a look at the data.
mushroom.head()


# Looks like all categorical data.Let's see:

# In[ ]:


mushroom.info()


# Seems like no missing values, but is it true? We will see this later.

# In[ ]:


mushroom.shape


# Let's see how many unique labels, are in each column.

# In[ ]:


for col in mushroom.columns:
    print('{}:{}'.format(col, mushroom[col].unique()))


# After close inspection of above output, we can see that there is missing value in the column 'stalk-root'. Which is denoted by '?'.

# Is it an imbalance class problem?

# In[ ]:


mushroom.groupby(by='class').agg({'class':'count'}).plot(kind='bar')


# So, it is almost a balanced class classification problem.

# **Data Preprocessing**
# 
# As it is a classification problem. Therefore, I will use **Label Encoding**. Because here classification algorithm will not give importance to one class over other and this will not increase number of columns.

# **Imputation of missing values**
# 
# As the data is categorical, therefore, I will use modal value of the column to impute missing values. But before that let's divide our data into train and test set.

# In[ ]:


# Separating the target from the data
y = mushroom['class']
X = mushroom.iloc[:, 1:]

# Splitting in training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=132)


# In[ ]:


# Imputing missing value
imputer = SimpleImputer(missing_values='?', strategy='most_frequent')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# In[ ]:


# Transforming X_train and X_test in dataframes
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# Adding column in train and test sets
X_train['label'] = 1
X_test['label'] = 0

# Concatnating both datasets
data = pd.concat([X_train, X_test], axis=0)
 
# Applying LabelEncoder on the dataset
encoder = LabelEncoder()
encoded_data = data.iloc[:,:22].apply(encoder.fit_transform)
encoded_data = pd.DataFrame(encoded_data, columns=X.columns)
encoded_data['label'] = data['label']
encoder1 = LabelEncoder()
y_train = encoder1.fit_transform(y_train)
y_test = encoder1.transform(y_test)


# In[ ]:


# Separating train and test set
X_train = encoded_data[encoded_data['label']==1]
X_test = encoded_data[encoded_data['label']==0]
X_train.drop('label', axis=1, inplace=True)
X_test.drop('label', axis=1, inplace=True)


# Now, we are ready to apply different machine learning algorithms. Let's begin the fun! 

# In[ ]:


tree = DecisionTreeClassifier(random_state=1)
tree.fit(X_train, y_train)
prediction = tree.predict(X_test)
print('Accuracy of the model is : {}'.format(accuracy_score(prediction, y_test)))


# In[ ]:


forest = RandomForestClassifier(random_state=1)
forest.fit(X_train, y_train)
prediction = forest.predict(X_test)
print("Accuracy of the model is : {}".format(accuracy_score(prediction, y_test)))


# As both models are doing great in accuracy, let's find there different capabilities.

# In[ ]:


print('Precision of model : {}'.format(precision_score(prediction, y_test)))

