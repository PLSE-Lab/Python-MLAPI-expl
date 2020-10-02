#!/usr/bin/env python
# coding: utf-8

# ## About
# This notebook contains a very fast fundamental Support Vector Machine (SVM) example in Python.
# 
# This work is part of a series called [Machine learning in minutes - very fast fundamental examples in Python](https://www.kaggle.com/jamiemorales/machine-learning-in-minutes-very-fast-examples).
# 
# The approach is designed to help grasp the applied machine learning lifecycle in minutes. It is not an alternative to actually taking the time to learn. What it aims to do is help someone get started fast and gain intuitive understanding of the typical steps early on

# ## Step 0: Understand the problem
# What we're trying to do here is to classify whether a mushroom is a poisoned mushroom.

# ## Step 1: Set-up and understand data
# This step helps uncover issues that we will want to address in the next step and take into account when building and evaluating our model. We also want to find interesting relationships or patterns that we can possibly leverage in solving the problem we specified.

# In[ ]:


# Set-up libraries
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report


# In[ ]:


# Check input data source
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Read-in data
df = pd.read_csv('../input/mushroom-classification/mushrooms.csv')


# In[ ]:


# Look at some details
df.info()


# In[ ]:


# Look at some records
df.head()


# In[ ]:


# Check for missing values
df.isna().sum()


# In[ ]:


# Check for duplicate values
df.duplicated().sum()


# In[ ]:


# Look at breakdown of label
sns.countplot(df['class'])
df['class'].value_counts()


# In[ ]:


# Summarise
df.describe()


# ## Step 2: Preprocess data and understand some more
# This step typically takes the most time in the cycle but for our purposes, most of the datasets chosen in this series are clean.
# 
# Real-world datasets are noisy and incomplete. The choices we make in this step to address data issues can impact downstream steps and the result itself. For example, it can be tricky to address missing data when we don't know why it's missing. Is it missing completely at random or not? It can also be tricky to address outliers if we do not understand the domain and problem context enough.

# In[ ]:


# Grab some samples
df = df.sample(5000, random_state=0)


# In[ ]:


# Transform categorical feature(s) to numeric
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    
df.describe()


# In[ ]:


# Explore correlation to label
df.corr()['class'].sort_values(ascending=False)


# In[ ]:


# Explore correlations visually
f, ax = plt.subplots(figsize=(24,6))
sns.heatmap(df.corr(), annot=True, fmt='.2f')


# In[ ]:


# Split dataset into 80% train and 20% validation
X = df.drop('class', axis=1)
y = df['class']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)


# ## Step 3: Model and evaluate
# This last step is three-fold.
# 
# We create the model and fit the model to the data we prepared for training.
# 
# We then proceed to classifying with the data we prepared for validation.
# 
# Lastly, we evaluate the model's performance with mainstream classification metrics.

# In[ ]:


# Build model and train
model = SVC()
model.fit(X_train, y_train)


# In[ ]:


# Apply model to validation data
y_predict = model.predict(X_val)


# In[ ]:


# Compare actual and predicted values
actual_vs_predict = pd.DataFrame({'Actual': y_val,
                                 'Predict': y_predict
                                 })

actual_vs_predict.sample(12)


# In[ ]:


# Evaluate model
print('Classification metrics: \n', classification_report(y_val, y_predict))


# ## Learn more
# If you found this example interesting, you may also want to check out:
# 
# * [Machine learning in minutes - very fast fundamental examples in Python](https://www.kaggle.com/jamiemorales/machine-learning-in-minutes-very-fast-examples)
# * [List of machine learning methods & datasets](https://www.kaggle.com/jamiemorales/list-of-machine-learning-methods-datasets)
# 
# Thanks for reading. Don't forget to upvote.

# In[ ]:




