#!/usr/bin/env python
# coding: utf-8

# # Scikit Learn Random Forest Classifier for "Learn With Other Kaggle Users" competition

# In this notebook I preform simple data exploration and feature engineering, followed by training and evaluation of the RandomForestClassifier. Finally, I retrain the model on full training dataset and create a submission.
# 
# The model is trained with all default options to generate a performence baseline.

# ## Import modules

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


# ## Read Training Data

# In[ ]:


X_full = pd.read_csv('../input/learn-together/train.csv', index_col='Id')


# ## Explore the data

# In[ ]:


X_full.head()


# In[ ]:


X_full.describe()


# In[ ]:


f, ax = plt.subplots(figsize=(25, 25))
sns.heatmap(X_full.corr())


# Looking closer at columns 'Soil_Type7' and 'Soil_Type15'

# In[ ]:


X_full[['Soil_Type7', 'Soil_Type15']].describe()


# Tese two columns contain only '0's for all training samples, therefore will provide no value for this model.

# ## Colum to predict

# In[ ]:


y = X_full.Cover_Type


# > ## Features

# 1. Removing the column to predict 'Cover_Type' from the list of all columns
# 2. Remove 2 the columns 'Soil_Type7' and 'Soil_Type15' which have only one value in all training samples.

# In[ ]:


features = list(X_full.columns)
features.remove('Cover_Type')
features.remove('Soil_Type7')
features.remove('Soil_Type15')


# In[ ]:


features


# ## Split Training set

# In[ ]:


# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X_full[features], y, train_size=0.8, test_size=0.2,
                                                      random_state=0)


# ## Train & Evaluate

# In[ ]:


model = RandomForestClassifier(random_state=0)


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


# Accuracy on training set
accuracy_score(y_train, model.predict(X_train))


# *Note: high accuracy on training data set indicates overfitting.*

# In[ ]:


# Accuracy on validation set
accuracy_score(y_valid, model.predict(X_valid))


# ## Re-Train model on full dataset

# In[ ]:


model.fit(X_full[features], y)


# In[ ]:


# Accuracy on full set
accuracy_score(y, model.predict(X_full[features]))


# ## Create Submission

# In[ ]:


# Read test data
X_test_full = pd.read_csv('../input/learn-together/test.csv', index_col='Id')


# In[ ]:


X_test_full.head()


# In[ ]:


preds_test = model.predict(X_test_full[features])


# In[ ]:


output = pd.DataFrame({'Id': X_test_full.index,
                       'Cover_Type': preds_test})
output.to_csv('submission.csv', index=False)

