#!/usr/bin/env python
# coding: utf-8

# # Setup
# First, we must setup our environment. We import various libraries (which you can view in the code cell below) and setup our directories according to the result we get from walking the filenames in our supplied files.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # splitting our data into training and testing data
import seaborn as sns # for creating a correlation heatmap
import matplotlib.pyplot as plt # for displaying our heatmap for analysis
from xgboost import XGBClassifier # eventually, we will use an XGBClassifier for our model
from sklearn.metrics import accuracy_score # to score our model

# Input data files are available in the "../input/" directory.

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Reading our Data
# Here, we read our data from the supplied .csv file using Pandas and representing it as a Pandas dataframe.
# 
# We are interested in predicting a diagnosis based on cell features, so we assign a 'y' variable to the diagnosis column.
# 
# It is worth noting that when using the 'id' feature as an index col, we get a column full of NaN entries. We remove this column as it provides no use to us. We also change some Pandas options. This is intended so that whenever we call 'head', we can see all the features and column names without truncation.
# 
# We then replace the diagnoses with 1 for malignant (original represented as an 'M') and 0 for benign (originally represented as a 'B'). This is useful for when we fit our model.

# In[ ]:


# Read the dataset
X_full = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv', index_col='id')

# Assign y to the diagnosis column
y = X_full.diagnosis

# Assigning our index_col to be the column 'id' shifted our data over, leaving a column with all NaN entries.
# We drop that here
X = X_full.drop(columns=['Unnamed: 32'])

# Show all values whenever we call head.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# If we run .dtypes on our data frame, we notice that all columns, aside from the diagnosis being a string, our integers.

# We replace a malignant diagnosis with 1, and benign with 0
X['diagnosis'].replace('M', 1, inplace=True)
X['diagnosis'].replace('B', 0, inplace=True)
y.replace('M', 1, inplace=True)
y.replace('B', 0, inplace=True)


# # Data Analysis
# To avoide overfitting, we find the features which seem to have a low impact on the diagnosis. We do this by using a heatmap correlation chart.
# 
# The figure we get will display the correlation one attribute has on another. We are interested in which attributes do (or don't) affect the diagnosis column. We analyze the results on the figure, and ignore the features which have less than an absolute value of 0.5.

# In[ ]:


# Here, we use the seaborn correlation heatmap to visualize the correlatons of features in our dataset on one another.
# Using the filter method, we will drop features which have an absolute value of less than 0.5 on the feature 'diagnosis'

# Setting up and displaying our heatmap correlation
plt.figure(figsize=(20,20))
cor = X.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds, fmt='.2f')
plt.show()


# # Applying our Analysis
# Now, we run some code to do exactly what we said we would do above: ignore the features which have a low impact on the diagnosis column.
# 
# We also split our data into training and testing data to both train and fit our model.

# In[ ]:


# Keep features which have a med-high correlation on the diagnosis
features = ['radius_mean', 'perimeter_mean', 'area_mean', 'compactness_mean', 'concavity_mean', 
            'concave points_mean', 'radius_se', 'perimeter_se', 'area_se', 'radius_worst', 'perimeter_worst',
           'area_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst']
X = X[features]

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)


# # Creating and Testing a Model
# Now we are ready to create, train, and test a model. We must use a classifying model, as the predictions are to either be a 0 (for benign) and 1 (for malignant). We assess the accuracy of this model using SKLearn's "accuracy_score" function.

# In[ ]:


# We will use an XGBoostClassifier, and score the model using SKLearn Accuracy Score

model = XGBClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_valid)
accuracy_score(y_valid, preds)

