#!/usr/bin/env python
# coding: utf-8

# # Dependencies
# 
# First import few dependencies
# 1. `os` to access the file system
# 2. `pandas` to prepare date
# 3. `sklearn.model_selection` to split the data into training and testing
# 4. `sklearn.naive_bayes` our bayesian classifier 

# In[ ]:


import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# Get file location and store it

# In[ ]:


os.listdir('../input')


# In[ ]:


data_file = os.path.join('..','input', 'FIFA 2018 Statistics.csv')


# Read the csv file into a pandas dataframe

# In[ ]:


dataframe = pd.read_csv(data_file)


# In[ ]:


dataframe.describe()


# In[ ]:


dataframe.head()


# In[ ]:


dataframe.columns


# Show the correlation matrix of the data to see which columns are more related to our goal

# In[ ]:


dataframe.corr()


# Define a function to normalize the numeric data points

# In[ ]:


def normalize(column):
    mean = column.mean()
    std = column.std()
    return column.apply(lambda x: (x - mean) / std)


# In[ ]:


numeric_cols = ['Goal Scored', 'Ball Possession %', 'Attempts', 'On-Target', 'Off-Target', 'Blocked', 'Corners', 'Offsides',
       'Free Kicks', 'Saves', 'Pass Accuracy %', 'Passes',
       'Distance Covered (Kms)', 'Fouls Committed', 'Yellow Card',
       'Yellow & Red', 'Red', '1st Goal',
       'Goals in PSO', 'Own goals', 'Own goal Time']


# Replace the missing values in numeric columns with NaN then fill that with zeros

# In[ ]:


for num_column in numeric_cols:
    dataframe[num_column] = pd.to_numeric(dataframe[num_column], errors='coerce')
dataframe.fillna(0, inplace=True)
dataframe.describe()


# Normalize each column to have a mean of ~zero and standard deviation of 1

# In[ ]:


for num_column in numeric_cols:
    dataframe[num_column] = normalize(dataframe[num_column])
dataframe.describe()


# In[ ]:


categorical_cols = ['Man of the Match']


# Change the categorical result into a 1 or 0 result

# In[ ]:


for categorical_column in categorical_cols:
    dataframe[categorical_column] = dataframe[categorical_column].apply(lambda x: 1 if x == 'Yes' else 0)


# Split the data with most columns

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(dataframe[numeric_cols], dataframe['Man of the Match'],
                                                    test_size=0.33, random_state=42)


# Define a Gaussian NB model and train it

# In[ ]:


model = GaussianNB()


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


model.score(X_test, y_test)


# To enhance the score, let's go back and see which columns in the correlation matrix are more promising

# In[ ]:


new_model = GaussianNB()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(dataframe[['Goal Scored', '1st Goal', 'On-Target', 'Attempts', 'Corners']], dataframe['Man of the Match'],
                                                    test_size=0.33, random_state=42)


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


model.score(X_test, y_test)


# And indeed the score is higher with less, and more related, columns

# This is just a 10 min try on this problem. I haven't used sklearn for a while so I'm sure that there are many better ways to process the data but I thought that I would start with this basic algorithm.

# In[ ]:




