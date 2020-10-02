#!/usr/bin/env python
# coding: utf-8

# Imports

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import os
print(os.listdir("../input"))


# Read CSV file.

# In[ ]:


data = pd.read_csv('../input/oasis_longitudinal.csv')
data = data.fillna(method='ffill')


# Show data head.

# In[ ]:


data.head(5)


# Print shape of the data.

# In[ ]:


print('data shape: ', data.shape)


# Print out the names of the columsn in the data and then print out what they stand for.

# In[ ]:


print('DATA COLUMNS: \n', [col for col in data])

print('\nM/F: Gender',
     '\nAge: Age',
     '\nEDUC: Education Level',
     '\nSES: Socioeconomic Status',
     '\nMMSE: Mini Mental State Exam',
     '\nCDR: Clinical Dementia Rating',
     '\neTIV: Estimated Total Intracranial Volume',
     '\nnWBV: Normalize Whole Brain Volume',
     '\nASF: Atlas Scaling Factor')


# Extract features we want to use, then use one-hot-encoding to replace categorical features with numbers.

# In[ ]:


features = ["M/F","Age","EDUC","SES","MMSE","eTIV","nWBV","ASF"]
x_data = pd.get_dummies(data[features])
y_data = data['CDR']


# Create data to use for plots.

# In[ ]:


plot_data = data[features + ['CDR']]


# Show head.

# In[ ]:


plot_data.head(5)


# Show a correlation between features with a heatmap.

# In[ ]:


plt.figure(figsize=(10,8))

sns.heatmap(plot_data.corr(), annot=True, linewidth=.2)


# Split data into training and testing sets for both x and y.

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=.2, random_state=0)


# Print shapes of new training and testing sets.

# In[ ]:


print('X_TRAIN SHAPE: ', x_train.shape)
print('X_TEST SHAPE: ', x_test.shape)
print('Y_TRAIN SHAPE: ', y_train.shape)
print('Y_TEST SHAPE: ', y_test.shape)


# Create our model. We will use the ExtraTreesRegressor from scikit-learn. We will then print out the mean absolute error and the score of the model.

# In[ ]:


etr = ExtraTreesRegressor(random_state=4, n_estimators=100, max_features='sqrt')

etr.fit(x_train, y_train)

pred = etr.predict(x_test)

print('Mean Absolute Error: ', round(mean_absolute_error(y_test, pred), 5))
print('Score: ', round(etr.score(x_test, y_test), 2) * 100, '%')


# Thank you!

# In[ ]:




