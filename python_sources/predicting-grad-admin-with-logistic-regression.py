#!/usr/bin/env python
# coding: utf-8

# # This kernel is designed to predict the chance of a candidate being accepted into a graduate program using a Logistic Regression model. This process is broken down into two parts:
# # 1. Part I consists of an exploratory data analysis and
# # 2. Part II implements machine learning using logistic regression.

# # Import the required libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# # Part I - Exploratory Data Analysis

# # Preprocessing the data

# In[ ]:


# Read the data into a pandas dataframe
df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
# Store a copy of the original dataframe
df_with_predictions = df.copy()


# In[ ]:


# Eyeball the data
print(df.info())
df.head()


# In[ ]:


# Remove whitespaces in the headers
df.rename(columns=lambda x: x.strip(), inplace=True) 

# Drop the column labelled 'Serial No.'. It does not affect our analysis because it is a nominal value.
df.drop(columns={'Serial No.'}, inplace=True)
df.head()


# ## Analysing the data

# In[ ]:


# Display a description of the dataframe
df.describe()


# ## Eyeball the correlation between all columns using sns.pairplot and sns.heatmap

# In[ ]:


# Eyeball the correlation between all columns using sns.pairplot
sns.pairplot(df, kind='scatter', hue='University Rating', palette='husl')


# In[ ]:


# Eyeball the correlation between all columns using sns.heatmap
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=1, fmt='.2f',
            cmap="viridis", vmin=0, vmax=1)
plt.show()


# ### Correlation values of all the columns with Chance of Admit - The features which are strongly correlated are the  GRE Score, TOEFL score, University Rating, SOP and CGPA.

# In[ ]:


df.corr()['Chance of Admit'].round(1)


# ### Based on the correlation of all the features on the heatmap with Chance of Admit, drop those features that have a low correlation; any correlation with a value < 70%.

# In[ ]:


df.drop(columns={'LOR', 'Research'}, inplace=True)
df.head()


# ### Create checkpoint - The final preprocessed dataframe contains only those features that have a correlation of atleast 70% with the chance of admit column of the dataframe.

# In[ ]:


df_preprocessed = df.copy()
df_preprocessed.head(10)


# In[ ]:


# Display the shape of the dataframe
df_preprocessed.shape


# # Part II - Implementing Machine Learning

# ## Create the targets

# ### Targets are chosen based on the median values of Chance of Admit. The values above the median are assumed to have a high probability of being accepted into a graduate program.

# In[ ]:


df['Chance of Admit'].median()


# In[ ]:


# Create the targets
targets = np.where(df['Chance of Admit'] >= df['Chance of Admit'].median(), 1, 0)
targets.shape


# ### I have used the median to balance the dataset, thereby, making it implicitly stable and rigid. Rougly half the values are 0's and the other half are 1's. This will prevent our model from learning to output one of the values exclusively.

# In[ ]:


# Adding the column 'Probability of Acceptance' to our dataframe
df['Probability of Acceptance'] = targets
df.head()


# ### A balance of 45% - 55% is sufficient to perform regression.

# In[ ]:


targets.sum()/len(targets)


# In[ ]:


# create a checkpoint
# drop column 'Chance of Admit' to avoid multicollinearity
data_with_targets = df.drop(['Chance of Admit'], axis=1)
data_with_targets.head()


# ## Select the inputs for regression

# In[ ]:


# Display the shape of the dataframe
data_with_targets.shape


# In[ ]:


# Select the inputs
inputs = data_with_targets.iloc[:, :-1]
inputs.head()


# ## Split the data into train & test sets

# In[ ]:


# Splitting the data into train and test sets with a 80-20 split percentage
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=.2, shuffle=True, random_state=20)


# In[ ]:


# Displaying the train and test datasets and targets
print(x_train, y_train)
print(x_test, y_test)


# In[ ]:


# Displaying the shape of the train and test datasets and targets
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# ## Logistic Regression with sklearn

# In[ ]:


# Create a logistic regression object
log_reg = LogisticRegression()
# Fit the data to the model
log_reg.fit(x_train, y_train)


# In[ ]:


# Display the accuracy of the model
log_reg.score(x_train, y_train)


# ## Manually checking the accuracy

# In[ ]:


# Accuracy means that x% of the model outputs match the targets
model_outputs = log_reg.predict(x_train)
model_outputs


# In[ ]:


y_train


# In[ ]:


model_outputs == y_train


# In[ ]:


# Model accuracy
np.sum(model_outputs == y_train) / len(model_outputs)


# ## Finding the intercept and coefficients 

# In[ ]:


# Intercept value
log_reg.intercept_


# In[ ]:


# Coefficient value
log_reg.coef_


# ### Logistic regression: log odds = b0 + b1X1 + b2X2 + b3X3 + ... + bnXn

# ## Creating a summary table

# In[ ]:


feature_name = inputs.columns.values
summary_table = pd.DataFrame(columns=['Feature Name'], data=feature_name)
summary_table['Coefficient'] = np.transpose(log_reg.coef_)
summary_table.head()


# In[ ]:


# Adding the intercept value to the summary table
summary_table.index += 1
summary_table.loc[0] = ['Intercept', log_reg.intercept_[0]]
summary_table.sort_index(inplace=True)
summary_table.head()


# In[ ]:


# Finding the odds ratio
summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)
summary_table


# # Interpreting the coefficients and odds ratio
# 
# ### Since the inputs are unscaled, the Coefficient column and the Odds_ratio column of the summary table provide an accurate measure of the importance of each feature. Features that have a coefficient value of 0 and odds ratio of 1 are insginificant with regard to affecting the chance of being admitted to a graduate program. 
# ### Therefore, by this interpretation, one can observe that the two most significant features influencing the chance of being admitted are a candidate's CPGA and SOP, followed by university rating.

# # Testing the model

# In[ ]:


log_reg.score(x_test, y_test)


# In[ ]:


# Predict the probability of an output being 0 (first column) or 1 (second column)
predicted_proba = log_reg.predict_proba(x_test)
predicted_proba


# In[ ]:


# Shape of the test-data set
predicted_proba.shape


# In[ ]:


# Slice out the values from the second column
probability_admit = predicted_proba[:,1]
probability_admit


# In[ ]:


# Predicted values
pred = log_reg.predict(x_test)
pred


# In[ ]:


predicted_value = log_reg.predict(x_test)
predicted_value


# ## Create the Dataframe with predictions

# In[ ]:


df_with_predicted_outcomes = inputs.copy()

probability_admit = pd.DataFrame(probability_admit)
df_with_predicted_outcomes['Probability'] = probability_admit

predicted_value = pd.DataFrame(predicted_value)
df_with_predicted_outcomes['Prediction'] = predicted_value


# In[ ]:


# Display the final dataframe with predictions
df_with_predicted_outcomes.head(10)


# # The Dataframe named 'data_with_predicted_outcomes' shows the probability of a candidate being accepted into a graduate program. The accuracy of this model is 78%. Prospective students can use this model as a tool while shortlisting their colleges.

# In[ ]:




