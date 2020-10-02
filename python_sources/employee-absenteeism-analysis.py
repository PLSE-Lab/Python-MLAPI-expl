#!/usr/bin/env python
# coding: utf-8

# ### Importing the relevant libraries

# In[ ]:


import pandas as pd
import numpy as np

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Load the CSV data

# In[ ]:


raw_csv_data = pd.read_csv('/kaggle/input/employee-absenteeism-prediction/Absenteeism-data.csv')
raw_csv_data.head()


# ### Data Preprocessing

# Copying the content of initial dataframe to a new one

# In[ ]:


df = raw_csv_data.copy()
df.head()


# Following code is for displaying all the rows and columns in the dataframe, despite of how large they are

# In[ ]:


pd.options.display.max_columns=None
pd.options.display.max_rows=None


# In[ ]:


display(df)


# Getting the information about the dataframe

# In[ ]:


df.info()


# Drop ID column from the dataframe as it is of no use for prediction

# In[ ]:


df = df.drop(['ID'], axis=1)


# Displaying the head of the dataframe

# In[ ]:


df.head()


# Exploration of Reason for Absence column

# In[ ]:


# Maximum value in 'Reason for Absence' column
df['Reason for Absence'].max()


# In[ ]:


# Minimum value in 'Reason for Absence' column
df['Reason for Absence'].min()


# In[ ]:


# Unique values in 'Reason for Absence' column
df['Reason for Absence'].unique()


# In[ ]:


# length of unique values in 'Reason for Absence' column
len(df['Reason for Absence'].unique())


# However wait didn't we already find out that minimum value contained in this column is zero while the
# largest one is 28. This makes up to 29 different values while using the len function instead we just obtained twenty eight.This has to say one thing and one thing only a number between 0 and 28 is missing.

# In[ ]:


sorted(df['Reason for Absence'].unique())


# You can spot that the value we lack is number 20.

# But wait which are the twenty eight reasons we have substituted with numbers. In other words reason one stands for a certain reason for absence as much as reason to stands for another. You may know from statistics that these variables are categorical nominal nominal because instead of using the numbers from 0 to 28 we could have had names disease dentist pregnancy etc..However using numbers is the convention for working with categorical nominal data 

# Converting to dummies

# In[ ]:


# Create dummy for 'Reason for Absence' column
reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)
reason_columns


# In[ ]:


# Check whether any missing value is there or not
reason_columns['check'] = reason_columns.sum(axis=1)
reason_columns


# In[ ]:


reason_columns['check'].sum(axis=0)


# Which is the length of dataframe, so there is no missing values in 'Reason for Absence' column
# 
# And thus, the validity of reason columns has been checked and we are satisfied with its state.

# In[ ]:


# So, we'll be dropping the check column from reason_columns
reason_columns = reason_columns.drop(['check'], axis=1)
reason_columns


# Group the Reason for Absence column

# In[ ]:


df.columns.values


# In[ ]:


reason_columns.columns.values


# In[ ]:


# Drop 'Reason for Absence' column to avoid multi-collinearity
df = df.drop(['Reason for Absence'], axis=1)
df.head()


# In[ ]:


# Group the variables from 'Reason for Absence' column
reason_type_1 = reason_columns.iloc[:, 1:14].max(axis=1)
reason_type_2 = reason_columns.iloc[:, 15:17].max(axis=1)
reason_type_3 = reason_columns.iloc[:, 18:21].max(axis=1)
reason_type_4 = reason_columns.iloc[:, 22:].max(axis=1)


# Concatenate column values from reason_columns to df

# In[ ]:


df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)
df.head()


# Rename the above four concatenated columns

# In[ ]:


df.columns.values


# In[ ]:


column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']


# In[ ]:


df.columns = column_names


# In[ ]:


df.head()


# Reorder columns

# In[ ]:


column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours']


# In[ ]:


df = df[column_names_reordered]


# In[ ]:


df.head()


# Create a Checkpoint (Creating checkpoint refers to storing the current version of once code)
# 
# (Create a copy of the current state of your dataframe)

# In[ ]:


df_reason_mod = df.copy()
df_reason_mod.head()


# Exploration of 'Date' column

# In[ ]:


type(df_reason_mod['Date'][0])


# In[ ]:


# Converting the Date column to timestamp format
df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'], format='%d/%m/%Y')
df_reason_mod['Date'].head()


# Extract the Month value

# In[ ]:


df_reason_mod['Date'][0]


# In[ ]:


df_reason_mod['Date'][0].month


# In[ ]:


list_months = []
list_months


# In[ ]:


len(df_reason_mod)


# In[ ]:


df_reason_mod.loc[:, 'Date'][0].month


# In[ ]:


for i in range (len(df_reason_mod)):
    list_months.append(df_reason_mod.loc[:, 'Date'][i].month)


# In[ ]:


list_months


# In[ ]:


len(list_months)


# In[ ]:


df_reason_mod['Month Value'] = list_months
df_reason_mod.head(20)


# Extract the Day of the Week

# In[ ]:


df_reason_mod.loc[:, 'Date'][0].weekday()


# In[ ]:


list_days = []


# In[ ]:


def date_to_weekday(date_value):
    return (date_value.weekday())


# In[ ]:


df_reason_mod['Day of the Week'] = df_reason_mod['Date'].apply(date_to_weekday)
df_reason_mod.head(20)


# In[ ]:


# Dropping Date column from dataframe to avoid multicollinearity
df_reason_mod = df_reason_mod.drop(['Date'], axis=1)
df_reason_mod.head()


# Reorder the columns

# In[ ]:


df_reason_mod.columns.values


# In[ ]:


column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4',
                          'Month Value', 'Day of the Week',
                           'Transportation Expense', 'Distance to Work', 'Age',
                           'Daily Work Load Average', 'Body Mass Index', 'Education',
                           'Children', 'Pets', 'Absenteeism Time in Hours']


# In[ ]:


df_reason_mod = df_reason_mod[column_names_reordered]


# In[ ]:


df_reason_mod.head(20)


# Create a Checkpoint

# In[ ]:


df_reason_date_mod = df_reason_mod.copy()


# In[ ]:


df_reason_date_mod.head()


# Exploring other variables

# In[ ]:


type(df_reason_date_mod['Transportation Expense'][0])


# In[ ]:


type(df_reason_date_mod['Distance to Work'][0])


# In[ ]:


type(df_reason_date_mod['Age'][0])


# In[ ]:


type(df_reason_date_mod['Daily Work Load Average'][0])


# In[ ]:


type(df_reason_date_mod['Body Mass Index'][0])


# Exploring 'Education', 'Children', 'Pets' columns

# In[ ]:


df_reason_date_mod['Education'].unique()


# In[ ]:


df_reason_date_mod['Education'].value_counts()


# In[ ]:


df_reason_date_mod['Education'] = df_reason_date_mod['Education'].map({1:0, 2:1, 3:1, 4:1})


# In[ ]:


df_reason_date_mod['Education'].unique()


# In[ ]:


df_reason_date_mod['Education'].value_counts()


# Final Checkpoint

# In[ ]:


df_preprocessed = df_reason_date_mod.copy()
df_preprocessed.head()


# In[ ]:


# Saving the preprocessed CSV file
df_preprocessed.to_csv('Absenteeism_preprocessed.csv', index=False)


# #### Create a logistic regression model to predict Absenteeism

# Load the preprocessed data

# In[ ]:


data_preprocessed = pd.read_csv('Absenteeism_preprocessed.csv')


# In[ ]:


data_preprocessed.head()


# Create the targets

# In[ ]:


data_preprocessed['Absenteeism Time in Hours'].median()


# In[ ]:


targets = np.where(data_preprocessed['Absenteeism Time in Hours'] > 
                   data_preprocessed['Absenteeism Time in Hours'].median(),
                   1, 0)


# In[ ]:


targets


# In[ ]:


data_preprocessed['Excessive Absenteeism'] = targets


# In[ ]:


data_preprocessed.head()


# A comment on the targets

# In[ ]:


targets.sum() / targets.shape[0]


# In[ ]:


# Drop 'Absenteeism Time in Hours' column
data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours', 
                                            'Daily Work Load Average', 
                                            'Education', 
                                            'Reason_4', 
                                            'Distance to Work'], 
                                             axis=1)
data_with_targets.head()


# In[ ]:


# Checking whether the following two dataframes are same or different
data_with_targets is data_preprocessed


# Select the inputs for the regression

# In[ ]:


data_with_targets.shape


# In[ ]:


data_with_targets.iloc[:, :14]


# In[ ]:


data_with_targets.iloc[:, :-1]


# In[ ]:


unscaled_inputs = data_with_targets.iloc[:, :-1]


# Standardize the data

# In[ ]:


# from sklearn.preprocessing import StandardScaler
# absenteeism_scaler = StandardScaler()


# In[ ]:


# import the libraries needed to create the Custom Scaler
# note that all of them are a part of the sklearn package
# moreover, one of them is actually the StandardScaler module, 
# so you can imagine that the Custom Scaler is build on it

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# create the Custom Scaler class

class CustomScaler(BaseEstimator,TransformerMixin): 
    
    # init or what information we need to declare a CustomScaler object
    # and what is calculated/declared as we do
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        
        # scaler is nothing but a Standard Scaler object
        self.scaler = StandardScaler(copy,with_mean,with_std)
        # with some columns 'twist'
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    
    # the fit method, which, again based on StandardScale
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    # the transform method which does the actual scaling

    def transform(self, X, y=None, copy=None):
        
        # record the initial order of the columns
        init_col_order = X.columns
        
        # scale all features that you chose when creating the instance of the class
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        
        # declare a variable containing all information that was not scaled
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        
        # return a data frame which contains all scaled features and all 'not scaled' features
        # use the original order (that you recorded in the beginning)
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# In[ ]:


unscaled_inputs.columns.values


# In[ ]:


# columns_to_scale = ['Month Value','Day of the Week', 'Transportation Expense', 
#                     'Distance to Work','Age', 'Daily Work Load Average', 'Body Mass Index', 
#                     'Children', 'Pets']

columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Education']


# In[ ]:


columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]


# In[ ]:


absenteeism_scaler = CustomScaler(columns_to_scale)


# In[ ]:


absenteeism_scaler.fit(unscaled_inputs)


# In[ ]:


scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)


# In[ ]:


scaled_inputs


# In[ ]:


scaled_inputs.shape


# Split the data into train and test and shuffle

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# Split
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size=0.8, random_state=20)


# In[ ]:


print(x_train.shape, y_train.shape)


# In[ ]:


print(x_test.shape, y_test.shape)


# Logistic Regression with Sklearn

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# Training model

# In[ ]:


reg = LogisticRegression()


# In[ ]:


reg.fit(x_train, y_train)


# In[ ]:


reg.score(x_train, y_train)


# Manually check the accuracy

# In[ ]:


model_outputs = reg.predict(x_train)


# In[ ]:


model_outputs


# In[ ]:


targets


# In[ ]:


np.sum(model_outputs == y_train)


# In[ ]:


model_outputs.shape[0]


# In[ ]:


np.sum(model_outputs == y_train) / model_outputs.shape[0]


# #### Finding the intercept and coefficients

# In[ ]:


# Finding the intercept
reg.intercept_


# In[ ]:


# Finding the coefficient
reg.coef_


# In[ ]:


# Finding the column names in unscaled dataframe
unscaled_inputs.columns.values


# In[ ]:


feature_name = unscaled_inputs.columns.values


# In[ ]:


# Creating summary table to store different attributes and their corresponding values
summary_table = pd.DataFrame(columns=['Feature Name'], data=feature_name)
summary_table['Coefficients'] = np.transpose(reg.coef_)
summary_table


# In[ ]:


# Inserting the value of intercept in the summary table
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table


# #### Interpreting the coefficients

# In[ ]:


summary_table['Odds_Ratio'] = np.exp(summary_table['Coefficients'])
summary_table


# In[ ]:


summary_table.sort_values('Odds_Ratio', ascending=False)


# #### Testing the model

# In[ ]:


reg.score(x_test, y_test)


# In[ ]:


predicted_proba = reg.predict_proba(x_test)
predicted_proba


# In[ ]:


predicted_proba[:, 1]


# #### Save the model

# In[ ]:


import pickle


# In[ ]:


# pickle the model file
with open('model', 'wb') as file:
    pickle.dump(reg, file)


# In[ ]:


# pickle the scaler file
with open('scaler', 'wb') as file:
    pickle.dump(absenteeism_scaler, file)

