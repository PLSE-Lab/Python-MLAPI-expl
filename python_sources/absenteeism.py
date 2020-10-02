#!/usr/bin/env python
# coding: utf-8

# In this tutorial we will try predict whether an employee is absent more than median time (or acceptable) or not using absentee data. There is not much informtion about the dataset on Kaggle. You can familirize yourself with the data here: https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work <br>
# If you like this tutorial please upvote it and let me know if you have any questions. Thanks

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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


raw_data = pd.read_csv('../input/Absenteeism_at_work.csv')
raw_data.head()


# In[ ]:


#Below code it just to let Ipython to show all the columns and rows
pd.options.display.max_columns = None
pd.options.display.max_rows = None


# It is always good practice to make a copy of original dataset so that we can make changes to the copy. We will make multiple copy of the data along the way at each step we make changes to the dataset so that we can go back and forth

# In[ ]:


df = raw_data.copy()
df.info() # this will give us number of rows and data types at each column 
#There are no missing values in our table


# # Descriptive Statistics

# In[ ]:


df.describe()


# In[ ]:


# we already said there is no missing value another way to do this is to plot heatmap graph df.isnull()
sns.heatmap(df.isnull(),cbar=False, yticklabels=False, cmap='plasma')


# # Preprocessing

# In[ ]:


# Let's get rid of unnecessary columns
df.drop(['ID'], axis=1, inplace=True)


# In[ ]:


# Let's how many reasons for absence are there
# We already know from descriptive statistics that min is 0 and max is 28
df['Reason for absence'].unique()


# In[ ]:


sorted(df['Reason for absence'].unique())


# Reason 20 has never been used as an excuse

# We will turn 'Reason for absence' column into dummy variable becuase although you might think this columns is numeric it is actually categorical. Each number represent different type of excuse.
# Again check https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work  to get familiar with dataset
# Lastly, I am  going to drop first column here to avoid multicoliniearity. reason 0 is actually represent no reason is given

# In[ ]:


reasons = pd.get_dummies(df['Reason for absence'], drop_first=True)
reasons.head()


# Having 27 dummy is not desirable, instead we can group reason for absence columns. If you look at the data we can group reason for absence into 4 categories
# disease related 0-14, pregnancy related 15-17,external 18-21, visit 22-28. 
# Code below requires a little explanation. When we get dummies at each row just one column is equal to 1 the rest are zeros. So instead of assigningn reasons.iloc[:, 1:14] to reason_type1 we assign the maximum value at each row, this way if value at column between 1 and 14 is 1 that means that employeee was absent for disease related excuse, if all are zero we will know that it was not disease related so just add zero to reason_type1

# In[ ]:


reason_type1 = reasons.iloc[:, 0:14].max(axis=1)
reason_type2 = reasons.iloc[:, 15:17].max(axis=1)
reason_type3 = reasons.iloc[:, 18:21].max(axis=1)
reason_type4 = reasons.iloc[:, 22:28].max(axis=1)


# In[ ]:


reason_type1.head(10)


# Let's concatenate these new columns we created and drop reasons for absence column.

# In[ ]:


df = pd.concat([df, reason_type1, reason_type2, reason_type3, reason_type4], axis=1)
df.head()


# In[ ]:


# drop 'Reason for absence' column
df.drop('Reason for absence', axis=1, inplace=True)


# column names 0, 1, 2 and 3 doesn't look that good. Let's rename them and reorder the columns. We want to see the reasons at the beginnign of the table

# In[ ]:


df.columns.values


# In[ ]:


column_names = ['Month of absence', 'Day of the week',
       'Seasons', 'Transportation expense',
       'Distance from Residence to Work', 'Service time', 'Age',
       'Work load Average/day ', 'Hit target', 'Disciplinary failure',
       'Education', 'Body mass index', 'Absenteeism time in hours', 'reason_1',
       'reason_2', 'reason_3', 'reason_4']


# In[ ]:


df.columns = column_names
df.head()


# In[ ]:


df = df[['reason_1', 'reason_2','reason_3', 'reason_4', 'Month of absence', 'Day of the week',
       'Seasons', 'Transportation expense',
       'Distance from Residence to Work', 'Service time', 'Age',
       'Work load Average/day ', 'Hit target', 'Disciplinary failure',
       'Education', 'Body mass index', 'Absenteeism time in hours']]
df.head()


# # Create a CheckPoint
# At this point we have done a lot of preprocession. We would like to make a copy of the dataframe to reduce risk of losing important data at later stage. Gave a name that will give you some information about what you have done until this point

# In[ ]:


df_reason_modified = df.copy()


# # Education

# In[ ]:


df_reason_modified['Education'].unique()


# In[ ]:


df_reason_modified['Education'].value_counts()


# These values repressent high school, graduate, post-graduate and master-Phd. Majority of employees are high school graduate and around 100 of them have college or higher degree. We will map high school to 0 and rest to 1. Are we losing any information here? Probably we do, but I doubt that will make a big change on our model's prediction. 

# In[ ]:


df_reason_modified['Education'] = df_reason_modified['Education'].map({1:0, 2:1, 3:1, 4:1})
df_reason_modified['Education'].unique()


# Lastly, 'Transportation expense', 'Distance from Residence to Work', 'Service time', 'Age', 'Work load Average/day ', 'Disciplinary failure','Body mass index' are important indicator, therefore I am not going to leave them intact. However, 'Hit target' is closely related with 'Work load Average/day ' therefore I am going to drop it.

# In[ ]:


df_reason_modified.drop(['Hit target'], axis=1, inplace=True)


# In[ ]:


df_preprocessed = df_reason_modified.copy()
df_preprocessed.head(10)


# # Create Targets
# zero absence is kind of extreme, everyone once in a while can't make it to work. So we will take the median of the 'Absenteeism in hours'. 

# In[ ]:


median = df_preprocessed['Absenteeism time in hours'].median()
median


# If an observation has been absent more than 3 hours we will assign them to 1, otherwise to 0. Easy way to do that is to use numpy where function. <br>
# Note: This way we will have balanced dataset, roughly the half of target will be zero and other half will be 1. This will also prevent our model from learning to output only 0s or 1s. If you have used  mean you wouldn't get the same thing unless you were lucky.

# In[ ]:


targets = np.where(df_preprocessed['Absenteeism time in hours']>median, 1,0)


# In[ ]:


targets[:10]


# In[ ]:


# let's check the ratio
targets.sum()/targets.shape[0]


# Great our dataset is balanced. Around 46% of the targets are 1s and 54% are zeros. That will work for logistic regression. For logistic regression usually 40 to 60 will work as well but that is not true for other algorithm such as NN. A balance of 45-55 is almost sufficient for NN.

# In[ ]:


df_preprocessed['Excessive Absenteeism'] = targets
df_preprocessed.head()


# ## Checkpoint

# In[ ]:


# data_with_targets = df_preprocessed.drop(['Absenteeism time in hours', 'reason_4', 'Body mass index', 'Age', 'Day of the week', 'Distance from Residence to Work'], axis=1)
data_with_targets = df_preprocessed.drop(['Absenteeism time in hours'], axis=1)


# In[ ]:


data_with_targets.head()


# In[ ]:


unscaled_inputs = data_with_targets.iloc[:,:-1]
unscaled_inputs.head()


# # Standardize the data
# We will standardize the data by subtracting the mean and dividing by standard deviation columnwise. However, we do not want to scale dummy variables otherwise they will lose their meanings. Along with dummies we created we are not going to scale Education and Disciplinary failure either since they are just zeros and ones. Unfortunately we can't use standard scaler instead we will create a custom scaler.

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy, with_mean, with_std)
        self.columns = columns
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.std_ = np.std(X[self.columns])
        return self
    
    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]
        


# In[ ]:


unscaled_inputs.columns.values


# In[ ]:


columns_to_omit = ['reason_1', 'reason_2', 'reason_3', 'reason_4','Disciplinary failure', 'Education',]
columns_to_scale = [x for x in unscaled_inputs if x not in columns_to_omit]


# In[ ]:


absenteeism_scaler = CustomScaler(columns_to_scale)


# In[ ]:


absenteeism_scaler.fit(unscaled_inputs)


# In[ ]:


scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)


# In[ ]:


scaled_inputs.head()


# As you can see all the dummies are reamined untouched

# In[ ]:


scaled_inputs.shape


# # Split the data into train and test

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(scaled_inputs, targets, test_size=0.2, 
                                                   random_state=42)


# # Training Data

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[ ]:


lr_model = LogisticRegression(solver='lbfgs')
lr_model.fit(X_train, y_train)


# In[ ]:


lr_model.score(X_train, y_train)


# That is not bad!

# In[ ]:


# or we could calculate accuracy manually like this
outputs = lr_model.predict(X_train)


# In[ ]:


outputs


# In[ ]:


print('total corretly predicted', np.sum(outputs == y_train))
print('accuracy', np.sum(outputs == y_train)/ outputs.shape[0])


# We get exactly the same number.
# Next let's get the coefficients and intercept to interpret our model

# In[ ]:


summary_table = pd.DataFrame(columns=['Feature'], data=unscaled_inputs.columns.values)
summary_table['Coefficients'] = lr_model.coef_.T # Takiign the transpose
summary_table


# In[ ]:


# Let's add intercept as well
summary_table.index = summary_table.index+1
summary_table.loc[0] = ['Intercept', lr_model.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table


# In[ ]:


summary_table.sort_values('Coefficients', ascending=False)


# How do we interpret coefficients? The closer a coefficient to zero, the less its predictive power. In logistic regression we take the log meaning coefficients are odds. So if an odd is close to zero that means for one standard deviation increase (not unit because we stadardized variables) it is close to zero times as likely a person will be absent. So in this case, 
# Service time, reason_4, Body mass index, Age, Day of the week, Distance from Residence to Work are not nessary. 
# If you go up all the way to data_with_targets check point and drop these variables and and rerun all the notebook you will get similar result. 
#  

# # Testing the Model
# Traingin accuracy doesn't show anything we should test our model on unseen data meaning we should run our model on X_test 

# In[ ]:


lr_model.score(X_test, y_test)


# 

# Test accuracy almost always is lower than training accuracy due overfitting. This model definitely can be improved but that is it from me, it has been an exhaustive work. Feel free to imporve it. Don't forget to up vote it, if you like it.

# In[ ]:




