#!/usr/bin/env python
# coding: utf-8

# # Simple predictions based on hour and distance

# This notebook creates two new features for the taxi database set, culls outliers, fits a gradient boosted regressor to the data. 
# 
# With 200 trees, it generates a prediction in the to 62%. Note, I have trees set to 50 for a quicker first run of the notebook.
# 
# This is my first kernel and and so thank you for checking it out. Constructive criticism is welcome.

# # I. Imports

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')


# # II. Data Overview

# In[ ]:


train.head(1)


# In[ ]:


test.head(1)


# In[ ]:


sample_submission.head(1)


# In[ ]:


train.columns


# In[ ]:


test.columns


# In[ ]:


len(train)


# In[ ]:


len(test)


# In[ ]:


# Extra features in train (which are note present in test)
[column for column in train.columns if column not in test.columns]


# # III. Cleaning

# In[ ]:


train.isnull().sum()


# In[ ]:


train[train.isnull().any(axis=1)] #No missing values


# In[ ]:


def remove_outliers(old_df,number_of_std,columns="All",skip="None"):
    """
    Removes outliers from a dataframe.
    
    Parameters:
    old_df: Series or dataframe
    
    number_of_std: Number of standard deviations for threshhold. 
                   Function will remove all outliers beyond this many standard deviations.
                   
    columns: The columns upon which the operation will be performed. (List of column names)
    
    skip: List of columns to be skipped.
    
    Returns:
    A dataframe with the outliers removed.
    
    """
    
    if isinstance(old_df,pd.core.series.Series): #If series passed, then only 
        current_series = old_df #set current series
        
        mean = np.mean(current_series)    #Mean
        std = np.std(current_series)      #Std
        threshold = number_of_std*std     #Threshhold = number of std * std
        
        new_df = old_df[np.abs(current_series-mean)<threshold] #Remove outliers from series
    else:
        if columns=="All": #Set columns
            columns=old_df.columns
            
        if skip!="None": #Skip any columns to be skipped
            columns = columns-skip
        
        for column in columns:
            current_series = old_df[column] #Iterate through each column

            mean = np.mean(current_series) #Set up threshold for which x should be within
            std = np.std(current_series)
            threshold = number_of_std*std

            new_df = old_df[np.abs(current_series-mean)<threshold] #Remove outliers from this column
    
    return new_df


# In[ ]:


#Remove outlier trips (in case length was caused by unusual circumstances)
#Outlier here defined as points more than 4 standard deviations from the mean (approx 0.3%)
train = remove_outliers(train,4,columns=['trip_duration']) 


# ***
# # IV. Feature Engineering
# 
# Feature engineering:
# - Distance
# - Hour of departure

# ### FE 1: 'dist' (distance travelled)

# In[ ]:


train['dist'] = np.sqrt((train['pickup_latitude']-train['dropoff_latitude'])**2 
                         + (train['pickup_longitude']-train['dropoff_longitude'])**2) 

test['dist'] = np.sqrt((test['pickup_latitude']-test['dropoff_latitude'])**2 
                         + (test['pickup_longitude']-test['dropoff_longitude'])**2) 


# ### FE 2: 'hour' (hour picked up)

# In[ ]:


train['hour'] = train['pickup_datetime'].apply(lambda x: int(x.split()[1][0:2]))

test['hour'] = test['pickup_datetime'].apply(lambda x: int(x.split()[1][0:2]))


# ***
# # V. Visualization

# In[ ]:


#train[['hour','dist','trip_duration']].corr()


# In[ ]:


#sns.barplot(x='hour',y='trip_duration',data=train)


# ***
# # VI. Fitting and predicting

# ### Model 1: Linear Regression

# In[ ]:


lm = LinearRegression()


# In[ ]:


X_train = train['dist'].values.reshape(-1,1)
y_train = train['trip_duration'].values.reshape(-1,1)

X_test = test['dist'].values.reshape(-1,1)


# In[ ]:


lm.fit(X_train,y_train)


# In[ ]:


lm.score(X_train,y_train)


# In[ ]:


lm_pred = lm.predict(X_test)


# ### Model 2: Gradient Boosting Regressor()

# In[ ]:


#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(train[['dist','hour']], train['trip_duration'], test_size=0.3, random_state=42)
#x_test = test[['dist','hour']]#.values.reshape(-1,1)

X_train = train[['dist','hour']]#.values.reshape(-1,1)
y_train = train['trip_duration']#.values.reshape(-1,1)

X_test = test[['dist','hour']]#.values.reshape(-1,1)


# In[ ]:


gm = GradientBoostingRegressor(n_estimators=50) 
#Change n_estimators to a higher number (e.g. 100, 150 or 200) for a more accurate score
#Note: it takes a long time to run with a higher number of estimators.

gm.fit(X_train,y_train)

gm.score(X_train,y_train)


# In[ ]:


gm.fit(X_train,y_train)


# In[ ]:


gm_pred = gm.predict(X_test)


# In[ ]:


#0.67998115447801677 - score with 100


# ***
# # X. Exporting submission_df

# In[ ]:


submission_pred = gm_pred
submission_name = 'Gradient Boosting Regressor - 200 estimators on hour and distance.csv'


# In[ ]:


submission_df = pd.DataFrame(submission_pred,index=test['id'],columns=['trip_duration']).reset_index()

submission_df.to_csv(submission_name,index=False)


# ***
# # Thank you for checking out this kernel.
