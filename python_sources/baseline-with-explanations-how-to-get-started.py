#!/usr/bin/env python
# coding: utf-8

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


# The goal of this kernel is to **establish a baseline **and **explain everything step by step** so anyone can get started. Code to read in and process the data is based on this kernel: <a href='https://www.kaggle.com/inversion/basic-feature-benchmark'>Basic Feature Benchmark.</a>
# 
# Any questions, comments, or suggestions are always welcome.

# In[ ]:


#Read in the training data
train_df = pd.read_csv('../input/train.csv',
                    dtype={'acoustic_data': np.int16,
                           'time_to_failure': np.float64}) 


# By specifying the datatypes, pandas can read in the csv faster and use less memory. Without specifying the datatypes the kernel can crash because the file is so large

# In[ ]:


#Look at the data and realize its not like a normal dataset
train_df.head()


# In[ ]:


train_df.shape


# Instead of each row being one sample, we have one long continous set of data. **We will split this one long sample into many different samples.**
# 
# The first column*** 'acoustic data'***  is our only feature. We use that feature to predict ***'time_to_failure'*.**
# 
# **Each new sample will be created from 150,000 rows** of only the *'acoustic_data'* column. The target variable is the* 'time_to_failure' *at the last row of the sample. (We use 150,000 rows because thats how long the test samples are)

# In[ ]:


#Define how long we want each sample to be
sample_length = 150000

#Divide length of our dataframe by how long we want each sample
num_samples = int((len(train_df) / sample_length)) 


# In[ ]:


#This is how many samples we will create
num_samples


# In[ ]:


#This is a list of features we will create
cols = ['mean','median','std','max',
        'min','var','ptp','10p',
        '25p','50p','75p','90p']


# Our goal is to condense each sample into one row . **We create these features to extract information out of the 150,000 rows that make up each sample. After we extract those features, we can then represent each sample as one row in a dataframe.**

# In[ ]:


#This creates an empty dataframe for now
#Later we will fill it with values
X_train = pd.DataFrame(index=range(num_samples), #The index will be each of our new samples
                       dtype=np.float64, #Assign a datatype
                       columns=cols) #The columns will be the features we listed above


# In[ ]:


#This creates a dataframe for our target variable 'time_to_failure'
y_train = pd.DataFrame(index=range(num_samples),
                       dtype=np.float64, 
                       columns=['time_to_failure']) #Our target variable


# In[ ]:


#Now we create the samples
for i in range(num_samples):
    
    #i*sample_length = the starting index (from train_df) of the sample we create
    #i*sample_length + sample_length = the ending index (from train_df)
    sample = train_df.iloc[i*sample_length:i*sample_length+sample_length]
    
    #Converts to numpy array
    x = sample['acoustic_data'].values
    
    #Grabs the final 'time_to_failure' value
    y = sample['time_to_failure'].values[-1]
    y_train.loc[i, 'time_to_failure'] = y
    
   #For every 150,000 rows, we make these calculations
    X_train.loc[i, 'mean'] = np.mean(x)
    X_train.loc[i, 'median'] = np.median(x)
    X_train.loc[i, 'std'] = np.std(x)
    X_train.loc[i, 'max'] = np.max(x)
    X_train.loc[i, 'min'] = np.min(x)
    X_train.loc[i, 'var'] = np.var(x)
    X_train.loc[i, 'ptp'] = np.ptp(x) #Peak-to-peak is like range
    X_train.loc[i, '10p'] = np.percentile(x,q=10) 
    X_train.loc[i, '25p'] = np.percentile(x,q=25) #We can also grab percentiles
    X_train.loc[i, '50p'] = np.percentile(x,q=50)
    X_train.loc[i, '75p'] = np.percentile(x,q=75)
    X_train.loc[i, '90p'] = np.percentile(x,q=90)


# In[ ]:


#Creates a simple train, test split
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)


# In[ ]:


#Fit a random forest
from sklearn.ensemble import RandomForestRegressor

#This creates the Randomforest with the given parameters
rf = RandomForestRegressor(n_estimators=100, #100 trees (Default of 10 is too small)
                          max_features=0.5, #Max number of features each tree can use 
                          min_samples_leaf=30, #Min amount of samples in each leaf
                          random_state=11)

#This trains the random forest on our training data
rf.fit(X_train,y_train)


# **n_estimators**: the number of decision trees in our random forest. 
# 
# **max_features**: The max amount of features each decision tree can use. This can be an integer (max_features=10 means use 10 features) or a ratio (0.5 means each tree is fit using half of the original features. Features are selected at random) 
# 
# **min_samples_leaf**: Minimum number of samples in each leaf of the decision trees

# In[ ]:


#Score the model
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_val, rf.predict(X_val))


# Now that you have a baseline, here are some next steps to explore:
# 
# * Try to think of any other useful features you can extract for each sample. Here are more summary statics to try from <a href='https://docs.scipy.org/doc/scipy/reference/stats.html#summary-statistics'>SciPy</a>. SciPy also has different signal processing functions to check out.
# 
# * Calculate the min, max, mean, etc over smaller chunks of the 150,000 rows per sample and add those as more features.
# 
# * Create more samples. Instead of our training data being generated from seperate sections of 150,000 rows with no overlap, you can adjust the indices so that the samples overlap. If you do this be very careful to make sure no samples in your validation set overlap with your training data.
# 
# * Try a LightGBM, XGBoost, or NN model.
# 
# * Plot some samples to explore the data
# 
# * Plot feature importances to see what your model finds important.

# How to submit predictions from the kernel:

# In[ ]:


#Read in the sample submission. We can use that as a dataframe to grab the segment ids
submission = pd.read_csv('../input/sample_submission.csv',
                         index_col = 'seg_id')


# In[ ]:


submission.head()


# In[ ]:


#Creates a test dataframe
X_test = pd.DataFrame(columns=X_train.columns, #Use the same columns as our X_train
                      dtype=np.float64,
                      index=submission.index) #Use the index ('seg_id') from the sample submission


# In the test folder each segment is its own csv file. To eactract the features we loop through each segment id one at a time.

# In[ ]:


for i in X_test.index:
    
    #Read in that segments csv file
    #By putting f before the string we can put any values between {} and it will be treated as a string
    seg = pd.read_csv(f'../input/test/{i}.csv') 
                                            
    #Grab the acoustic_data values
    x = seg['acoustic_data'].values

    #These are the same features we calcuted on the training data
    X_test.loc[i, 'mean'] = np.mean(x)
    X_test.loc[i, 'median'] = np.median(x)
    X_test.loc[i, 'std'] = np.std(x)
    X_test.loc[i, 'max'] = np.max(x)
    X_test.loc[i, 'min'] = np.min(x)
    X_test.loc[i, 'var'] = np.var(x)
    X_test.loc[i, 'ptp'] = np.ptp(x)
    X_test.loc[i, '10p'] = np.percentile(x,q=10) 
    X_test.loc[i, '25p'] = np.percentile(x,q=25)
    X_test.loc[i, '50p'] = np.percentile(x,q=50)
    X_test.loc[i, '75p'] = np.percentile(x,q=75)
    X_test.loc[i, '90p'] = np.percentile(x,q=90)
    


# In[ ]:


#Predict on the test data
test_predictions = rf.predict(X_test)

#Assign the target column in our submission to be our predictions
submission['time_to_failure'] = test_predictions


# In[ ]:


#Output the predictions to a csv file
submission.to_csv('submission.csv')


# Now click Commit. After it runs, click Open Version. Scroll down to output and click submit to competition.
