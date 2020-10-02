#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:


# Reading the main data file "application_train/test.csv" for baseline model
train_data = pd.read_csv("../input/application_train.csv")
print('training data shape: ', train_data.shape)

test_data = pd.read_csv("../input/application_test.csv")
print('testing data shape: ', test_data.shape)


# In[5]:


# Looking at different features in training data
train_data.head()


# In[6]:


# Looking at overall statistics of training data
train_data.describe()


# Here are the observations from the first look at data - 
# 
# 1. There are few columns with object type - need to change into categorical data
# 2. There are many columns with missing values - need to fix it
# 3. There are few columns with high standard deviation - need to normalize the data

# ### Examine the Distribution of the Target Column
# The target is what we are asked to predict: either a 0 for the loan was repaid on time, or a 1 indicating the client had payment difficulties. We can first examine the number of loans falling into each category.

# In[7]:


#Lets first drop the target column from the train data
target = train_data['TARGET']
#train_data = train_data.drop(columns = ['TARGET'])
#print('training data shape: ', train_data.shape)


# In[8]:


# Lets look the the target distribution in the data set
target.value_counts()


# In[9]:


# Now plot the histogram to visualize it further
target.plot.hist()


# Here we observe that it is an imbalanced class problem. There are far more loans that were repaid on time than loans that were not repaid.

# ### Column Types
# Let's look at the number of columns of each data type. int64 and float64 are numeric variables (which can be either discrete or continuous). object columns contain strings and are categorical features. .

# In[10]:


train_data.dtypes.value_counts()


# ### Encoding Categorical Variables
# Before we go any further, we need to deal with these categorical variables. A machine learning model unfortunately cannot deal with categorical variables (except for some models such as LightGBM). Therefore, we have to find a way to encode (represent) these variables as numbers before handing them off to the model. There are two main ways to carry out this process:
# 
# 1. Label encoding: assign each unique category in a categorical variable with an integer. No new columns are created.
# 
# 2. One-hot encoding: create a new column for each unique category in a categorical variable. Each observation recieves a 1 in the column for its corresponding category and a 0 in all other new columns.
# 
# The problem with label encoding is that it gives the categories an arbitrary ordering. The value assigned to each of the categories is random and does not reflect any inherent aspect of the category. Therefore, when we perform label encoding, the model might use the relative value of the feature (for example programmer = 4 and data scientist = 1) to assign weights which is not what we want. If we only have two unique values for a categorical variable (such as Male/Female), then label encoding is fine, but for more than 2 unique categories, one-hot encoding is the safe option.
# The only downside to one-hot encoding is that the number of features (dimensions of the data) can explode with categorical variables with many categories. To deal with this, we can perform one-hot encoding followed by PCA or other dimensionality reduction methods to reduce the number of dimensions.

# ### One Hot Encoding for all categorical data

# In[11]:


# Utility function to change Object type training data to categorical data
def one_hot_encoding(train_data, test_data):
    """
    examine columns with object type in training and test data
    do one hot encoding of such columns
    """

    encoded_train_data = pd.get_dummies(train_data)
    encoded_test_data = pd.get_dummies(test_data)
    return encoded_train_data, encoded_test_data


# In[12]:


encoded_train_data, encoded_test_data = one_hot_encoding(train_data, test_data)
print('Training Features shape after one hot encoding: ', encoded_train_data.shape)
print('Testing Features shape after one hot encoding: ', encoded_test_data.shape) 


# ### Aligning Training and Testing Data
# There need to be the same features (columns) in both the training and testing data. One-hot encoding has created more columns in the training data because there were some categorical variables with categories not represented in the testing data. To remove the columns in the training data that are not in the testing data, we need to align the dataframes. When we do the align, we must make sure to set axis = 1 to align the dataframes based on the columns and not on the rows!

# In[13]:


align_train_data, align_test_data = encoded_train_data.align(encoded_test_data, join = 'inner', axis = 1)
align_train_data['TARGET'] = target
print('Training Features shape: ', align_train_data.shape)
print('Testing Features shape: ', align_test_data.shape)


# The training and testing datasets now have the same features which is required for machine learning. The number of features has grown significantly due to one-hot encoding. At some point we probably will want to try dimensionality reduction (removing features that are not relevant) to reduce the size of the datasets.

# ### Examine Missing Values
# Now we will examine the columns with missing values

# In[14]:


missing_val_count_by_column = align_train_data.isnull().sum()
missing_val_count_by_column = missing_val_count_by_column[missing_val_count_by_column > 0]
print('Number of columns with missing values: ', missing_val_count_by_column.shape[0])
missing_val_count_by_column.head()


# When it comes time to build our machine learning models, we will have to fill in these missing values (known as imputation). In later work, we will use models such as XGBoost that can handle missing values with no need for imputation. Another option would be to drop columns with a high percentage of missing values, although it is impossible to know ahead of time if these columns will be helpful to our model. Therefore, we will keep all of the columns for now.

# ### Back to Exploratory Data Analysis
# Lets look at the statistics of columns again to have a deeper look of training data and identify all important features

# In[15]:


align_train_data.describe()


# In[16]:


align_test_data.describe()


# Here we obsereve that few features have negative mean values which looks suspective. This might be a sign of data anomalies.
# Below are the features we need to look at - 

# In[17]:


col_with_mean = align_train_data.mean(axis = 0)
col_with_negative_mean = col_with_mean[col_with_mean < 0]
col_with_negative_mean


# It seems all the days are past number of days relative to the present day when loan application is registered. We are just going to change it in years to find the age of applicants.

# In[18]:


col_name_with_negative_mean = ['DAYS_BIRTH','DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']
align_train_data[col_name_with_negative_mean] = align_train_data[col_name_with_negative_mean]/-365
align_test_data[col_name_with_negative_mean] = align_test_data[col_name_with_negative_mean]/-365
align_train_data.describe()


# Age looks reasonable now. If we look at the DAYS_EMPLOYED column, it seems very unreasonable. The maximum number is close to 1000 years.

# In[19]:


import matplotlib.pyplot as plt
align_train_data['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');


# Here we go..
# We have bunch of entries which seems unreasonable. We have to remove those. Lets consider the maximum age of the person who is applying is 100.

# In[20]:


col_with_anom = align_train_data['DAYS_EMPLOYED'] >= 36500
print('Number of anomalies : ', col_with_anom.sum())
align_train_data['DAYS_EMPLOYED'][col_with_anom].value_counts()


# Here we see that all the anomalies are subjected to only one number - 365243. We will replace this number to NaN.

# In[21]:


# Replace the anomalous values with nan
align_train_data['DAYS_EMPLOYED'].replace({365243 : np.nan}, inplace=True)
align_test_data['DAYS_EMPLOYED'].replace({365243 : np.nan}, inplace=True)
align_train_data['DAYS_EMPLOYED'].describe()


# As usual, the day counts are negative since it is relative to the date of load application. Let's make it postitive for better visualization.

# In[22]:


align_train_data['DAYS_EMPLOYED'] = abs(align_train_data['DAYS_EMPLOYED'])
align_test_data['DAYS_EMPLOYED'] = abs(align_test_data['DAYS_EMPLOYED'])

#Now plot histogram to visualize days of employment distribution
align_train_data['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');


# ### Correlation
# Now we will look at the correlation of these features with target value in training dataset...

# In[23]:


correlations = align_train_data.corr()['TARGET'].sort_values()
# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))


# We have all the feature correlation with the target. We are going to consider only those features whose abs correlation is more than 0.7.
# So, below are the features we are going to consider for our model - 
# 1. EXT_SOURCE_3/2/1
# 2. DAYS_BIRTH
# 3. DAYS_EMPLOYED

# In[24]:


feature_col = ['EXT_SOURCE_3','EXT_SOURCE_2','EXT_SOURCE_1','DAYS_BIRTH','DAYS_EMPLOYED']
X = align_train_data[feature_col]
Y = align_train_data['TARGET']
X_test = align_test_data[feature_col]
print('Training Features shape : ', X.shape)
print('Testing Features shape : ', X_test.shape) 


# In[25]:


# have a final look at train  data to check if we miss something
X.describe()


# In[26]:


# have a final look at test data to check if we miss something
X_test.describe()


# ### Handling Missing value

# In[27]:


# Utility functions to fix missing values    
def impute_missing_values(train_data, test_data):
    """
    check for the missing values and impute it with mean values of the columns
    in both train and test data
    """
    from sklearn.preprocessing import Imputer
    imputer = Imputer(strategy = 'median')
    imputed_train_data = imputer.fit_transform(train_data)
    imputed_test_data = imputer.transform(test_data)
    return imputed_train_data, imputed_test_data


# In[28]:


#impute the missing values
X, X_test = impute_missing_values(X, X_test)
print('training data shape after imputing missing values: ', X.shape)
print('testing data shape after imputing missing values: ', X_test.shape)


# ### Normalize tarining and test data

# In[29]:


# Utility functions to normalize data   
def normalize_values(train_data, test_data):
    """
    Normalizing the feature data values using MinMaxScaler library 
    in both train and test data
    """
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range = (0, 1)) 
    normalized_train_data = scaler.fit_transform(train_data)
    normalized_test_data = scaler.transform(test_data)
    return normalized_train_data, normalized_test_data


# In[30]:


#normalize data
X, X_test = normalize_values(X, X_test)
print('training data shape after normalizing values: ', X.shape)
print('testing data shape after normalizing values: ', X_test.shape)


# ### Creating improved model
# 1. Random forest - To try and beat the poor performance of our baseline, we can update the algorithm. Let's try using a Random Forest on the same training data to see how that affects performance. The Random Forest is a much more powerful model especially when we use hundreds of trees. We will use 100 trees in the random forest.
# 2. XGBoost

# In[39]:


#Utility function to create logistic regression model
import time   
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
    
def create_random_forest_model(X, Y):
    """
    split data into training and validation data, for both features and target
    The split is based on a random number generator. Supplying a numeric value to
    the random_state argument guarantees we get the same split every time we
    run this script.
    
    create model using sklearn random forest library and measuere model performance
    """

    print('Starting random forest model training...')
    t0 = time.time()
    
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state = 0, test_size=0.2)

    # Make the random forest classifier
    model = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)

    # Train on the training data
    model.fit(X_train, Y_train)
    
    accuracy = model.score(X_val, Y_val)
    print('Accuray of random forest model is : ', accuracy)
    
    Y_pred = model.predict(X_val)
    mae = mean_absolute_error(Y_val, Y_pred)
    print('mean absoute error of random forest model is : ', mae)
    
    t1 = time.time()
    print('Time elapsed during random forest model training is : ', t1-t0)
    
    return model


# In[40]:


# create logistic regression model
random_forest_model = create_random_forest_model(X,Y)


# In[47]:


from xgboost import XGBRegressor

def create_xgboost_model(X, Y):
    """
    split data into training and validation data, for both features and target
    The split is based on a random number generator. Supplying a numeric value to
    the random_state argument guarantees we get the same split every time we
    run this script.
    
    create model using xgboost library and measuere model performance
    """

    print('Starting xgboost model training...')
    t0 = time.time()
    
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state = 0, test_size=0.2)

    # Make the random forest classifier
    model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

    # Train on the training data
    model.fit(X_train, Y_train, early_stopping_rounds=5, eval_set=[(X_val, Y_val)], verbose=False)
    
    accuracy = model.score(X_val, Y_val)
    print('Accuray of xgboost model is : ', accuracy)
    
    Y_pred = model.predict(X_val)
    mae = mean_absolute_error(Y_val, Y_pred)
    print('mean absoute error of xgboost model is : ', mae)
    
    t1 = time.time()
    print('Time elapsed during xgboost model training is : ', t1-t0)
    
    return model


# In[48]:


#xgboost_model = create_xgboost_model(X,Y)


# Now that the model has been trained, we can use it to make predictions. We want to predict the probabilities of not paying a loan, so we use the model predict.proba method. This returns an m x 2 array where m is the number of observations. The first column is the probability of the target being 0 and the second column is the probability of the target being 1 (so for a single row, the two columns must sum to 1). We want the probability the loan is not repaid, so we will select the second column.

# In[49]:


# generate the prediction for test data
# Make sure to select the second column only
random_forest_pred = random_forest_model.predict_proba(X_test)[:,1]


# The predictions must be in the format shown in the sample_submission.csv file, where there are only two columns: SK_ID_CURR and TARGET. We will create a dataframe in this format from the test set and the predictions called submit.

# In[50]:


# Submission dataframe
submit = align_test_data[['SK_ID_CURR']]
submit['TARGET'] = random_forest_pred
submit.head()


# The predictions represent a probability between 0 and 1 that the loan will not be repaid. If we were using these predictions to classify applicants, we could set a probability threshold for determining that a loan is risky.

# In[51]:


# Save the submission to a csv file
submit.to_csv('random_forest_baseline.csv', index = False)

