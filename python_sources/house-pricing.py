#!/usr/bin/env python
# coding: utf-8

# **PREDICTING HOUSE PRICES**
# 
# The aim of this competition is to predict the house prices based on the various informations provided about the houses in the dataset.
# 
# The datasets have been provided by Ames Housing Dataset, and compiled by Mr. Dean de Cock.

# In[36]:


import os
print(os.listdir("../input"))


# **Importing Libraries**
# 
# We require these packages to manipulate the data and visualize the features/labels and check how well our model performs. For Data Visualisation, we are using seaborn and matplotlib.
# 
# For making a Neural Network, we use Keras.
# We probably aren't going to use Neural Network approach for Regression problems like these, because it is less favourable than other methods.
# 
# We use scikit-learn (sklearn) for implementing various models like Random Forest Classifier, SVC, KNN, Naive Bayes, Decision Trees, etc.

# In[37]:


# For making Neural Networks, if required

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

# For using standard machine learning

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import make_scorer, accuracy_score 
from sklearn.model_selection import GridSearchCV

# Data Visualisation Tools

import seaborn as sb
import matplotlib.pyplot as plt

# Standard required libraries

import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor


# In[38]:


def get_data():
    '''
    For getting data from the input files.
    '''
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    
    return train , test

# Now we combine the training and testing data to collectively deal with imputations

def get_combined_data():
    '''
    For combining the data of train and test files
    '''
    train, test = get_data()

    target = train.SalePrice
    train.drop(['SalePrice'],axis = 1 , inplace = True)

    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop(['index', 'Id'], inplace=True, axis=1)

    return combined, target


#Loading train and test data into Pandas DataFrames
train_data, test_data = get_data()

#Combine train and test data to process them together
combined, target = get_combined_data()


# In[39]:


combined.describe()


# In[40]:


combined.head()


# Now, we need to get the columns that don't have any missing values. As missing values may create problems.

# In[41]:


def get_cols_with_no_nans(df,col_type):
    '''
    Function to return columns that have no missing values.
    
    df : The dataframe to process
    col_type : 
          num : to only get numerical columns with no nans
          no_num : to only get nun-numerical columns with no nans
          all : to get any columns with no nans    
    '''
    if (col_type == 'num'):
        check = df.select_dtypes(exclude=['object'])
    elif (col_type == 'no_num'):
        check = df.select_dtypes(include=['object'])
    elif (col_type == 'all'):
        check = df
    else :
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in check.columns :
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans


# In[42]:


# Columns with numeric values
num_cols = get_cols_with_no_nans(combined , 'num')

# Columns with categorical values
cat_cols = get_cols_with_no_nans(combined , 'no_num')


# In[43]:


print ('Number of numerical columns with no nan values :',len(num_cols))
print ('Number of nun-numerical columns with no nan values :',len(cat_cols))


# In[44]:


combined = combined[num_cols + cat_cols]
combined.hist(figsize = (12,10))
plt.show()


# ***One Hot Encoding***
# 
# One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction

# In[45]:


def oneHotEncode(df,colNames):
    '''
    Function to implement One Hot Encoding
    
    df : The dataframe where it is to be implemented
    colNames : The columns which do not have any numeric values
    '''
    for col in colNames:
        if( df[col].dtype == np.dtype('object')):
            dummies = pd.get_dummies(df[col],prefix=col)
            df = pd.concat([df,dummies],axis=1)

            #drop the encoded column
            df.drop([col],axis = 1 , inplace=True)
    return df


# In[46]:


print('There were {} columns before encoding categorical features'.format(combined.shape[1]))
combined = oneHotEncode(combined, cat_cols)
print('There are {} columns after encoding categorical features'.format(combined.shape[1]))


# As we have performed the imputations, now we can split the combined dataset into training and testing sets again.

# In[47]:


def split_combined():
    '''
    Function to split combined training and testing sets into individual dataframes.
    '''
    global combined
    train = combined[:1460]
    test = combined[1460:]

    return train , test 


# In[48]:


train, test = split_combined()


# **Function to make a submission**

# In[49]:


def create_submission(prediction, sub_name):
    '''
    Function to finally submit our predicted prices to Kaggle.
    
    prediction : predicted prices by the model
    sub_name : name of the to be submitted file
    '''
    submission = pd.DataFrame({'Id':pd.read_csv('../input/test.csv').Id,'SalePrice':prediction})
    submission.to_csv('{}.csv'.format(sub_name),index=False)
    print('A submission file has been made')


# **Creating a Validation Set**
# 
# Validation or Cross-Validation sets are required to test our model how it is performing before actually applying it to the final test set.

# In[50]:


train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.2, random_state = 16)


# ***Random Forest***
# 
# First, we use Random Forest Classifier to predict the data.

# In[51]:


RandomForest = RandomForestRegressor()
RandomForest.fit(train_X,train_y)


# In[52]:


# Making a prediction
predicted_prices = RandomForest.predict(val_X)

# Calculating the mean absolute error
mae = mean_absolute_error(val_y , predicted_prices)
print('Random forest validation Mean Absolute Error = ', mae)


# Now, we finally create our first submission using Random Forest Classifier.

# Now, we use ***Support Vector Machine (SVM)*** model. 

# In[53]:


SVM = SVC()
SVM.fit(train_X,train_y)

# Making a prediction
predicted_prices = SVM.predict(val_X)

# Calculating the mean absolute error
mae = mean_absolute_error(val_y , predicted_prices)
print('SVM validation Mean Absolute Error = ', mae)


# Now, we use ***Linear Support Vector Machine (SVM)*** model. 

# In[54]:


LSVM = LinearSVC()
LSVM.fit(train_X,train_y)

# Making a prediction
predicted_prices = LSVM.predict(val_X)

# Calculating the mean absolute error
mae = mean_absolute_error(val_y , predicted_prices)
print('LinearSVM validation Mean Absolute Error = ', mae)


# Now, we use ***K-Neighbours Classifier (KNN)*** model.

# In[55]:


KNN = KNeighborsClassifier()
KNN.fit(train_X,train_y)

# Making a prediction
predicted_prices = KNN.predict(val_X)

# Calculating the mean absolute error
mae = mean_absolute_error(val_y , predicted_prices)
print('KNN validation Mean Absolute Error = ', mae)


# Now, we use ***Naive Bayes*** model.

# In[56]:


Naive_Bayes = GaussianNB()
Naive_Bayes.fit(train_X,train_y)

# Making a prediction
predicted_prices = Naive_Bayes.predict(val_X)

# Calculating the mean absolute error
mae = mean_absolute_error(val_y , predicted_prices)
print('Naive Bayes validation Mean Absolute Error = ', mae)


# Now, we use ***Decision Tree Classifier*** model.

# In[57]:


DecisionTree = DecisionTreeClassifier()
DecisionTree.fit(train_X,train_y)

# Making a prediction
predicted_prices = DecisionTree.predict(val_X)

# Calculating the mean absolute error
mae = mean_absolute_error(val_y , predicted_prices)
print('Decision Tree validation Mean Absolute Error = ', mae)


# Now, we use ***XGBRegressor*** model.

# In[58]:


XGB = XGBRegressor()
XGB.fit(train_X,train_y, verbose=False)

# Making a prediction
predicted_prices = XGB.predict(val_X)

# Calculating the mean absolute error
mae = mean_absolute_error(val_y , predicted_prices)
print('XGBRegressor validation Mean Absolute Error = ', mae)


# Now, we use ***Stochastic Gradient Descent (SGD) Regressor*** model.

# In[59]:


SGDClass = SGDClassifier()
SGDClass.fit(train_X,train_y)

# Making a prediction
predicted_prices = SGDClass.predict(val_X)

# Calculating the mean absolute error
mae = mean_absolute_error(val_y , predicted_prices)
print('SGD Classifier validation Mean Absolute Error = ', mae)


# Now, we use ***Lasso Regressor*** model.

# In[60]:


LassoR = Lasso()
LassoR.fit(train_X,train_y)

# Making a prediction
predicted_prices = LassoR.predict(val_X)

# Calculating the mean absolute error
mae = mean_absolute_error(val_y , predicted_prices)
print('SGD Classifier validation Mean Absolute Error = ', mae)


# Now, we use ***Elastic Net Regression*** model.

# In[61]:


ElasticNet = ElasticNet()
ElasticNet.fit(train_X,train_y)

# Making a prediction
predicted_prices = ElasticNet.predict(val_X)

# Calculating the mean absolute error
mae = mean_absolute_error(val_y , predicted_prices)
print('Elastic Net Regressor validation Mean Absolute Error = ', mae)


# **Hyperparameter Tuning**
# 
# We can observe that XGBRegressor works best with our validation set, so let's try to improve on it by tuning the hyperparameters. We use GridSearchCV for that.

# In[63]:


XGBR = XGBRegressor()
parameters = {'nthread':[4],                        #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [0.003, .03, 0.05, .07],    #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

XGB_GCV = GridSearchCV(XGBR, parameters, cv = 2, n_jobs = 5, verbose=True)

XGB_GCV.fit(train_X,train_y)

print(XGB_GCV.best_score_)
print(XGB_GCV.best_params_)

# Making a prediction
predicted_prices = XGB_GCV.predict(val_X)

# Calculating the mean absolute error
mae = mean_absolute_error(val_y , predicted_prices)
print('After hyperparameter tuning to XGB Regressor model, validation Mean Absolute Error = ', mae)


# As we saw, we were able to succesfully reduce our Mean Absolute Error from 19671.2 to 18946.2 after tuning the hyperparameters, and checking which hyperparameters work best with our data and model.
# 
# So, now we finally make our submission with the model and hyperparameters with least Mean Absolute Error, with **Hyperparameter tuned XGB Regressor model**.

# In[ ]:


predicted_prices = XGB_GCV.predict(test)
create_submission(predicted_prices,'PredictionXGB_GCV')


# We have succesfully submitted our predictions for prices of houses.
