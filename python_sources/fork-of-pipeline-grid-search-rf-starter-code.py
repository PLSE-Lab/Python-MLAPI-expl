#!/usr/bin/env python
# coding: utf-8

# This is my first kernel.  I am new to machine learning so any help would be much appreciated! I created a kernel with a random forest regressor because these regressors are a great place to start and have the following advantages:
# 
#  + Limited parameter tuning ... default parameters often have great performance
#  + Fast
#  + Versitile
#  + Great for feature selection because it evaluates a lot of decision tree variations
# 
# Additionally, I used a pipeline to help with the data preparation steps:
#  + selector - Converst DataFrame to a NumPy array.
#  + Imputer - There is a lot of missing data in this set, so NaN's are replaced with median
#  + std_scaler - Not needed for decision trees, but a good step for future algorithms
#  + KBest - Selects the most important features
#  + PCA - Transforms the features into correlated components
#  + Random Forest Regressor - Used to make predictions
# 
# Finally, I ran a grid search to tune the pipeline paramaters for KBest, PCA, and the Random Forest Regressor

# In[1]:


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import pandas as pd

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


# In[2]:


train_df = pd.read_csv('../input/train_2016_v2.csv', parse_dates=["transactiondate"])


# In[3]:


train_df.info()


# In[4]:


test_df = pd.read_csv('../input/sample_submission.csv')


# In[5]:


test_df = test_df.rename(index=str, columns={"ParcelId": "parcelid"})


# In[6]:


properties_2016_df = pd.read_csv('../input/properties_2016.csv')


# In[7]:


train_df = pd.merge(train_df, properties_2016_df, on='parcelid', how='left')


# In[8]:


test_df = pd.merge(test_df, properties_2016_df, on='parcelid', how='left')


# In[9]:


# to make this notebook's output identical at every run
np.random.seed(42)


# ### Prepare the data for Machine Learning algorithms

# In[10]:


#remove categorical variables... I plan to come back to these later by using one hot encoding.
train_df = train_df.drop(['propertyzoningdesc', 'propertycountylandusecode', 'transactiondate', 'parcelid',
                         'taxdelinquencyflag', 'fireplaceflag'], axis = 1)


# In[11]:


# We need to ensure the test set has the same features as the training set.
test_df = test_df.drop(['propertyzoningdesc', 'propertycountylandusecode', 
                        'parcelid','taxdelinquencyflag', 'fireplaceflag'], axis = 1)


# In[12]:


# this class is used to help with the pipeline.  
# More details on this class can found in chapter 2 of Hands-on Machine Learning with Scikit-Learn and Tensorflow
from sklearn.base import BaseEstimator, TransformerMixin

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# In[13]:


# Create a Y value to predict for our training set
train_df_labels = train_df['logerror'].values


# In[14]:


# Remove the Y value from our training X set
train_df = train_df.drop(['logerror'], axis = 1)


# In[15]:


# split the training set into a test and train set
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split =     train_test_split(train_df, train_df_labels, test_size=0.3, random_state=42)


# In[16]:


# Create a pipeline for quicker data preparation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, confusion_matrix, recall_score
from sklearn.ensemble import RandomForestRegressor

num_attribs = list(train_df)

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),        
        ('KBest', SelectKBest(k = 10)),
        ('pca', PCA(n_components = 5)),
        ('reg', RandomForestRegressor(random_state=42))
         ])
        


# In[17]:


# Create parameters to random search to find an optimal pipeline
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
sss = StratifiedShuffleSplit(100, test_size=0.5, random_state=42)
param_distribs = {
    'KBest__k' : randint(low=1, high=53),
    'pca__n_components' : randint(low=1, high=30),
    'reg__min_samples_split': randint(low=1, high=100)
    }


# In[18]:


# Create the random search
from sklearn.model_selection import RandomizedSearchCV
#Try from sklearn.cross_validation import StratifiedKFold... keeps balancing constant

rnd_search = RandomizedSearchCV(num_pipeline, param_distributions=param_distribs,
                                n_iter=5, cv=5, scoring='neg_mean_squared_error', random_state=42)


# In[19]:


# Fit the random search to the training set
rnd_search_fit = rnd_search.fit(X_train, y_train)


# In[20]:


# return the best parameters of the random search
rnd_search.best_params_


# In[21]:


best_parameters = rnd_search.best_params_


# In[22]:


# return the best model
rnd_search.best_estimator_


# In[23]:


final_model = rnd_search.best_estimator_


# In[24]:


# review each of the 5 searchs of the random search and their respective scores
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[25]:


# save model for future use
from sklearn.externals import joblib

joblib.dump(final_model, "final_model_zillow6.pkl")


# In[26]:


# save paramaters for future use
joblib.dump(best_parameters, "best_param_zillow6.pkl")


# In[27]:


submission_file = pd.read_csv('../input/sample_submission.csv') 


# In[28]:


# Run final predictions
RF_rand_final_predictions_test = final_model.predict(test_df)


# In[29]:


for column in submission_file.columns[submission_file.columns != 'ParcelId']: 
    submission_file[column] = RF_rand_final_predictions_test


# In[30]:


submission_file.to_csv('RF_final_predictions.csv', index=False, float_format='%.4f')

