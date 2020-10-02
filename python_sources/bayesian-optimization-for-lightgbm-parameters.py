#!/usr/bin/env python
# coding: utf-8

# In this competition feature selection and feature engineering are 2 very important steps in achieving a good model and hopefully high score. Another important task is choosing the parameters for your tool/model of choice wisely. There are many ways to choose or search those parameters.
# 
# In this notebook I will setup a basic solution to use Bayesian optimization to search for an optimal set of parameters for LightGBM. It should be no problem to modify this code and use it for XGBoost for example.
# 
# Some points to mention upfront. Because of the time needed I specified only 15 initialization rounds and 15 optimization rounds .. however the more rounds the better. I also limited the number of rows used and the maximum iterations for LightGBM. These could also be increased to get better results.
# 
# For more background information visit the github site for Bayesian Optimization package used [https://github.com/fmfn/BayesianOptimization](https://github.com/fmfn/BayesianOptimization)

# In[ ]:


# Import Modules
import pandas as pd
import numpy as np
import gc
import random
import lightgbm as lgbm

import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))


# Let's import the modules needed for Bayesian optimization

# In[ ]:


# Import modules specific for Bayesian Optimization
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events


# The script will run LightGBM 5 folds Cross Validation and will only load the first 1000000 rows of the train set. 

# In[ ]:


# Specify some constants
seed = 4249
folds = 5
number_of_rows = 1000000


# For the features I just choose a couple of them. I'am still working on my own feature selection and engineering ;-)

# In[ ]:


# Select Features
features = ['AVProductStatesIdentifier',
            'AVProductsInstalled', 
            'Census_ProcessorModelIdentifier',
            'Census_TotalPhysicalRAM',
            'Census_PrimaryDiskTotalCapacity',
            'EngineVersion',
            'Census_SystemVolumeTotalCapacity',
            'Census_InternalPrimaryDiagonalDisplaySizeInInches',
            'Census_OSBuildRevision',
            'AppVersion',
            'Census_OEMNameIdentifier',
            'Census_InternalPrimaryDisplayResolutionVertical',
            'Census_ProcessorCoreCount',
            'Census_OEMModelIdentifier',
            'CountryIdentifier',
            'LocaleEnglishNameIdentifier',
            'GeoNameIdentifier',
            'Census_InternalPrimaryDisplayResolutionHorizontal',
            'IeVerIdentifier',
            'HasDetections']


# Load the train dataframe

# In[ ]:


# Load Data with selected features
X = pd.read_csv('../input/train.csv', usecols = features, nrows = number_of_rows)


# Assign the labels to Y and drop the label column from the train dataframe.

# In[ ]:


# Labels
Y = X['HasDetections']

# Remove Labels from Dataframe
X.drop(['HasDetections'], axis = 1, inplace = True)


# 2 columns are factorized. The remainder of the columns are used as-is.

# In[ ]:


# Factorize Some Columns
X['EngineVersion'] = pd.to_numeric(pd.factorize(X['EngineVersion'])[0])
X['AppVersion'] = pd.to_numeric(pd.factorize(X['AppVersion'])[0])


# In[ ]:


# Final Data Shapes
print(X.shape)
print(Y.shape)


# In[ ]:


# Create LightGBM Dataset
lgbm_dataset = lgbm.Dataset(data = X, label = Y)


# I specify a function to run LightGBM Cross Validation with the specified parameters. After running for a maximum of 1250 iterations the function will return the achieved AUC.
# 
# The specified parameters are:
# * learning_rate
# * num_leaves
# * feature_fraction
# * bagging_fraction
# * max_depth

# In[ ]:


# Specify LightGBM Cross Validation function
def lgbm_cv_evaluator(learning_rate, num_leaves, feature_fraction, bagging_fraction, max_depth):
    # Setup Parameters
    params = {  'objective':            'binary',
                'boosting':             'gbdt',
                'num_iterations':       1250, 
                'early_stopping_round': 100, 
                'metric':               'auc',
                'verbose':              -1
            }
    params['learning_rate'] =       learning_rate
    params['num_leaves'] =          int(round(num_leaves))
    params['feature_fraction'] =    feature_fraction
    params['bagging_fraction'] =    bagging_fraction
    params['max_depth'] =           int(round(max_depth))
        
    # Run LightGBM Cross Validation
    result = lgbm.cv(params, lgbm_dataset, nfold = folds, seed = seed, 
                     stratified = True, verbose_eval = -1, metrics = ['auc']) 
    
    # Return AUC
    return max(result['auc-mean'])


# Next we create a function to display a custom progress status for each round of Bayesian Optimization

# In[ ]:


def display_progress(event, instance):
    iter = len(instance.res) - 1
    print('Iteration: {} - AUC: {} - {}'.format(iter, instance.res[iter].get('target'), instance.res[iter].get('params')))


# The following function initializes the BayesianOptimization package with the function to use and the different ranges for the parameters. For each parameter a lower and upper bound is specified.
# Also we subscribe to each Optimization Step a logger to log all results to json file and the function to show the progress.

# In[ ]:


def bayesian_parameter_optimization(init_rounds = 1, opt_rounds = 1):    
    
    # Initialize Bayesian Optimization
    optimizer = BayesianOptimization(f = lgbm_cv_evaluator, 
                                    pbounds = { 'learning_rate':        (0.02, 0.06),
                                                'num_leaves':           (20, 100),
                                                'feature_fraction':     (0.25, 0.75),
                                                'bagging_fraction':     (0.75, 0.95),
                                                'max_depth':            (8, 15) },
                                    random_state = seed, 
                                    verbose = 2)
    
    # Subscribe Logging to file for each Optimization Step
    logger = JSONLogger(path = 'parameter_output.json')
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
    
    # Subscribe the custom display_progress function for each Optimization Step
    optimizer.subscribe(Events.OPTMIZATION_STEP, " ", display_progress)

    # Perform Bayesian Optimization. 
    # Modify acq, kappa and xi to change the behaviour of Bayesian Optimization itself.
    optimizer.maximize(init_points = init_rounds, n_iter = opt_rounds, acq = "ei", kappa = 2, xi = 0.1)
    
    # Return Found Best Parameter values and Target
    return optimizer.max


# Finally we will trigger the optimization process and show the found optimal results. Note that the results from all rounds will be logged to the .json file in the output. In the Kaggle webpage it will show only 1 round..if you download the file you will see the information for all rounds.

# In[ ]:


# Configure and Perform Bayesian Optimization 
max_params = bayesian_parameter_optimization(init_rounds = 15, opt_rounds = 15)

print('================= Results')
print('Found Max AUC: {} with the following Parameters: '.format(max_params.get('target')))
print(max_params.get('params'))


# I hope you enjoyed this notebook and that you can use it for your own benefit.
# 
# Please let me know if you have any questions/remarks/improvements. Those are allways welcome.
