#!/usr/bin/env python
# coding: utf-8

# ### Create the four parameters required for Bayesian optimization:
# 1.  Domain(space)
# 2. Objective function
# 3. Optimization Algorithym
# 4. Results history

# **Import the required packages**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
import seaborn as sns
from hyperopt import hp
import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer
from hyperopt.pyll.stochastic import sample
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
import ast
import json

nfolds = 5
max_eval = 5


# **Import the Dataset**

# In[ ]:


# Import the train dataset
train = pd.read_csv('../input/home-credit-default-risk/application_train.csv')

# Create the sample data
train = train.sample(n = 16000, random_state = 42)
# Select numeric variables for Bayesian optimization
train = train.select_dtypes('number')

# Create the label and 
labels = train['TARGET'].values.astype(np.int32).reshape((-1, ))
train = train.drop(columns = ['TARGET', 'SK_ID_CURR'])

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size = 0.3, random_state = 42, stratify = labels)

print('Train shape: ', X_train.shape)
print('Test shape: ', X_test.shape)

train.head()

# Create the train dataset in lightgbm format
X_train_lgb = lgb.Dataset(X_train, label = y_train)


# **Create the file and Open the connection for track record or can be used later in Bayesian optimization**

# In[ ]:


OUT_FILE = 'Bayesian_v1.csv'
of_connection = open(OUT_FILE, 'w')
writer = csv.writer(of_connection)

ITERATION = 0

# Write column names
headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 'score']
writer.writerow(headers)
of_connection.close()


# **1) Create the Domain (space)**

# In[ ]:


space = {
    'boosting_type': hp.choice('boosting_type', 
                                            [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1.0)}, 
                                             {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1.0)},
                                             {'boosting_type': 'goss', 'subsample': 1.0}]),
    'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'is_unbalance': hp.choice('is_unbalance', [True, False]),
}


# **2) Create the Objective function (probability model)**

# In[ ]:


def Objective_function(hyperparameters):
    # Keep the track record
    global ITERATION
    ITERATION += 1
    
    # We are calculating the n_estimators as per early stopping. Hence, it has to be updated after every iteration
    if "n_estimators" in hyperparameters:
        del hyperparameters["n_estimators"]
        
    # Extract the boosting type and subsample in proper format
    subsample = hyperparameters["boosting_type"].get("subsample", 1.0)
    hyperparameters["boosting_type"] = hyperparameters["boosting_type"]["boosting_type"]
    
    # Make sure all parameters are integers form
    for i in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        hyperparameters[i] = int(hyperparameters[i])
    
    start = timer()
    # Create the lightgbm model alog with cross validation
    model = lgb.cv(hyperparameters, 
                   X_train_lgb, 
                   num_boost_round=10000, 
                   metrics ="auc",
                   nfold = nfolds
              )
    
    # Run time of model
    run_time = timer() - start
    
    # best score 
    best_score = model["auc-mean"][-1]
    
    # loss
    loss = 1-best_score
    
    # Assigned n_estimators in hyperparameter as per the iterations in model
    hyperparameters["n_estimators"] = len(model["auc-mean"])
    
    # Write to the csv file ('a' means append)
    OUT_FILE = "Bayesian_v1.csv"
    of_connection = open(OUT_FILE, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, hyperparameters, ITERATION, run_time, best_score])
    of_connection.close()

    # Dictionary with information for evaluation
    return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION,
            'train_time': run_time, 'status': STATUS_OK}
    
    


# **3) Create Optimization algorithym**

# In[ ]:


# Create the algorithm
tpe_algorithm = tpe.suggest


# **4) Create Result history**

# In[ ]:


# Record results
trials = Trials()


# **5) Create the Automated function for optimization**

# In[ ]:


# Global variable
global  ITERATION

ITERATION = 0

# Run optimization
best = fmin(fn = Objective_function, space = space, algo = tpe.suggest, trials = trials,
            max_evals = max_eval)

best


# **Sort the trials with lowest loss (highest AUC) first**

# In[ ]:


trials_dict = sorted(trials.results, key = lambda x: x['loss'])
trials_dict[:1]


# **Read the history in the csv format**

# In[ ]:


results = pd.read_csv(OUT_FILE)
new_results = results.copy()
# String to dictionary
new_results['hyperparameters'] = new_results['hyperparameters'].map(ast.literal_eval)
    
# Sort with best values on top
new_results = new_results.sort_values('score', ascending = False).reset_index(drop=True)
    
# Print out cross validation high score
print('The highest cross validation score from Bayesian was {:.5f} found on iteration {}.'.format(new_results.loc[0, 'score'], new_results.loc[0, 'iteration']))


# **Make predictions using best parameters**

# In[ ]:


# Use best hyperparameters to create a model
hyperparameters = new_results.loc[0, 'hyperparameters']
model = lgb.LGBMClassifier(**hyperparameters)
    
# Train and make predictions
model.fit(X_train, y_train)
preds = model.predict_proba(X_test)[:, 1]
    
print('ROC AUC from Bayesian on test data = {:.5f}.'.format(roc_auc_score(y_test, preds)))
    


# **Save the Trials(history) in json format for later use**

# In[ ]:


# Save the trial results
with open('trials.json', 'w') as f:
    f.write(json.dumps(trials_dict))


# To start the training from where it left off, simply load in the `Trials` object and pass it to an instance of `fmin`. (You might even be able to tweak the hyperparameter distribution and continue searching with the `Trials` object because the algorithm does not maintain an internal state. Someone should check this and let me know in the comments!).

# In[ ]:


# MAX_EVALS = 1000

# # Create a new file and open a connection
# OUT_FILE = 'bayesian_trials_1000.csv'
# of_connection = open(OUT_FILE, 'w')
# writer = csv.writer(of_connection)

# # Write column names
# headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 'score']
# writer.writerow(headers)
# of_connection.close()

# # Record results
# trials = Trials()

# global ITERATION

# ITERATION = 0 

# best = fmin(fn = objective, space = space, algo = tpe.suggest,
#             trials = trials, max_evals = MAX_EVALS)

# # Sort the trials with lowest loss (highest AUC) first
# trials_dict = sorted(trials.results, key = lambda x: x['loss'])

# print('Finished, best results')
# print(trials_dict[:1])

# # Save the trial results
# with open('trials.json', 'w') as f:
#     f.write(json.dumps(trials_dict))

