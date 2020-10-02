#!/usr/bin/env python
# coding: utf-8

# # H2O AutoML tutorial on IEEE-CIS Fruad detection compitition

# <img src='https://image.slidesharecdn.com/joeamsautoml-171107063815/95/using-h2o-automl-for-kaggle-competitions-10-638.jpg'></img>

# ### The objective of this kernel is to detect fraudulant transactions taking place on a ecommerce site using H2O AutoML
# 
# ### This kernel helps provide a fair understanding of the H2O's AutoML and it's syntactic nitty gritties. 

# In[ ]:


#Standard imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#to show the large number of features in this dataset
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Loading the H2O library and Starting a local H2O cluster (on your machine)

# In[ ]:


import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.grid.grid_search import H2OGridSearch
# Number of threads, nthreads = -1, means use all cores on your machine
# max_mem_size is the maximum memory (in GB) to allocate to H2O
h2o.init(nthreads = -1, max_mem_size = 16)


# # Data Prep

# We shall first import the data in a pandas dataframe, carry out a memory reduction operation on the dataset (pandas dataframe) and then load it in a H2OFrame.

# In[ ]:


#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings      
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)

    return df, NAlist


# In[ ]:


# import Dataset
train_identity= pd.read_csv("../input/train_identity.csv", index_col='TransactionID')
train_transaction= pd.read_csv("../input/train_transaction.csv", index_col='TransactionID')
test_identity= pd.read_csv("../input/test_identity.csv", index_col='TransactionID')
test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')


# In[ ]:


# Creat our train & test dataset
train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)


# In[ ]:


#Delete the imports to free up memory
del train_identity,train_transaction,test_identity, test_transaction


# ### Memory reduction for train

# In[ ]:


train, NAlist = reduce_mem_usage(train)


# The size of train dataset reduced by almost 65% after the operation (for the sake of simplicity, size reduction logs have not been printed)

# ### Memory reduction for Test

# In[ ]:


test, NAlist = reduce_mem_usage(test)


# The size of train dataset reduced by almost 76% after the operation (for the sake of simplicity, size reduction logs have not been printed)

# In[ ]:


train.head()


# In[ ]:


train.shape


# ### Converting the pandas dataframe (memory reduced) to H2OFrame

# Also, encoding the response variable to factor is important bacause otherwise H2O will asssume it as numeric and will train a regression model instead of a classifiaction model.

# In[ ]:


hf_train = h2o.H2OFrame(train)
hf_test = h2o.H2OFrame(test)
#encode the binary repsonse as a factor
hf_train['isFraud'] = hf_train['isFraud'].asfactor()


# ### Partition data

# In[ ]:


# Partition data into 70%, 15%, 15% chunks
# Setting a seed will guarantee reproducibility

splits = hf_train.split_frame(ratios=[0.7, 0.15], seed=1)  

train_x = splits[0]
valid = splits[1]
test_x = splits[2]


# Notice that split_frame() uses approximate splitting not exact splitting (for efficiency), so these are not exactly 70%, 15% and 15% of the total rows.

# In[ ]:


print(train_x.nrow)
print(valid.nrow)
print(test_x.nrow)


# ### Setting the target and predictor variables
# 
# In H2O, we use y to designate the response variable and x to designate the list of predictor columns.

# In[ ]:


y = 'isFraud'
x = list(hf_train.columns)
x.remove(y)


# # Creating a stacked Emsemble Model

# # 1. Generate a 2-model stacked ensemble (GBM + RF)**
# 
# ### Train and cross-validate a GBM

# In[ ]:


# Number of CV folds (to generate level-one data for stacking)
nfolds = 5


# In[ ]:


my_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",
                                      ntrees=10,
                                      max_depth=3,
                                      min_rows=2,
                                      learn_rate=0.2,
                                      nfolds=nfolds,
                                      fold_assignment="Modulo",
                                      keep_cross_validation_predictions=True,
                                      seed=1)
my_gbm.train(x=x, y=y, training_frame=train_x)


# ### Train and cross-validate a RF

# The default number of trees in an H2O Random Forest is 50, here, we have taken ntrees=100 so this RF will be twice as big as the default. Usually increasing the number of trees in an RF will increase performance as well. 

# In[ ]:


my_rf = H2ORandomForestEstimator(ntrees=100,
                                 nfolds=nfolds,
                                 fold_assignment="Modulo",
                                 keep_cross_validation_predictions=True,
                                 seed=1)
my_rf.train(x=x, y=y, training_frame=train_x)


# ### Train a stacked ensemble using the GBM and RF above

# In[ ]:


ensemble = H2OStackedEnsembleEstimator(model_id="my_ensemble_binomial",
                                       base_models=[my_gbm, my_rf])
ensemble.train(x=x, y=y, training_frame=train_x)


# Eval ensemble performance on the test data

# In[ ]:


stack_test = ensemble.model_performance(test_x)


# # Compare to base learner performance on the test set

# In[ ]:


perf_gbm_test = my_gbm.model_performance(test_x)
perf_rf_test = my_rf.model_performance(test_x)
baselearner_best_auc_test = max(perf_gbm_test.auc(), perf_rf_test.auc())
stack_auc_test = stack_test.auc()
print("Best Base-learner Test AUC:  {0}".format(baselearner_best_auc_test))
print("Ensemble Test AUC:  {0}".format(stack_auc_test))


# In[ ]:


# Generate predictions on test set 
pred = ensemble.predict(hf_test)


# In[ ]:


sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission.shape


# In[ ]:


sample_submission['isFraud'] = pred['p1'].as_data_frame().values
sample_submission.to_csv('h2o_automl_submission_3.csv', index=False)


# This kernel acts as a baseline for H2o AutoML stacked ensembles'. One can further build high performance models by adding/removing base learners, hyperparameter tuning etc.
# 
# **Please upvote the kernel if you found it helpful.**

# In[ ]:




