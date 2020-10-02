## House Price: Advanced Regression Technique using H2o

#### Implementing various Advanced regression techniques to predict target on Kaggle's House price problem.


## Import Libraries


import numpy as np    # linear algebra
import pandas as pd    # data processing, CSV file I/O (e.g. pd.read_csv)
import h2o
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import os, time, sys
import warnings
warnings.filterwarnings("ignore")

## Import the dataset

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
test_original = test
print(train.shape)
print(train.columns)


## Clean the dataset



# Identify the columns containing NA values in both train and test dataset
train.isnull().sum().sort_values(ascending = False)
test.isnull().sum().sort_values(ascending = False)
# Remove the columns which contain NA with more than 100 rows
train.drop( ['PoolQC', 'MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'], inplace = True, axis = 'columns')
test.drop( ['PoolQC', 'MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'], inplace = True, axis = 'columns')
print(train.shape)
print(test.shape)
# Identify the columns which contain numeric values
train_numericCol = train.select_dtypes(include=[np.number]).columns.values
print(train_numericCol)
# Fill missing values in the numeric columns.
train.fillna(train.mean(),inplace = True)
test.fillna(test.mean(),inplace = True)
# check if still any NA's are available in the numerical coulumns of train data
train[train_numericCol].isnull().sum().sort_values(ascending = False)


## Initiate the h2o Server


h2o.init(ip="localhost", port=54321)


## Convert data into h2o Frame


train = h2o.H2OFrame(train)
test = h2o.H2OFrame(test)
test_original = h2o.H2OFrame(test_original)


# Split the train dataset
train, valid, test = train.split_frame(ratios=[0.7, 0.15], seed=42)

# Seperate the target data and store it into y variable
y = 'SalePrice'
Id = test['Id']

# remove target and Id column from the dataset and store rest of the columns in X variable
X = list(train.columns)
X.remove(y)
X.remove('Id')
X


## H2o Machine Learning models

#We will now perform training of the models using below H2o supervised algorithms


#Gradient Boosting Machine (RF)

#Random Forest (RF)

#Deep Learning (DL)

### 1. Gradient Boosting Machine (GBM)

# Prepare the hyperparameters
gbm_params = {
                'learn_rate': [0.01, 0.1], 
                'max_depth': [4, 5, 7],
                'sample_rate': [0.6, 0.8],               # Row sample rate
                'col_sample_rate': [0.2, 0.5, 0.9]       # Column sample rate per split (from 0.0 to 1.0)
                }



# Prepare the grid object
gbm_grid = H2OGridSearch(model=H2OGradientBoostingEstimator,   # Model to be trained
                          grid_id='gbm_grid1',                  # Grid Search ID
                          hyper_params=gbm_params,              # Dictionary of parameters
                          search_criteria={"strategy": "Cartesian"}   # RandomDiscrete
                          )



# Train the Model
start = time.time() 
gbm_grid.train(x=X,y=y, 
                training_frame=train,
                validation_frame=valid,
                ntrees=100,      # Specify other GBM parameters not in grid
                score_tree_interval=5,     # For early stopping
                stopping_rounds=3,         # For early stopping
                stopping_tolerance=0.0005,
                seed=1)

end = time.time()
(end - start)/60



# Find the Model grid performance 
gbm_gridperf = gbm_grid.get_grid(sort_by='RMSE',decreasing = False)
gbm_gridperf



# Identify the best model generated with least error
best_gbm_model = gbm_gridperf.models[0]
best_gbm_model


### 2. Random Forest Algorithm



# Prepare the hyperparameters
nfolds = 5
rf_params = {
                'max_depth': [3, 4,5],
                'sample_rate': [0.8, 1.0],               # Row sample rate
                'mtries' : [2,4,3]
                }



# Search criteria for parameter space
search_criteria = {'strategy': "RandomDiscrete",
                   "seed": 1,
                   'stopping_metric': "AUTO",
                   'stopping_tolerance': 0.0005
                   }



# Prepare the grid object
rf_grid = H2OGridSearch(model=H2ORandomForestEstimator,   # Model to be trained
                          grid_id='rf_grid',                  # Grid Search ID
                          hyper_params=rf_params,              # Dictionary of parameters
                          search_criteria=search_criteria,   # RandomDiscrete
                          )



# Train the Model
start = time.time() 
rf_grid.train(x=X,y=y, 
                training_frame=train,
                validation_frame=valid,
                ntrees=100,      
                score_each_iteration=True,
                nfolds = nfolds,
                fold_assignment= "Modulo",
                seed=1
                )

end = time.time()
(end - start)/60



# Find the Model performance 
rf_gridperf = rf_grid.get_grid(sort_by='RMSE',decreasing = False)
rf_gridperf



# Identify the best model generated with least error
best_rf_model = rf_gridperf.models[0]
best_rf_model


### 3. Deep Learning Algorithm



activation_opt = ["RectifierWithDropout",
                  "TanhWithDropout"]
#L1 & L2 regularization
l1_opt = [0, 0.00001,
          0.0001,
          0.001,
          0.01,
          0.1]

l2_opt = [0, 0.00001,
          0.0001,
          0.001,
          0.01,
          0.1]



# Create the Hyperparameters
dl_params = {
             'activation': activation_opt,
             "input_dropout_ratio" : [0,0.05, 0.1],  # input layer dropout ratio to improve generalization. Suggested values are 0.1 or 0.2.
             'l1': l1_opt,
             'l2': l2_opt,
             'hidden_dropout_ratios':[[0.1,0.2,0.3], # hidden layer dropout ratio to improve generalization: one value per hidden layer.
                                      [0.1,0.5,0.5],
                                      [0.5,0.5,0.5]]
             }



search_criteria = {
                   'strategy': 'RandomDiscrete',
                   'max_runtime_secs': 1000,
                   'seed':1
                   }



# Prepare the grid object
dl_grid = H2OGridSearch(model=H2ODeepLearningEstimator(
                                                    epochs = 1000,   ## hopefully converges earlier...
                                                    adaptive_rate = True,  # http://cs231n.github.io/neural-networks-3/#sgd
                                                    stopping_metric="AUTO",
                                                    stopping_tolerance=1e-2,    ## stop when misclassification does not improve by >=1% for 2 scoring events
                                                    stopping_rounds=3,
                                                    hidden=[128,128,128],      ## more hidden layers -> more complex interactions
                                                    balance_classes= False,
                                                    standardize = True,  # If enabled, automatically standardize the data (mean 0, variance 1). If disabled, the user must provide properly scaled input data.
                                                    loss = "quantile"  # quantile for regression
                                                    ),
                        grid_id='dl_grid',
                        hyper_params=dl_params,
                        search_criteria=search_criteria)



# Train the Model
start = time.time() 
dl_grid.train(x=X,y=y, 
                training_frame=train,
                validation_frame=valid,
                stopping_rounds=2,
                stopping_tolerance=0.0005,
                seed=1
                )

end = time.time()
(end - start)/60



# Find the Model performance 
dl_gridperf = dl_grid.get_grid(sort_by='RMSE',decreasing = False)
dl_gridperf



# Identify the best model generated with least error
best_dl_model = dl_gridperf.models[0]
best_dl_model


## Compare Model Performances


best_gbm_perf= best_gbm_model.model_performance(test)  # GBM Model
best_rf_perf = best_rf_model.model_performance(test)   # Random Forest Model
best_dl_perf = best_dl_model.model_performance(test)   #deep Learning Model

### Retreive test set AUC
print(best_gbm_perf.gini)
print(best_rf_perf.gini)
print(best_dl_perf.gini)

## Prediction of Model
gbm_pred= best_gbm_model.predict(test_original).as_data_frame()
rf_pred = best_rf_model.predict(test_original).as_data_frame()
dl_pred = best_dl_model.predict(test_original).as_data_frame()

## Submission into kaggle

sub = pd.DataFrame()
sub['Id'] = gbm_pred.index + 1461
sub['SalePrice'] = gbm_pred
sub.head()
sub.to_csv('gbm_h2o.csv', index=False)