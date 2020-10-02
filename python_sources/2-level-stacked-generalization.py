#!/usr/bin/env python
# coding: utf-8

# What is this script about?
# =======
# ## 11-8-2016
# I first learned about model stacking from [Thompson's script][1]. I thought the idea is very refreshing and wanted to implement something similar in python on my own. My goal is to write a script that allows others to easily add their models into the stack.
# 
# This script implemented the stacked generalization described at [mlwave.come][2]. In short, the training data is divided into K-fold and each base model (level 0 model) creates predictions for each fold. After generating K predictions, the base model is also fitted to the whole training data and make prediction for the test data. The K-fold predictions from all the base models are then used as training features for the stacker model (level 1 model).
# 
# MLWave did a much better job at explaining the process so go read it from them if you are confused by my admittedly convoluted explanation. I am still fairly new to Python and Data Science so any feedback on my first Kaggle script is more than welcomed. :)
# 
# *Disclimair*: This script taken as is receive a score of 0.12441  at the leaderboard. I did not spend a lot of time doing features engineering and tuning Hyper-parameters since both topics have been covered by a lot of great Kaggle scripts already.  This also means that I did not spend a lot of time exploring different base models and finding the best model to stack the base model results. Something I (or others) may try in the future.
# 
# 
#   [1]: https://www.kaggle.com/jimthompson/house-prices-advanced-regression-techniques/ensemble-model-stacked-model-example
#   [2]: http://mlwave.com/kaggle-ensembling-guide/

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from scipy.stats import skew
import xgboost as xgb

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


train_raw = pd.read_csv('../input/train.csv')
test_raw = pd.read_csv('../input/test.csv')


# In[ ]:


# Join train and test data for some basic preprocessing
df = train_raw.append(test_raw, ignore_index = True)


# In[ ]:


# Fill categorical variables missing data with MISSING string
cats = df.select_dtypes(include = ['O']).fillna('MISSING')
dummies_cats = pd.get_dummies(cats) #Convert categorical variable to dummies


# In[ ]:


# Find the continuous features
cont_features = [col for col in df.select_dtypes(exclude = ['O']).columns if col not in ['SalePrice', 'Id']]
cont_features_df = df[cont_features]


# In[ ]:


# Identify the skewed features and transform them
skewed_features = cont_features_df.columns[skew(cont_features_df) > 0.75]
cont_features_df.loc[:, skewed_features] = np.log1p(cont_features_df[skewed_features])


# In[ ]:


# Impute missing value for continuous features
imp = Imputer(strategy = 'median')
imp_features = imp.fit_transform(cont_features_df)
# Normalize the continuous features
norm_scaler = StandardScaler()
norm_imp_features = norm_scaler.fit_transform(imp_features)


# In[ ]:


# Put the continuous features back into a Pandas dataframe
norm_imp_features = pd.DataFrame(norm_imp_features, columns = cont_features_df.columns, index = dummies_cats.index)


# In[ ]:


# Join the categorical and continuous features back to one dataframe (includes Id, and SalePrice)
analytic_df = dummies_cats.join([norm_imp_features, df.Id, df.SalePrice])
# SPlit the data back to train and test set
train_df = analytic_df[:train_raw.shape[0]]
test_df = analytic_df[train_raw.shape[0]:]


# In[ ]:


# Transform the SalePrice
train_df['logPrice'] = np.log(train_df.SalePrice)


# In[ ]:


# Function to Calulate RMSE
def rmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse


# In[ ]:


# This function takes one model and fit it to the train and test data
# It returns the model RMSE, CV prediction, and test prediction
def baseFit(clf, folds, features, target, trainData, testData):
    # Initialize empty lists and matrix to store data
    model_rmse = []
    model_val_predictions = np.empty((trainData.shape[0], 1))
    
    # Loop through the index in KFolds
    for train_index, val_index in folds:
        # Split the train data into train and validation data
        train, validation = trainData.iloc[train_index], trainData.iloc[val_index]
        # Get the features and target
        train_features, train_target = train[features], train[target]
        validation_features, validation_target = validation[features], validation[target]
        
        # Fit the base model to the train data and make prediciton for validation data
        clf.fit(train_features, np.ravel(train_target))
        validation_predictions = clf.predict(validation_features)

        # Calculate and store the RMSE for validation data
        model_rmse.append(rmse(validation_target, validation_predictions))
        # Save the validation prediction for level 1 model training
        model_val_predictions[val_index, 0] = validation_predictions.reshape(validation.shape[0])
    
    # Fit the base model to the whole training data
    clf.fit(trainData[features], np.ravel(trainData[target]))
    # Get base model prediction for the test data
    model_test_predictions = clf.predict(testData[features])
    
    return(model_rmse, model_val_predictions, model_test_predictions)


# In[ ]:


# Function that takes a dictionary of classifiers and fit it to the data using baseFit
# The results of the classifiers are then aggregated and returned for level 1 model training
def stacks(level0_classifiers, folds, features, target, trainData, testData):
    num_classifiers = len(level0_classifiers.keys()) #Number of classifiers
    
    # Initialize empty lists and matrix
    level0_trainFeatures = np.empty((trainData.shape[0], num_classifiers))
    level0_testFeatures = np.empty((testData.shape[0], num_classifiers))
    
    # Loop through the classifiers
    for i, key in enumerate(level0_classifiers.keys()):
        print('Fitting %s -----------------------' % (key))
        model_rmse, val_predictions, test_predictions = baseFit(level0_classifiers[key], folds, features, target, trainData, testData)
        
        # Print the average RMSE for the classifier
        print('%s average RMSE: %s' % (key, np.mean(model_rmse)))
        print('\n')
        
        # Aggregate the base model validation and test data predictions
        level0_trainFeatures[:, i] = val_predictions.reshape(trainData.shape[0])
        level0_testFeatures[:, i] = test_predictions.reshape(testData.shape[0])
        
    return(level0_trainFeatures, level0_testFeatures)


# In[ ]:


# Function that takes a dictionary of classifiers and train them on base model predictions
def stackerTraining(stacker, folds, level0_trainFeatures, level0_testFeatures, trainData):
    for k in stacker.keys():
        print('Training stacker %s' % (k))
        stacker_clf = stacker[k]
        stacker_rmse = []
        for t, v in folds:
            stacker_clf.fit(level0_trainFeatures[t], trainData.iloc[t][target])
            stacker_rmse.append(rmse(trainData.iloc[v][target], stacker_clf.predict(level0_trainFeatures[v])))

        print('%s Stacker RMSE: %s' % (k, np.mean(stacker_rmse)))


# In[ ]:


# Get the feature and target anmes
features = train_df.columns.difference(['Id', 'SalePrice', 'logPrice'])
target = ['logPrice']


# In[ ]:


# Get the K fold indexes
n_folds = 5
kf = KFold(train_df.shape[0], n_folds = n_folds, random_state = 1)


# In[ ]:


# A dictionary of base models
level0_classifiers = {}
#level0_classifiers['Lasso'] = Lasso(alpha = 0.001)
level0_classifiers['Ridge'] = Ridge()
#level0_classifiers['KernelRidge'] = KernelRidge()
#level0_classifiers['ElasticNet'] = ElasticNet(alpha = 0.001)
level0_classifiers['XGB'] = xgb.XGBRegressor(max_depth=4,
                                             reg_alpha=0.9,
                                             reg_lambda=0.6,
                                             seed=1)
level0_classifiers['GradientBoostingRegressor'] = GradientBoostingRegressor()
level0_classifiers['RandomForestRegressor'] = RandomForestRegressor(n_estimators = 200, 
                                                                    min_samples_split = 5)


# In[ ]:


# Train all the base models in the dictionary
level0_trainFeatures, level0_testFeatures = stacks(level0_classifiers, kf, features, target, train_df, test_df)


# In[ ]:


# A dictionary of level 1 model to train on base model predictions
stacker = {'Ridge': Ridge(),
           'ElasticNet': ElasticNet(alpha = 0.001)}


# In[ ]:


# Traing the level 1 model to examine their CV performance
stackerTraining(stacker, kf, level0_trainFeatures, level0_testFeatures, train_df)


# In[ ]:


# Use Ridge regression to stack the base models and generate prediction for test data
level1_model = Ridge().fit(level0_trainFeatures, train_df[target])
level1_test_predictions = level1_model.predict(level0_testFeatures)


# In[ ]:


# Transform the data back to original scale
test_predictions = np.exp(level1_test_predictions).reshape(test_df.shape[0])


# In[ ]:


d = {'Id': test_df['Id'], 'SalePrice': test_predictions}
pd.DataFrame(data = d).to_csv('submissions.csv', index = False)

