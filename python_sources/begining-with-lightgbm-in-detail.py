#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


print(os.listdir("../input/"))


# ## Basic LightGBM - First Step

# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import train_test_split
MAX_EVALS = 5


# In[ ]:


features=pd.read_csv('../input/application_train.csv')
features=features.sample(n=16000,random_state=42)
print(features.shape)


# In[ ]:


features.dtypes.value_counts()


# In[ ]:


features = features.select_dtypes('number')
labels = np.array(features['TARGET'])
features = features.drop(columns = ['TARGET', 'SK_ID_CURR'])
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 6000, random_state = 50)


# In[ ]:


print("Training features shape: ", train_features.shape)
print("Testing features shape: ", test_features.shape)


# In[ ]:


train_set = lgb.Dataset(data = train_features, label = train_labels)
test_set = lgb.Dataset(data = test_features, label = test_labels)


# In[ ]:


model = lgb.LGBMClassifier()
default_params = model.get_params()

# Remove the number of estimators because we set this to 10000 in the cv call
del default_params['n_estimators']

# Cross validation with early stopping
cv_results = lgb.cv(default_params, train_set, num_boost_round = 10000, early_stopping_rounds = 100, 
                    metrics = 'auc', nfold = 5, seed = 42)


# In[ ]:


print(max(cv_results['auc-mean']))
print(len(cv_results['auc-mean']))


# In[ ]:


from sklearn.metrics import roc_auc_score


# In[ ]:


model.n_estimators = len(cv_results['auc-mean'])
# Train and make predicions with model
model.fit(train_features, train_labels)
preds = model.predict_proba(test_features)[:, 1]
baseline_auc = roc_auc_score(test_labels, preds)
print('The model scores {:.5f} ROC AUC on the test set.'.format(baseline_auc))


# In[ ]:


def objective(hyperparameters, iteration):
    
    if 'n_estimators' in hyperparameters.keys():
        del hyperparameters['n_estimators']   

    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 10000, nfold =5, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 42)

    score = cv_results['auc-mean'][-1]
    estimators = len(cv_results['auc-mean'])
    hyperparameters['n_estimators'] = estimators     
    return [score, hyperparameters, iteration]


# In[ ]:


score, params, iteration = objective(default_params, 1)

print('The cross-validation ROC AUC was {:.5f}.'.format(score))


# In[ ]:


model = lgb.LGBMModel()
model.get_params()


# In[ ]:


param_grid = {
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'num_leaves': list(range(20, 150)),
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'subsample': list(np.linspace(0.5, 1, 100)),
    'is_unbalance': [True, False]
}


# In[ ]:


import random

random.seed(50)

# Randomly sample a boosting type
boosting_type = random.sample(param_grid['boosting_type'], 1)[0]

# Set subsample depending on boosting type
subsample = 1.0 if boosting_type == 'goss' else random.sample(param_grid['subsample'], 1)[0]

print('Boosting type: ', boosting_type)
print('Subsample ratio: ', subsample)


# In[ ]:


random_results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                              index = list(range(MAX_EVALS)))

grid_results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                              index = list(range(MAX_EVALS)))


# In[ ]:


import itertools

def grid_search(param_grid, max_evals = MAX_EVALS):
    """Grid search algorithm (with limit on max evals)"""
    results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                              index = list(range(MAX_EVALS)))
    keys, values = zip(*param_grid.items())    
    i = 0
    for v in itertools.product(*values):
        hyperparameters = dict(zip(keys, v))
        hyperparameters['subsample'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters['subsample']
        eval_results = objective(hyperparameters, i)       
        results.loc[i, :] = eval_results
        i += 1
        if i > MAX_EVALS:
            break
    results.sort_values('score', ascending = False, inplace = True)
    results.reset_index(inplace = True)    
    return results    


# In[ ]:


grid_results = grid_search(param_grid)
print('The best validation score was {:.5f}'.format(grid_results.loc[0, 'score']))
print('\nThe best hyperparameters were:')
import pprint
pprint.pprint(grid_results.loc[0, 'params'])


# In[ ]:


grid_search_params = grid_results.loc[0, 'params']
model = lgb.LGBMClassifier(**grid_search_params, random_state=42)
model.fit(train_features, train_labels)

preds = model.predict_proba(test_features)[:, 1]

print('The best model from grid search scores {:.5f} ROC AUC on the test set.'.format(roc_auc_score(test_labels, preds)))


# In[ ]:


random.seed(50)

# Randomly sample from dictionary
random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
# Deal with subsample ratio
random_params['subsample'] = 1.0 if random_params['boosting_type'] == 'goss' else random_params['subsample']

random_params


# In[ ]:


def random_search(param_grid, max_evals = MAX_EVALS):
    """Random search for hyperparameter optimization"""
    
    # Dataframe for results
    results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                                  index = list(range(MAX_EVALS)))
    
    # Keep searching until reach max evaluations
    for i in range(MAX_EVALS):
        
        # Choose random hyperparameters
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        hyperparameters['subsample'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters['subsample']

        # Evaluate randomly selected hyperparameters
        eval_results = objective(hyperparameters, i)
        
        results.loc[i, :] = eval_results
    
    # Sort with best score on top
    results.sort_values('score', ascending = False, inplace = True)
    results.reset_index(inplace = True)
    return results 


# In[ ]:


random_results = random_search(param_grid)

print('The best validation score was {:.5f}'.format(random_results.loc[0, 'score']))
print('\nThe best hyperparameters were:')

import pprint
pprint.pprint(random_results.loc[0, 'params'])


# In[ ]:


random_search_params = random_results.loc[0, 'params']

# Create, train, test model
model = lgb.LGBMClassifier(**random_search_params, random_state = 42)
model.fit(train_features, train_labels)

preds = model.predict_proba(test_features)[:, 1]

print('The best model from random search scores {:.5f} ROC AUC on the test set.'.format(roc_auc_score(test_labels, preds)))


# In[ ]:


train = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')

# Extract the test ids and train labels
test_ids = test['SK_ID_CURR']
train_labels = np.array(train['TARGET'].astype(np.int32)).reshape((-1, ))

train = train.drop(columns = ['SK_ID_CURR', 'TARGET'])
test = test.drop(columns = ['SK_ID_CURR'])

print('Training shape: ', train.shape)
print('Testing shape: ', test.shape)


# In[ ]:


le = LabelEncoder()
le_count = 0
for col in train:
    if train[col].dtype == 'object':
        if len(list(train[col].unique())) <= 2:
            le.fit(train[col])
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])
            le_count += 1
print('%d columns were label encoded.' % le_count)


# In[ ]:


train = pd.get_dummies(train)
test=pd.get_dummies(test)
print(train.shape,test.shape)


# In[ ]:


train, test = train.align(test, join = 'inner', axis = 1)

# Add the target back in
#train['TARGET'] = train_labels

print('Training Features shape: ', train.shape)
print('Testing Features shape: ', test.shape)


# In[ ]:


train_set = lgb.Dataset(train, label = train_labels)

hyperparameters = dict(**random_results.loc[0, 'params'])
del hyperparameters['n_estimators']

# Cross validation with n_folds and early stopping
cv_results = lgb.cv(hyperparameters, train_set,
                    num_boost_round = 10000, early_stopping_rounds = 100, 
                    metrics = 'auc', nfold = 5)


# In[ ]:


model = lgb.LGBMClassifier(n_estimators = len(cv_results['auc-mean']), **hyperparameters)
model.fit(train, train_labels)
                        
# Predictions on the test data
preds = model.predict_proba(test)[:, 1]


# In[ ]:


submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': preds})
submission.to_csv('submission_simple_features_random.csv', index = False)


# ## LightGBM - Second Step

# In[ ]:


app_train = pd.read_csv('../input/application_train.csv')
print('Training data shape: ', app_train.shape)
app_train.head()


# In[ ]:


app_test = pd.read_csv('../input/application_test.csv')
print('Testing data shape: ', app_test.shape)
app_test.head()


# In[ ]:


app_train['TARGET'].value_counts()


# In[ ]:


plt.figure(figsize=(10,5))
sns.set(style="whitegrid",font_scale=1)
g=sns.distplot(app_train['TARGET'],kde=False,hist_kws={"alpha": 1, "color": "#DA1A32"})
plt.title('Distribution of target (1:default, 0:no default)',size=15)
plt.show()


# ## About missing values

# In[ ]:


def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns


# In[ ]:


missing_values = missing_values_table(app_train)
missing_values.head(10)


# ## Column Types

# In[ ]:


# Number of each type of column
app_train.dtypes.value_counts()


# In[ ]:


# Number of unique classes in each object column
app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# In[ ]:


le = LabelEncoder()
le_count = 0
for col in app_train:
    if app_train[col].dtype == 'object':
        if len(list(app_train[col].unique())) <= 2:
            le.fit(app_train[col])
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            le_count += 1
print('%d columns were label encoded.' % le_count)


# In[ ]:


app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)
print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)


# In[ ]:


train_labels_last=app_train['TARGET']


# In[ ]:


train_labels = app_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

# Add the target back in
app_train['TARGET'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)


# In[ ]:


(app_train['DAYS_BIRTH'] / -365).describe()


# In[ ]:


app_train['DAYS_EMPLOYED'].describe()


# In[ ]:


plt.hist(app_train['DAYS_EMPLOYED'],color="#DA1A32")
plt.title('Ditribution of employed days')
plt.show()


# **350000 days of employment seems wrong.**

# In[ ]:


anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]
print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(anom))


# **Replace it with nan.**

# In[ ]:


# Create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
app_train['DAYS_EMPLOYED'].plot.hist(title = 'Ditribution of employed days',color="#DA1A32")
plt.xlabel('Days')


# In[ ]:


app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
print('There are %d anomalies in the test data out of %d entries' % (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))


# ## Correlations

# In[ ]:


# Find correlations with the target and sort
correlations = app_train.corr()['TARGET'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))


# In[ ]:


app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])


# In[ ]:


# Extract the EXT_SOURCE variables and show correlations
ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs = ext_data.corr()
ext_data_corrs


# In[ ]:


plt.figure(figsize = (8, 6))
# Heatmap of correlations
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');


# ## Feature engineering

# <font size="4">Manual features. </font>

# In[ ]:


manual_features=app_train[['SK_ID_CURR','AMT_CREDIT','AMT_ANNUITY','AMT_INCOME_TOTAL',
                           'DAYS_BIRTH','DAYS_EMPLOYED']]
manual_features_test=app_test[['SK_ID_CURR','AMT_CREDIT','AMT_ANNUITY','AMT_INCOME_TOTAL',
                           'DAYS_BIRTH','DAYS_EMPLOYED']]


# In[ ]:


cols=list(manual_features.columns)
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')
manual_features = imputer.fit_transform(manual_features)
manual_features_test = imputer.transform(manual_features_test)


# In[ ]:


manual_features=pd.DataFrame(manual_features,columns=cols)
manual_features_test=pd.DataFrame(manual_features_test,columns=cols)


# In[ ]:


manual_features['CREDIT_INCOME_PERCENT'] = manual_features['AMT_CREDIT'] /manual_features['AMT_INCOME_TOTAL']
manual_features['ANNUITY_INCOME_PERCENT'] = manual_features['AMT_ANNUITY'] / manual_features['AMT_INCOME_TOTAL']
manual_features['CREDIT_TERM'] = manual_features['AMT_ANNUITY'] / manual_features['AMT_CREDIT']
manual_features['DAYS_EMPLOYED_PERCENT'] = manual_features['DAYS_EMPLOYED'] / manual_features['DAYS_BIRTH']


# In[ ]:


manual_features_test['CREDIT_INCOME_PERCENT'] = manual_features_test['AMT_CREDIT'] / manual_features_test['AMT_INCOME_TOTAL']
manual_features_test['ANNUITY_INCOME_PERCENT'] = manual_features_test['AMT_ANNUITY'] / manual_features_test['AMT_INCOME_TOTAL']
manual_features_test['CREDIT_TERM'] = manual_features_test['AMT_ANNUITY'] / manual_features_test['AMT_CREDIT']
manual_features_test['DAYS_EMPLOYED_PERCENT'] = manual_features_test['DAYS_EMPLOYED'] / manual_features_test['DAYS_BIRTH']


# <font size="4">Polynomial features & Imputation of missing values. </font>

# In[ ]:


poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')

poly_target = poly_features['TARGET']
poly_features = poly_features.drop(columns = ['TARGET'])

poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)

from sklearn.preprocessing import PolynomialFeatures
                                
poly_transformer = PolynomialFeatures(degree = 3)


# In[ ]:


poly_transformer.fit(poly_features)

poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features.shape)


# In[ ]:


poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])


# In[ ]:


poly_features = pd.DataFrame(poly_features, 
                             columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                           'EXT_SOURCE_3', 'DAYS_BIRTH']))

poly_features['TARGET'] = poly_target

poly_corrs = poly_features.corr()['TARGET'].sort_values()

print(poly_corrs.head(10))
print(poly_corrs.tail(5))


# In[ ]:


poly_features_test = pd.DataFrame(poly_features_test, 
                                  columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                                'EXT_SOURCE_3', 'DAYS_BIRTH']))


poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']
app_train_poly_1 = app_train.merge(poly_features, on = 'SK_ID_CURR', how = 'left')
app_train_poly_3=app_train_poly_1.merge(manual_features,on = 'SK_ID_CURR', how = 'left')

poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
app_test_poly_2 = app_test.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')
app_test_poly_4=app_test_poly_2.merge(manual_features_test,on = 'SK_ID_CURR', how = 'left')

app_train_poly, app_test_poly = app_train_poly_3.align(app_test_poly_4, join = 'inner', axis = 1)

print('Training data with polynomial features shape: ', app_train_poly.shape)
print('Testing data with polynomial features shape:  ', app_test_poly.shape)


# In[ ]:


app_train_poly=app_train_poly_3.T.drop_duplicates().T


# In[ ]:


app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)


# In[ ]:


app_train_poly.head()


# In[ ]:


app_train_poly.isnull().sum()


# In[ ]:


col_2=list(app_train_poly.columns)


# In[ ]:


print(app_train_poly.shape,app_test_poly.shape)


# In[ ]:


from sklearn.preprocessing import Imputer

if 'TARGET' in app_train_poly:
    train = app_train_poly.drop(columns = ['TARGET'])
else:
    train = app_train_poly.copy()

features = list(train.columns)

test = app_test_poly.copy()

imputer = Imputer(strategy = 'median')

imputer.fit(train)

train = imputer.transform(train)
test = imputer.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)


# In[ ]:


train=pd.DataFrame(train,columns=col_2)
test=pd.DataFrame(test,columns=col_2)


# In[ ]:


train['TARGET']=app_train['TARGET']


# In[ ]:


train.head()


# In[ ]:


#train_last=train.copy()
#test_last=test.copy()


# ## LightGBM

# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import train_test_split
MAX_EVALS = 5


# In[ ]:


features=train.sample(n=16000,random_state=42)


# In[ ]:


print(features.shape)


# In[ ]:


labels = np.array(features['TARGET'])
features = features.drop(columns = ['TARGET', 'SK_ID_CURR'])
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 6000, random_state = 50)


# In[ ]:


print("Training features shape: ", train_features.shape)
print("Testing features shape: ", test_features.shape)


# In[ ]:


train_set = lgb.Dataset(data = train_features, label = train_labels)
test_set = lgb.Dataset(data = test_features, label = test_labels)


# In[ ]:


model = lgb.LGBMClassifier()
default_params = model.get_params()

# Remove the number of estimators because we set this to 10000 in the cv call
del default_params['n_estimators']

# Cross validation with early stopping
cv_results = lgb.cv(default_params, train_set, num_boost_round = 10000, early_stopping_rounds = 100, 
                    metrics = 'auc', nfold = 5, seed = 42)


# In[ ]:


print(max(cv_results['auc-mean']))


# In[ ]:


print(len(cv_results['auc-mean']))


# In[ ]:


from sklearn.metrics import roc_auc_score


# In[ ]:


model.n_estimators = len(cv_results['auc-mean'])
# Train and make predicions with model
model.fit(train_features, train_labels)
preds = model.predict_proba(test_features)[:, 1]
baseline_auc = roc_auc_score(test_labels, preds)
print('The baseline model scores {:.5f} ROC AUC on the test set.'.format(baseline_auc))


# In[ ]:


def objective(hyperparameters, iteration):
    """Objective function for grid and random search. Returns
       the cross validation score from a set of hyperparameters."""
    
    # Number of estimators will be found using early stopping
    if 'n_estimators' in hyperparameters.keys():
        del hyperparameters['n_estimators']   
     # Perform n_folds cross validation
    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 10000, nfold =5, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 42)
    # results to retun
    score = cv_results['auc-mean'][-1]
    estimators = len(cv_results['auc-mean'])
    hyperparameters['n_estimators'] = estimators     
    return [score, hyperparameters, iteration]


# In[ ]:


score, params, iteration = objective(default_params, 1)

print('The cross-validation ROC AUC was {:.5f}.'.format(score))


# In[ ]:


model = lgb.LGBMModel()
model.get_params()


# ## Grid search

# In[ ]:


param_grid = {
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'num_leaves': list(range(20, 150)),
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'subsample': list(np.linspace(0.5, 1, 100)),
    'is_unbalance': [True, False]
}


# In[ ]:


import random

random.seed(50)

# Randomly sample a boosting type
boosting_type = random.sample(param_grid['boosting_type'], 1)[0]

# Set subsample depending on boosting type
subsample = 1.0 if boosting_type == 'goss' else random.sample(param_grid['subsample'], 1)[0]

print('Boosting type: ', boosting_type)
print('Subsample ratio: ', subsample)


# In[ ]:


random_results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                              index = list(range(MAX_EVALS)))

grid_results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                              index = list(range(MAX_EVALS)))


# In[ ]:


import itertools

def grid_search(param_grid, max_evals = MAX_EVALS):
    """Grid search algorithm (with limit on max evals)"""
    results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                              index = list(range(MAX_EVALS)))
    keys, values = zip(*param_grid.items())    
    i = 0
    for v in itertools.product(*values):
        hyperparameters = dict(zip(keys, v))
        hyperparameters['subsample'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters['subsample']
        eval_results = objective(hyperparameters, i)       
        results.loc[i, :] = eval_results
        i += 1
        if i > MAX_EVALS:
            break
    results.sort_values('score', ascending = False, inplace = True)
    results.reset_index(inplace = True)    
    return results    


# In[ ]:


grid_results = grid_search(param_grid)
print('The best validation score was {:.5f}'.format(grid_results.loc[0, 'score']))
print('\nThe best hyperparameters were:')
import pprint
pprint.pprint(grid_results.loc[0, 'params'])


# In[ ]:


grid_search_params = grid_results.loc[0, 'params']
model = lgb.LGBMClassifier(**grid_search_params, random_state=42)
model.fit(train_features, train_labels)

preds = model.predict_proba(test_features)[:, 1]

print('The best model from grid search scores {:.5f} ROC AUC on the test set.'.format(roc_auc_score(test_labels, preds)))


# ## Random search

# In[ ]:


random.seed(50)
random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
random_params['subsample'] = 1.0 if random_params['boosting_type'] == 'goss' else random_params['subsample']


# In[ ]:


def random_search(param_grid, max_evals = MAX_EVALS):
   
    results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                                 index = list(range(MAX_EVALS)))
    for i in range(MAX_EVALS):
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        hyperparameters['subsample'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters['subsample']
        eval_results = objective(hyperparameters, i)
        results.loc[i, :] = eval_results
        
    results.sort_values('score', ascending = False, inplace = True)
    results.reset_index(inplace = True)
    return results 


# In[ ]:


random_results = random_search(param_grid)

print('The best validation score was {:.5f}'.format(random_results.loc[0, 'score']))
print('\nThe best hyperparameters were:')

import pprint
pprint.pprint(random_results.loc[0, 'params'])


# In[ ]:


random_search_params = random_results.loc[0, 'params']

model = lgb.LGBMClassifier(**random_search_params, random_state = 42)
model.fit(train_features, train_labels)

preds = model.predict_proba(test_features)[:, 1]

print('The best model from random search scores {:.5f} ROC AUC on the test set.'.format(roc_auc_score(test_labels, preds)))


# In[ ]:


#train = pd.read_csv('../input/application_train.csv')
#test = pd.read_csv('../input/application_test.csv')

# Extract the test ids and train labels
#test_ids = test['SK_ID_CURR']
#train_labels = np.array(train['TARGET'].astype(np.int32)).reshape((-1, ))

#train = train.drop(columns = ['SK_ID_CURR', 'TARGET'])
#test = test.drop(columns = ['SK_ID_CURR'])

#print('Training shape: ', train.shape)
#print('Testing shape: ', test.shape)


# In[ ]:


#le = LabelEncoder()
#le_count = 0
#for col in train:
#    if train[col].dtype == 'object':
#        if len(list(train[col].unique())) <= 2:
#            le.fit(train[col])
#            train[col] = le.transform(train[col])
#            test[col] = le.transform(test[col])
#            le_count += 1
#print('%d columns were label encoded.' % le_count)


# In[ ]:


#train = pd.get_dummies(train)
#test=pd.get_dummies(test)
#print(train.shape,test.shape)


# In[ ]:



# Align the training and testing data, keep only columns present in both dataframes
#train, test = train.align(test, join = 'inner', axis = 1)

# Add the target back in
#train['TARGET'] = train_labels

#print('Training Features shape: ', train.shape)
#print('Testing Features shape: ', test.shape)


# In[ ]:


#train=train.select_dtypes('number')
#test=test.select_dtypes('number')


# In[ ]:


train_set = lgb.Dataset(train, label = train_labels_last)

hyperparameters = dict(**random_results.loc[0, 'params'])
del hyperparameters['n_estimators']

# Cross validation with n_folds and early stopping
cv_results = lgb.cv(hyperparameters, train_set,
                    num_boost_round = 10000, early_stopping_rounds = 100, 
                    metrics = 'auc', nfold = 5)
print(max(cv_results['auc-mean']))


# In[ ]:


test['SK_ID_CURR'].describe()


# In[ ]:


test_ids = test['SK_ID_CURR']


# In[ ]:


train=train.drop(columns=['TARGET','SK_ID_CURR'])
test=test.drop(columns='SK_ID_CURR')


# In[ ]:


print(train.shape,test.shape)


# In[ ]:


model = lgb.LGBMClassifier(n_estimators = len(cv_results['auc-mean']), **hyperparameters)
model.fit(train, train_labels_last)
                        
# Predictions on the test data
preds = model.predict_proba(test)[:, 1]
#auc1=roc_auc_score(y_test, preds)
#print(auc1)


# In[ ]:


#submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': preds})
#submission['SK_ID_CURR']=submission['SK_ID_CURR'].astype('int32')
#submission.to_csv('submission_simple_features_random.csv', index = False)

