#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import h5py

import numpy as np
import pandas as pd

from glob import glob

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Explore all csv

# In[ ]:


# Age and 4 anonymized targets, 443 partially missed observations
train_scores = pd.read_csv('/kaggle/input/trends-assessment-prediction/train_scores.csv').sort_values(by='Id')

# Somehow preprocessed morphometry (after group ICA), simplest feature set
loadings = pd.read_csv('/kaggle/input/trends-assessment-prediction/loading.csv')

# resting-state fMRI Functional Network Connectivity matrices. 
# In simple setting, these are cross-correlations (in this case something more sophisticated) between
# every pair of brain regions presented in train/test *.mat
fnc = pd.read_csv('/kaggle/input/trends-assessment-prediction/fnc.csv')

# Submit Age and 4 scores for test ids
sample = pd.read_csv('/kaggle/input/trends-assessment-prediction/sample_submission.csv')

# List of some of subjects from test set whose data were collected from different scanner
reveal = pd.read_csv('/kaggle/input/trends-assessment-prediction/reveal_ID_site2.csv')

# 53 unique numbers between 2 and 99 (somehow related to brain regions? regions keys?)
icn_nums = pd.read_csv('/kaggle/input/trends-assessment-prediction/ICN_numbers.csv')

# Brain template
# /kaggle/input/trends-assessment-prediction/fMRI_mask.nii 

# train/test fMRI spatial maps
# *.mat


# In[ ]:


loadings.head(3)


# In[ ]:


# 53 * 52 / 2 = 1378 + Id column
fnc.head(3)


# # Exploring fnc

# In[ ]:


import re
from tqdm import tqdm


# In[ ]:


r = re.compile('\d+')
col_dict = {}
for col in fnc.columns:
    ind = r.findall(col)
    if ind:
        col_dict[col] = [int(i) for i in ind]


# In[ ]:


def get_matrix(df_row, return_idx=False):
    matrix = np.zeros((100, 100))
    for col in df_row.index[1:]:
        i, j = col_dict[col]
        matrix[i, j] = df_row[col]
    matrix += matrix.T
    
    idx = np.array([ 2,  3,  4,  5,  7,  8,  9, 11, 12, 13, 15, 16, 17, 18, 20, 21, 23,
                     27, 32, 33, 37, 38, 40, 43, 45, 48, 51, 53, 54, 55, 56, 61, 62, 63,
                     66, 67, 68, 69, 70, 71, 72, 77, 79, 80, 81, 83, 84, 88, 93, 94, 96,
                     98, 99])
    if return_idx:
        return matrix[:, idx][idx, :], idx 
    return matrix[:, idx][idx, :]


# In[ ]:


degrees = []
for row in tqdm(fnc.iterrows()):
    mat = get_matrix(row[1])
    degrees.append(mat.sum(axis=1))


# In[ ]:


_, idx = get_matrix(fnc.iloc[0], return_idx=True)
degrees = pd.DataFrame(degrees, columns=idx)
degrees['Id'] = fnc['Id']


# In[ ]:


len(glob('/kaggle/input/trends-assessment-prediction/fMRI_train/*.mat')), len(glob('/kaggle/input/trends-assessment-prediction/fMRI_test/*.mat'))


# In[ ]:


sbj = glob('/kaggle/input/trends-assessment-prediction/fMRI_train/*.mat')[10]

with h5py.File(sbj, 'r') as f:
    mat = f['SM_feature'][()]
    mat = np.moveaxis(mat, [0,1,2,3], [3,2,1,0])
    
print(mat.shape)


# # Build a model
# 
# upd1. add mape_scorer
# 
# upd2. add fnc to features
# 
# upd3. add degrees features
# 
# upd4. add fcn/500 from https://www.kaggle.com/aerdem4/rapids-svm-on-trends-neuroimaging , switch to Ridge regression
# 
# todo: fit on all available targets (currently observation is dropped if any target is missing)

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict


# In[ ]:


# train/test Ids
train_ids = sorted(loadings[loadings['Id'].isin(train_scores.Id)]['Id'].values)
test_ids = sorted(loadings[~loadings['Id'].isin(train_scores.Id)]['Id'].values)

# generate test DataFrame
test_prediction = pd.DataFrame(test_ids, columns=['Id'], dtype=str)

target_columns = ('age', 'domain1_var1', 'domain1_var2','domain2_var1','domain2_var2')
fnc_columns = fnc.columns[1:]
degrees_columns = degrees.columns[:-1]

# generate X, targets
data = pd.merge(loadings, train_scores, on='Id')#.dropna()
data = pd.merge(data, fnc, on='Id')
data = pd.merge(data, degrees, on='Id')

# X_train = data.drop(list(target_columns), axis=1).drop('Id', axis=1)
# y_train = data[list(target_columns)]

X_test = pd.merge(loadings[loadings.Id.isin(test_ids)], fnc, on='Id')
X_test = pd.merge(X_test, degrees, on='Id').drop('Id', axis=1)


# ## Implement mape scorer
# 
# Since lb uses weighted mape (all targets are > 0), we will implement mape scorer to pass into GridSearchCV

# In[ ]:


from sklearn.metrics import make_scorer

def MAPE(y_true, y_pred, **kwargs):
    '''Returns MAPE between y_true and y_pred'''
    return np.sum(np.abs(y_true - y_pred)) / y_true.sum()

mape_scorer = make_scorer(MAPE, greater_is_better=False)


# In[ ]:


# Setting up the model
# model = RandomForestRegressor(
#     max_depth=5,
#     min_samples_split=10,
#     min_samples_leaf=5
# )

# model = Lasso()
model = Ridge()
# model = SVR()


cv = KFold(n_splits = 5, shuffle=True, random_state=29)

# grid = {
#     'max_depth':[2, 5, 10],
#     'n_estimators':[20, 30],
#     'max_features':[0.1, 0.2, 0.3, 0.5]
# }

grid = {
    'alpha': [0.0003, 0.001, 0.003, 0.01, 0.03]
}
# grid = {
#     'C': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.85, 1, 3, 5, 10]
# }

# grid = {
#     'alpha': np.linspace(0.0001, 0.001, 20)
# }
gs = GridSearchCV(model, grid, n_jobs=-1, cv=cv, verbose=0, scoring=mape_scorer)


# In[ ]:


# age 0.1446

# domain1_var1 0.1512

# domain1_var2 0.1513

# domain2_var1 0.1819

# domain2_var2 0.1763


# In[ ]:


# Training the model
best_models = {}
total_score = []

for col in target_columns:
    
    X_train = data.dropna(subset=[col], axis=0).drop(list(target_columns), axis=1).drop('Id', axis=1)
    X_train[fnc_columns] /= 500
    y_train = data.dropna(subset=[col], axis=0)[col]
    
    gs.fit(X_train, y_train)
    best_models[col] = gs.best_estimator_
    
    # Train performance
    y_pred = cross_val_predict(gs.best_estimator_, X_train, y_train, cv=cv, n_jobs=-1)
    total_score.append(MAPE(y_train, y_pred))
    print(col, MAPE(y_train, y_pred))

total_score = np.array(total_score)
print(f'Total score: {np.sum(total_score*[.3, .175, .175, .175, .175])}')


# In[ ]:


def get_pred(col, model):
    X_train = data.dropna(subset=[col], axis=0).drop(list(target_columns), axis=1).drop('Id', axis=1)
    X_train[fnc_columns] /= 500
    y_train = data.dropna(subset=[col], axis=0)[col]
    
    # Train performance
    y_pred = cross_val_predict(model, X_train, y_train, cv=cv, n_jobs=-1)
    return y_pred


# In[ ]:


# Predicting test
X_test[fnc_columns] /= 500

for col in target_columns:
    test_prediction[col] = best_models[col].predict(X_test)


# In[ ]:


# Evaluate the lb metric on local cv

# def lb_metric(y_true, y_pred):
#     '''Computes lb metric, both y_true and y_pred should be DataFrames of shape n x 5'''
#     y_true = y_true[['age', 'domain1_var1', 'domain1_var2','domain2_var1','domain2_var2']]
#     y_pred = y_pred[['age', 'domain1_var1', 'domain1_var2','domain2_var1','domain2_var2']]
    
#     weights = np.array([.3, .175, .175, .175, .175])
#     return np.sum(weights * np.abs(y_pred.values - y_true.values).sum(axis=0) / y_train.values.sum(axis=0))


# In[ ]:


# train_prediction_cv = {}
# for col in target_columns:
#     train_prediction_cv[col] = cross_val_predict(best_models[col], X_train, y_train[col], cv = cv, n_jobs=-1)
# train_prediction_cv = pd.DataFrame(train_prediction_cv)

# lb_metric(y_train, train_prediction_cv)


# ## Making a submission

# In[ ]:


def make_sub(test_prediction):
    '''Converts 5877 x 6 DataFrame of predictions into 29385 x 2 DataFrame with valid Id'''
    target_columns = ('age', 'domain1_var1', 'domain1_var2','domain2_var1','domain2_var2')
    _columns = (0,1,2,3,4)
    tst = test_prediction.rename(columns=dict(zip(target_columns, _columns)))
    tst = tst.melt(id_vars='Id',
           value_vars=_columns,
           value_name='Predicted')

    tst['target_type'] = tst.variable.map(dict(zip(_columns, target_columns)))
    tst['Id_'] = tst[['Id', 'target_type']].apply(lambda x: '_'.join((str(x[0]), str(x[1]))), axis=1)

    return tst.sort_values(by=['Id', 'variable'])              .drop(['Id', 'variable', 'target_type'],axis=1)              .rename(columns={'Id_':'Id'})              .reset_index(drop=True)              [['Id', 'Predicted']]


# In[ ]:


sub = make_sub(test_prediction)

sub.head()


# In[ ]:


sub.to_csv('ridge_mape_500.csv', index=False)


# In[ ]:




