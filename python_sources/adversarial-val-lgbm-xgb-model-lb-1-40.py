#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import scipy
import gc
import progressbar

from scipy.stats import ks_2samp, skew, kurtosis
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.metrics import mean_squared_error, roc_auc_score

from sklearn.decomposition import PCA, TruncatedSVD, FastICA, FactorAnalysis, KernelPCA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.manifold import TSNE
from sklearn.grid_search import GridSearchCV

import lightgbm as lgb
import xgboost as xgb
import catboost as cb


# ### **Read in the data**

# In[ ]:


def read_data():
    print("############# Read the data #############")
    
    train = pd.read_csv("../input/train.csv", index_col = 0)
    test = pd.read_csv("../input/test.csv", index_col = 0)
    print("\nTrain shape: {}\nTest shape: {}".format(train.shape, test.shape))
    
    return train, test


# ### **Convert and drop target column**

# In[ ]:


def convert_and_drop_target(train, test):
    print("############# Convert and drop target column #############")
    
    y_train = train.target
    y_train = np.log1p(y_train)
    test_ID = test.index

    train = train.drop(['target'], 1)
    
    return train, test, y_train, test_ID


# ### **Removing duplicate columns**

# In[ ]:


def remove_duplicate_columns(train, test):
    print("############# Remove duplicate columns #############")
    
    train = train.T.drop_duplicates().T
    columns_not_to_be_dropped = train.columns
    columns_to_be_dropped = [col for col in test.columns if col not in columns_not_to_be_dropped]
    print("Number of columns removed - " + str(len(columns_to_be_dropped)))
    
    test = test.drop(columns_to_be_dropped, 1)
    print("\nTrain shape: {}\nTest shape: {}".format(train.shape, test.shape))
    
    return train, test


# ### **Removing columns with constant values**

# In[ ]:


def remove_constant_columns(train, test):
    print("############# Remove constant columns #############")
    
    col_with_std_zero = train.loc[:, train.std(axis = 0) == 0].columns
    print("Number of columns removed - " + str(len(col_with_std_zero)))
    
    train = train.loc[:, train.std(axis = 0) != 0]
    test = test.drop(col_with_std_zero, 1)
    print("\nTrain shape: {}\nTest shape: {}".format(train.shape, test.shape))
    
    return train, test


# ### **Removing columns having low importance**

# In[ ]:


def remove_features_using_importance(x_train, y_train, x_test):
    print("############# Remove columns with low importance #############")
    
    def rmsle(actual, predicted):
        return np.sqrt(np.mean(np.power(np.log1p(actual)-np.log1p(predicted), 2)))
    
    num_of_features = 1000
    
    print("Split train and test")
    x1, x2, y1, y2 = train_test_split(x_train, y_train, test_size = 0.20, random_state = 42)
    model = RandomForestRegressor(n_jobs = -1, random_state = 7)
    model.fit(x1, y1)
    print(rmsle(np.expm1(y2), np.expm1(model.predict(x2))))
    
    print("Get columns by feature importances")
    col_df = pd.DataFrame({'importance': model.feature_importances_, 'feature': x_train.columns})
    col_df_sorted = col_df.sort_values(by = ['importance'], ascending = [False])
    columns = col_df_sorted[:num_of_features]['feature'].values
    print("Number of columns removed - " + str(len(x_train.columns) - len(columns)))
    
    x_train = x_train[columns]
    x_test = x_test[columns]
    print("\nTrain shape: {}\nTest shape: {}".format(x_train.shape, x_test.shape))
    
    return x_train, x_test


# ### **Removing columns having different distributions in train and test set**

# In[ ]:


def remove_features_having_different_distributions(train, test):
    print("############# Remove columns having different distributions in train and test set #############")
    
    threshold_p_value = 0.01 
    threshold_statistic = 0.3
    
    cols_with_different_distributions = []
    for col in train.columns:
        statistic, pvalue = ks_2samp(train[col].values, test[col].values)
        if pvalue <= threshold_p_value and np.abs(statistic) > threshold_statistic:
            cols_with_different_distributions.append(col)
    
    print("Number of columns removed - " + str(len(cols_with_different_distributions)))
    for col in cols_with_different_distributions:
        if col in train.columns:
            train = train.drop(col, axis = 1)
            test = test.drop(col, axis = 1)
    print("\nTrain shape: {}\nTest shape: {}".format(train.shape, test.shape))
    
    
    return train, test


# ### **Adding aggregate features**

# In[ ]:


def add_aggregate_features(train, test):
    print("############# Add aggregate features #############")
    
    weight = ((train != 0).sum()/len(train)).values
    
    tmp_train = train[train != 0]
    tmp_test = test[test != 0]

    print("Adding row features.....")
    
    print("weight count")
    train["weight_count"] = (tmp_train * weight).sum(axis = 1)
    test["weight_count"] = (tmp_test * weight).sum(axis = 1)
    
    print("number of non-zero values")
    train["count_non_0"] = (train != 0).sum(axis = 1)
    test["count_non_0"] = (test != 0).sum(axis = 1)
    
    print("number of different")
    train["num_different"] = tmp_train.nunique(axis = 1)
    test["num_different"] = tmp_test.nunique(axis = 1)
    
    print("sum")
    train["sum"] = train.sum(axis=1)
    test["sum"] = test.sum(axis=1)

    print("variance")
    train["var"] = tmp_train.var(axis=1)
    test["var"] = tmp_test.var(axis=1)

    print("mean")
    train["mean"] = tmp_train.mean(axis=1)
    test["mean"] = tmp_test.mean(axis=1)
    
    print("median")
    train["median"] = tmp_train.median(axis=1)
    test["median"] = tmp_test.median(axis=1)

    print("std")
    train["std"] = tmp_train.std(axis=1)
    test["std"] = tmp_test.std(axis=1)

    print("max")
    train["max"] = tmp_train.max(axis=1)
    test["max"] = tmp_test.max(axis=1)

    print("min")
    train["min"] = tmp_train.min(axis=1)
    test["min"] = tmp_test.min(axis=1)
    
    print("skew")
    train["skew"] = tmp_train.apply(skew, axis=1)
    test["skew"] = tmp_test.apply(skew, axis=1)
    
    print("kurtosis")
    train["kurtosis"] = tmp_train.apply(kurtosis, axis=1)
    test["kurtosis"] = tmp_test.apply(kurtosis, axis=1)
    
    print("\nTrain shape: {}\nTest shape: {}".format(train.shape, test.shape))
    
    # Remove an NA valuess
    train = train.fillna(0)
    test = test.fillna(0)
    
    del(tmp_train)
    del(tmp_test)
    return train, test


# ### **Mean-variance scale columns**

# In[ ]:


def mean_variance_scale_columns(total):
    print("############# Mean-variance scale columns #############")
    
    p = progressbar.ProgressBar()
    p.start()

    # Mean-variance scale all columns excluding 0-values' 
    number_of_columns = len(total.columns)
    for col_index, col in enumerate(total.columns):    
        p.update(col_index/number_of_columns * 100)

        # Detect outliers in this column
        data = total[col].values
        data_mean, data_std = np.mean(data), np.std(data)
        cut_off = data_std * 3
        lower, upper = data_mean - cut_off, data_mean + cut_off
        outliers = [x for x in data if x < lower or x > upper]

        # If there are crazy high values, do a log-transform
        if len(outliers) > 0:
            non_zero_idx = data != 0
            total.loc[non_zero_idx, col] = np.log(data[non_zero_idx])

        # Scale non-zero column values
        nonzero_rows = (total[col] != 0)
        if  np.isfinite(total.loc[nonzero_rows, col]).all():
            total.loc[nonzero_rows, col] = scale(list(total.loc[nonzero_rows, col]))
            if  np.isfinite(total[col]).all():
                # Scale all column values
                total[col] = scale(list(total[col]))
        gc.collect()

    p.finish()
    
    return total


# ### **Add dimensional reduction features back to dataset**

# In[ ]:


def add_decomposed_features_back_to_df(train, 
                                       test,
                                       total,
                                       n_components,
                                       use_pca = False,
                                       use_tsne = False,
                                       use_tsvd = False,
                                       use_ica = False,
                                       use_fa = False,
                                       use_grp = False,
                                       use_srp = False):
    print("############# Add decomposed components back to dataframe #############")
    
    N_COMP = n_components
    ntrain = len(train)
    sparse_matrix = scipy.sparse.csr_matrix(total.values)

    print("\nStart decomposition process...")
    
    if use_pca:
        print("PCA")
        pca = PCA(n_components = N_COMP, random_state = 42)
        pca_results = pca.fit_transform(total)
        pca_results_train = pca_results[:ntrain]
        pca_results_test = pca_results[ntrain:]
        
    if use_tsne:  
        print("TSNE")
        tsne = TSNE(n_components = 3, init = 'pca')
        tsne_results = tsne.fit_transform(total)
        tsne_results_train = tsne_results[:ntrain]
        tsne_results_test = tsne_results[ntrain:]

    if use_tsvd:
        print("tSVD")
        tsvd = TruncatedSVD(n_components = N_COMP, random_state = 42)
        tsvd_results = tsvd.fit_transform(sparse_matrix)
        tsvd_results_train = tsvd_results[:ntrain]
        tsvd_results_test = tsvd_results[ntrain:]

    if use_ica:
        print("ICA")
        ica = FastICA(n_components = N_COMP, random_state=42)
        ica_results = ica.fit_transform(total)
        ica_results_train = ica_results[:ntrain]
        ica_results_test = ica_results[ntrain:]

    if use_fa:
        print("FA")
        fa = FactorAnalysis(n_components = N_COMP, random_state=42)
        fa_results = fa.fit_transform(total)
        fa_results_train = fa_results[:ntrain]
        fa_results_test = fa_results[ntrain:]

    if use_grp:
        print("GRP")
        grp = GaussianRandomProjection(n_components = N_COMP, eps = 0.1, random_state = 42)
        grp_results = grp.fit_transform(total)
        grp_results_train = grp_results[:ntrain]
        grp_results_test = grp_results[ntrain:]

    if use_srp:
        print("SRP")
        srp = SparseRandomProjection(n_components = N_COMP, dense_output=True, random_state=42)
        srp_results = srp.fit_transform(total)
        srp_results_train = srp_results[:ntrain]
        srp_results_test = srp_results[ntrain:]

    print("Append decomposition components to datasets...")
    for i in range(1, N_COMP + 1):
        
        if use_pca:
            train['pca_' + str(i)] = pca_results_train[:, i - 1]
            test['pca_' + str(i)] = pca_results_test[:, i - 1]
            
        if use_tsne:
            train['tsne_' + str(i)] = tsne_results_train[:, i - 1]
            test['tsne_' + str(i)] = tsne_results_test[:, i - 1]
            
        if use_tsvd:
            train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
            test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]
        
        if use_ica:
            train['ica_' + str(i)] = ica_results_train[:, i - 1]
            test['ica_' + str(i)] = ica_results_test[:, i - 1]
        
        if use_fa:
            train['fa_' + str(i)] = fa_results_train[:, i - 1]
            test['fa_' + str(i)] = fa_results_test[:, i - 1]

        if use_grp:
            train['grp_' + str(i)] = grp_results_train[:, i - 1]
            test['grp_' + str(i)] = grp_results_test[:, i - 1]
        
        if use_srp:
            train['srp_' + str(i)] = srp_results_train[:, i - 1]
            test['srp_' + str(i)] = srp_results_test[:, i - 1]
    print("\nTrain shape: {}\nTest shape: {}".format(train.shape, test.shape))
    
    return train, test


# ### **Building a new dataframe with the dimensional reduction rechniques**

# In[ ]:


def use_decomposed_features_as_new_df(train, 
                                      test,
                                      total,
                                      n_components,
                                      use_pca = False,
                                      use_tsvd = False,
                                      use_ica = False,
                                      use_fa = False,
                                      use_grp = False,
                                      use_srp = False):
    print("############# Create new dataframe from decomposed components #############")
    
    N_COMP = n_components
    ntrain = len(train)

    print("\nStart decomposition process...")
    
    if use_pca:
        print("PCA")
        pca = PCA(n_components = N_COMP, random_state = 42)
        pca_results = pca.fit_transform(total)
        pca_results_train = pca_results[:ntrain]
        pca_results_test = pca_results[ntrain:]

    if use_tsvd:
        print("tSVD")
        tsvd = TruncatedSVD(n_components = N_COMP, random_state=42)
        tsvd_results = tsvd.fit_transform(total)
        tsvd_results_train = tsvd_results[:ntrain]
        tsvd_results_test = tsvd_results[ntrain:]

    if use_ica:
        print("ICA")
        ica = FastICA(n_components = N_COMP, random_state=42)
        ica_results = ica.fit_transform(total)
        ica_results_train = ica_results[:ntrain]
        ica_results_test = ica_results[ntrain:]

    if use_fa:
        print("FA")
        fa = FactorAnalysis(n_components = N_COMP, random_state=42)
        fa_results = fa.fit_transform(total)
        fa_results_train = fa_results[:ntrain]
        fa_results_test = fa_results[ntrain:]

    if use_grp:
        print("GRP")
        grp = GaussianRandomProjection(n_components = N_COMP, eps=0.1, random_state=42)
        grp_results = grp.fit_transform(total)
        grp_results_train = grp_results[:ntrain]
        grp_results_test = grp_results[ntrain:]

    if use_srp:
        print("SRP")
        srp = SparseRandomProjection(n_components = N_COMP, dense_output=True, random_state=42)
        srp_results = srp.fit_transform(total)
        srp_results_train = srp_results[:ntrain]
        srp_results_test = srp_results[ntrain:]
        
    print("Append decomposition components together...")
    train_decomposed = np.concatenate([srp_results_train, grp_results_train, ica_results_train, pca_results_train, tsvd_results_train], axis=1)
    test_decomposed = np.concatenate([srp_results_test, grp_results_test, ica_results_test, pca_results_test, tsvd_results_test], axis=1)

    train_with_only_decomposed_features = pd.DataFrame(train_decomposed)
    test_with_only_decomposed_features = pd.DataFrame(test_decomposed)
    
    for agg_col in ['sum', 'var', 'mean', 'median', 'std', 'weight_count', 'count_non_0', 'num_different', 'max', 'min']:
        train_with_only_decomposed_features[col] = train[col]
        test_with_only_decomposed_features[col] = test[col]
    
    # Remove any NA
    train_with_only_decomposed_features = train_with_only_decomposed_features.fillna(0)
    test_with_only_decomposed_features = test_with_only_decomposed_features.fillna(0)
    
    return train_with_only_decomposed_features, test_with_only_decomposed_features


# ### **Build validation set using adversarial validation**

# In[ ]:


def generate_adversarial_validation_set(train, test):
    x_test = test.drop(["is_test", "target"], 1)

    train, val = train[train.predicted_probs < 0.9], train[train.predicted_probs > 0.9]
    train = train.drop(["is_test", "predicted_probs"], 1)
    val = val.drop(["is_test", "predicted_probs"], 1)

    x_train, y_train = train.drop("target", 1), train.target
    x_val, y_val = val.drop("target", 1), val.target
    print("\nTrain shape: {}\nValidation shape: {}\nTest shape: {}".format(x_train.shape, x_val.shape, x_test.shape))
    
    return x_train, y_train, x_val, y_val, x_test

def get_training_set_with_test_set_similarity_predictions(X_train, Y_train, X_test):
    print("############# Generate adversarial validation set #############")
    
    print("Add target column")
    X_train['target'] = Y_train
    X_test['target'] = 0
    
    X_train["is_test"] = 0
    X_test["is_test"] = 1
    assert(np.all(train.columns == test.columns))
    
    print("Concat train and test data")
    total = pd.concat([X_train, X_test])
    total = total.fillna(0)
    
    x = total.drop(["is_test", "target"], axis = 1)
    y = total.is_test
    
    print("Start cross-validating")
    n_estimators = 100
    classifier = RandomForestClassifier(n_estimators = n_estimators, n_jobs = -1)
    predictions = np.zeros(y.shape)
    
    stratified_kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 5678)
    
    for fold_index, (train_indices, test_indices) in enumerate(stratified_kfold.split(x, y)):
        print("Fold - " + str(fold_index))
        
        x_train = x.iloc[train_indices]
        y_train = y.iloc[train_indices]
        x_test = x.iloc[test_indices]
        y_test = y.iloc[test_indices]
        
        classifier.fit(x_train, y_train)
        
        predicted_probabilities = classifier.predict_proba(x_test)[:, 1]
        
        auc = roc_auc_score(y_test, predicted_probabilities)
        print("AUC Score - " + str(auc) + "%")
        
        predictions[test_indices] = predicted_probabilities
    total['predicted_probs'] = predictions
    
    print("Generating training set")
    total = total[total.is_test == 0]
    
    print("Sorting according to predictions")
    train_set_with_predictions_for_test_set_similarity = total.sort_values(["predicted_probs"], ascending = False)
    
    return train_set_with_predictions_for_test_set_similarity, X_test


# ### **Building the model**

# In[ ]:


def RMSLE(actual, predicted):
    return np.sqrt(np.mean(np.power(np.log1p(actual) - np.log1p(predicted), 2)))

# LightGBM Model
def run_lgb(x_train, y_train, x_val, y_val, x_test):
    print("############# Build LightGBM model #############")
    model_lgb = lgb.LGBMRegressor(objective='regression',
                                  num_leaves = 144,
                                  learning_rate = 0.005, 
                                  n_estimators = 720, 
                                  max_depth = 13,
                                  metric = 'rmse',
                                  is_training_metric = True,
                                  max_bin = 55, 
                                  bagging_fraction = 0.8,
                                  verbose = -1,
                                  bagging_freq = 5, 
                                  feature_fraction = 0.9)
    
    print("Validating.....")
    model_lgb.fit(x_train, y_train, eval_set = (x_val, y_val), early_stopping_rounds = 100, verbose = True, eval_metric = 'rmse')
    
    print("Train on full dataset")
    X_TRAIN = pd.concat([x_train, x_val])
    Y_TRAIN = pd.concat([y_train, y_val])
    
    model_lgb.fit(X = X_TRAIN,
                  y = Y_TRAIN)
    
    
    y_pred_test = np.expm1(model_lgb.predict(X_TEST))
    print("LightGBM Training Completed...")
    
    return y_pred_test

# XGBoost Model
def run_xgb(x_train, y_train, x_val, y_val, X_TEST):
    print("############# Build XGBoost model #############")
    model_xgb = xgb.XGBRegressor(colsample_bytree = 0.055, 
                                 colsample_bylevel = 0.5, 
                                 gamma = 1.5, 
                                 learning_rate = 0.02, 
                                 max_depth = 32, 
                                 objective = 'reg:linear',
                                 booster = 'gbtree',
                                 min_child_weight = 57, 
                                 n_estimators = 1000, 
                                 reg_alpha = 0, 
                                 reg_lambda = 0,
                                 eval_metric = 'rmse', 
                                 subsample = 0.7, 
                                 silent = 1, 
                                 n_jobs = -1, 
                                 early_stopping_rounds = 14,
                                 random_state = 7, 
                                 nthread = -1)
    
    print("Validating.....")
    model_xgb.fit(x_train, y_train, eval_set = [(x_train, y_train), (x_val, y_val)], eval_metric = 'rmse', early_stopping_rounds = 100, verbose = True)
    
    print("Train on full dataset")
    X_TRAIN = pd.concat([x_train, x_val])
    Y_TRAIN = pd.concat([y_train, y_val])
    
    model_xgb.fit(X = X_TRAIN, 
                  y = Y_TRAIN)
    
    y_pred_test = np.expm1(model_xgb.predict(X_TEST))
    print("XGBoost Training Completed...")
    
    return y_pred_test
    
# CatBoost Model
# def run_cbm(x_train, y_train, x_val, y_val, X_TEST):
#     print("############# Build CatBoost model #############")
    
#     model_cb = cb.CatBoostRegressor(iterations = 500,
#                                  learning_rate = 0.05,
#                                  depth = 10,
#                                  eval_metric = 'RMSE',
#                                  random_seed = 42,
#                                  bagging_temperature = 0.2,
#                                  od_type = 'Iter',
#                                  metric_period = 50,
#                                  od_wait = 20)
    
#     model_cb.fit(x_train, y_train)
#     rmsle = RMSLE(np.expm1(y_val), np.expm1(model_cb.predict(x_val)))
#     print("The RMSLE score on validation set is - " + str(rmsle))
    
#     X_TRAIN = pd.concat([x_train, x_val])
#     Y_TRAIN = pd.concat([y_train, y_val])
    
#     model_cb.fit(X = X_TRAIN, 
#                  y = Y_TRAIN)
    
#     y_pred_test = np.expm1(model_cb.predict(X_TEST))
#     print("CatBoost Training Completed...")
    
#     return y_pred_test


# ### **Ensemble predictions**

# In[ ]:


def ensemble_predictions(lgb_pred = None, xgb_pred = None, cbm_pred = None, lgb_ratio = 1/3, xgb_ratio = 1/3, cbm_ratio = 1/3):
    print("############# Ensemble model predictions #############")
    
    y_pred_test_final = lgb_pred * lgb_ratio + xgb_pred * xgb_ratio #+ cbm_pred * cbm_ratio
    return y_pred_test_final


# ### **Save submission file**

# In[ ]:


def save_submission_file(y_pred_test, test_ID, model_name):
    print("############# Save submission files #############")
    
    sub = pd.DataFrame(y_pred_test)
    sub.columns = ['target']
    sub.insert(0, 'ID', test_ID)
    print(sub.head())
    sub.to_csv(model_name +'_10_all_decomposition_features.csv', index = False)


# ### **Creating submission**

# In[ ]:


train_orig, test_orig = read_data()


# In[ ]:


train, test, y_train, test_ID = convert_and_drop_target(train = train_orig.copy(), 
                                                        test = test_orig.copy())

train_without_duplicate_columns, test_without_duplicate_columns = remove_duplicate_columns(train = train.copy(), 
                                                                                           test = test.copy())

train_without_constant_columns, test_without_constant_columns = remove_constant_columns(train = train_without_duplicate_columns.copy(), 
                                                                                        test = test_without_duplicate_columns.copy())

train_with_columns_of_high_importance, test_with_columns_of_high_importance = remove_features_using_importance(x_train = train_without_constant_columns.copy(), 
                                                                                                              y_train = y_train.copy(), 
                                                                                                              x_test = test_without_constant_columns.copy())

train_with_columns_having_same_distributions, test_with_columns_having_same_distributions = remove_features_having_different_distributions(train = train_with_columns_of_high_importance.copy(), 
                                                                                                                                           test = test_with_columns_of_high_importance.copy())

total = pd.concat([train_with_columns_having_same_distributions, test_with_columns_having_same_distributions])
total_scaled = mean_variance_scale_columns(total = total.copy())

train_with_aggregate_features, test_with_aggregate_features = add_aggregate_features(train = train_with_columns_having_same_distributions.copy(), 
                                                                                     test = test_with_columns_having_same_distributions.copy())


train_with_decomposed_features, test_with_decomposed_features = add_decomposed_features_back_to_df(train = train_with_aggregate_features.copy(),
                                                                                                 test = test_with_aggregate_features.copy(),
                                                                                                 total = total_scaled.copy(),
                                                                                                 n_components = 100,
                                                                                                 use_pca = False,
                                                                                                 use_tsne = False,
                                                                                                 use_tsvd = False,
                                                                                                 use_ica = False,
                                                                                                 use_fa = False,
                                                                                                 use_grp = False,
                                                                                                 use_srp = True)


training_set, testing_set = get_training_set_with_test_set_similarity_predictions(train_with_decomposed_features.copy(), y_train.copy(), test_with_decomposed_features.copy())

x_train, y_train, x_val, y_val, X_TEST = generate_adversarial_validation_set(training_set.copy(), testing_set.copy())


# In[ ]:


# y_pred_test_lbg = run_lgb(x_train, y_train, x_val, y_val, X_TEST)
y_pred_test_xgb = run_xgb(x_train, y_train, x_val, y_val, X_TEST)
# y_pred_test_cbm = run_cbm(x_train, y_train, x_val, y_val, X_TEST)

y_pred_test_final = ensemble_predictions(lgb_pred = y_pred_test_lbg, 
                                         xgb_pred = y_pred_test_xgb, 
                                         cbm_pred = None, 
                                         lgb_ratio = 1/3, 
                                         xgb_ratio = 1/3, 
                                         cbm_ratio = 1/3)

save_submission_file(y_pred_test = y_pred_test_final, test_ID = test_ID, model_name = "Ensemble_LGB_XGB_with_columns_scaled")


# In[ ]:




