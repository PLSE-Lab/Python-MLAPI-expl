#!/usr/bin/env python
# coding: utf-8

# # Summary
# * LightGBM regressor.
# * A customized loss function (a weighted combo of fair+huber+RMSE+MAE).
# * Testing a bootstrapping approach by adding an estimate of `time_after_failure` to the features, to which the true `time_to_failure` is highly correlated to. 
# * Conclusion: bootstrapping will lead to leakage and deteriorate our model. 
# 
# Reference:
# * [My tries to find magic features](https://www.kaggle.com/scaomath/lanl-earthquakes-try-to-find-magic-features)
# * [BigIronSphere's data augmentation](https://www.kaggle.com/bigironsphere/basic-data-augmentation-feature-reduction)
# * [Andrew's "Even more features" notebook](https://www.kaggle.com/artgor/even-more-features)

# In[ ]:


import os
paths = os.listdir("../input")
print(paths)


# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()

from tqdm import tqdm_notebook

from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold
from scipy import stats


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train_features_ = pd.read_csv('../input/lanl-earthquake-nonmagic-features/train_X.csv')
train_y_ = pd.read_csv('../input/lanl-earthquake-nonmagic-features/train_y.csv')
train_y_extra_ = pd.read_csv('../input/lanl-earthquake-nonmagic-features/train_y_extra.csv')
test_features_ = pd.read_csv('../input/lanl-earthquake-nonmagic-features/test_X.csv')


# In[ ]:


X_train = train_features_.copy()
X_test = test_features_.copy()
y_train = train_y_


# In[ ]:


X_train.head(3)


# In[ ]:


sns.distplot(X_train['mean'],label='Training samples')
sns.distplot(X_test['mean'],label='Testing samples')
plt.legend();


# In[ ]:


sns.distplot(X_train['fft_skew'],label='Training samples')
sns.distplot(X_test['fft_skew'],label='Testing samples')
plt.legend();


# # time_after_failure

# In[ ]:


rows = 150_000
y_train_new = pd.DataFrame(columns=['taf'], dtype=np.float64, index = y_train.index)
y_train_new['taf'] = train_y_extra_['time_after_failure']


# In[ ]:


# most correlated features with time_after_failure
cols = list(np.abs(X_train.corrwith(y_train_new['taf'])).sort_values(ascending=False).head(50).index)
cols[::5]


# In[ ]:


cols_plot = ['num_peaks_5',
             'std_0_to_10',
             'fft_100_roll_std_70',
             'std_0_to_10',
             'iqr',
             'energy_spectra_9306hz']
_, ax1 = plt.subplots(3, 2, figsize=(20, 12))
ax1 = ax1.reshape(-1)

for i, col in enumerate(cols_plot):
    ax1[i].plot(X_train[col], color='blue')
    ax1[i].set_title(col)
    ax1[i].set_ylabel(col, color='b')

    ax2 = ax1[i].twinx()
    ax2.plot(y_train_new['taf'], color='g', linewidth=2)
    ax2.set_ylabel('time_after_failure', color='g')
    ax2.legend([col, 'time_after_failure'], loc= 'upper right')
    ax2.grid(False)


# # LightGBM model

# In[ ]:


coef = [0.35, 0.5, 0.05, 0.1]

def custom_objective(y_true, y_pred):
    
    # fair
    c = 0.5
    residual = y_pred - y_true
    grad = c * residual /(np.abs(residual) + c)
    hess = c ** 2 / (np.abs(residual) + c) ** 2
    
    # huber
    h = 1.2  #h is delta in the formula
    scale = 1 + (residual / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad_huber = residual / scale_sqrt
    hess_huber = 1 / scale / scale_sqrt

    # rmse grad and hess
    grad_rmse = residual
    hess_rmse = 1.0

    # mae grad and hess
    grad_mae = np.array(residual)
    grad_mae[grad_mae > 0] = 1.
    grad_mae[grad_mae <= 0] = -1.
    hess_mae = 1.0

    return coef[0] * grad + coef[1] * grad_huber + coef[2] * grad_rmse + coef[3] * grad_mae,            coef[0] * hess + coef[1] * hess_huber + coef[2] * hess_rmse + coef[3] * hess_mae


def logcosh_objective(y_true, y_pred):
    d = y_pred - y_true 
    grad = np.tanh(d)/y_true
    hess = (1.0 - grad*grad)/y_true
    return grad, hess


def huber_objective(y_true, y_pred):
    d = y_pred - y_true
    h = 1.2  #h is delta in the formula
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess


# In[ ]:


params = {'bagging_fraction': 0.71,
         'boosting': 'gbdt',
         'feature_fraction': 0.94,
         'lambda_l1': 3.131216244016188,
         'lambda_l2': 2.4124061313905836,
         'learning_rate': 0.049886848207269734,
         'max_bin': 193,
         'max_depth': 15,
         'metric': 'MAE',
         'min_data_in_bin': 167,
         'min_data_in_leaf': 62,
         'min_gain_to_split': 2.07,
         'num_leaves': 38,
         'objective': custom_objective,
         'subsample': 0.9133120405819966}


# In[ ]:


def get_prediction(X, y, X_test=None, 
                   n_fold=5, random_state=1127, eval_metric='mae',
                   verbose=0, early_stopping_rounds=1000):
    '''
    X: dataframe
    y: series or dataframe
    params: global variable
    '''
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=random_state)
    n_fold_disp = n_fold // 2
    
    if X_test is None:
        X_test = X
    
    pred_train = np.zeros(len(X))
    pred_oof = np.zeros(len(X))
    pred_test = np.zeros(len(X_test))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X,y)):
        
        if (fold_% n_fold_disp == 0 or fold_ == n_fold-1) and verbose > 0:
            print("Fold {}".format(fold_))

        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y.values[trn_idx].reshape(-1), y.values[val_idx].reshape(-1)

        model = lgb.LGBMRegressor(**params, n_estimators = 10000, n_jobs = -1)
        model.fit(X_tr, y_tr, 
                  eval_set=[(X_tr, y_tr), (X_val, y_val)], 
                  eval_metric=eval_metric,
                  verbose=verbose, 
                  early_stopping_rounds=early_stopping_rounds)

        #predictions
        pred_train += model.predict(X, num_iteration=model.best_iteration_) / folds.n_splits
        pred_test += model.predict(X_test, num_iteration=model.best_iteration_) / folds.n_splits
        pred_oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration_)
        
    return pred_train, pred_test, pred_oof, model


# # Trial run on training data using selected features
# 
# * Compute `time_to_failure`'s estimate based on `X_train`.
# * Estimate `time_after_failure` for training set.
# * Adding `time_after_failure`'s estimate into `X_train`.
# * Compute `time_to_failure` for training set again.

# In[ ]:


features_select = ['abs_min',
             'abs_q01',
             'abs_q05',
             'abs_trend',
             'autocorrelation_10',
             'autocorrelation_1000',
             'autocorrelation_500',
             'av_change_abs_roll_mean_10',
             'av_change_abs_roll_std_10',
             'av_change_abs_roll_std_100',
             'av_change_abs_roll_std_1000',
             'avg_first_10000',
             'avg_last_10000',
             'avg_last_5000',
             'c3_5',
             'c3_500',
             'classic_sta_lta1_mean',
             'classic_sta_lta4_mean',
             'classic_sta_lta5_mean',
             'count_big_50000_less_threshold_5',
             'energy_spectra_2640hz',
             'energy_spectra_9306hz',
             'energy_spectra_norm_109306hz_denoised',
             'energy_spectra_norm_129306hz',
             'energy_spectra_norm_135973hz_denoised',
             'energy_spectra_norm_142640hz_denoised',
             'energy_spectra_norm_149306hz_denoised',
             'energy_spectra_norm_15973hz_denoised',
             'energy_spectra_norm_169306hz_denoised',
             'energy_spectra_norm_22640hz_denoised',
             'energy_spectra_norm_29306hz_denoised',
             'energy_spectra_norm_35973hz_denoised',
             'energy_spectra_norm_42640hz_denoised',
             'energy_spectra_norm_49306hz_denoised',
             'energy_spectra_norm_55973hz_denoised',
             'energy_spectra_norm_62640hz_denoised',
             'energy_spectra_norm_89306hz_denoised',
             'energy_spectra_norm_9306hz_denoised',
             'energy_spectra_norm_95973hz_denoised',
             'fft_1000_roll_std_20',
             'fft_1000_roll_std_25',
             'fft_1000_roll_std_70',
             'fft_100_roll_std_1',
             'fft_100_roll_std_20',
             'fft_100_roll_std_70',
             'fft_100_roll_std_75',
             'fft_10_roll_std_75',
             'fft_mean_change_rate',
             'fft_min',
             'fft_min_roll_mean_100',
             'fft_min_roll_std_100',
             'fft_skew',
             'fft_skew_first_50000',
             'fft_spkt_welch_density_100',
             'fft_spkt_welch_density_5',
             'fft_spkt_welch_density_50',
             'fft_time_rev_asym_stat_10',
             'fft_time_rev_asym_stat_100',
             'iqr',
             'kstat_3',
             'mad',
             'max_first_5000',
             'max_last_10000',
             'max_roll_mean_1000',
             'max_to_min',
             'mean_change_abs',
             'med',
             'min_last_10000',
             'min_roll_std_100',
             'num_crossings',
             'num_peaks_10',
             'num_peaks_5',
             'q01_roll_mean_100',
             'q01_roll_std_10',
             'q01_roll_std_1000',
             'q05_roll_mean_100',
             'q05_roll_std_1000',
             'q95',
             'q95_roll_mean_10',
             'q95_roll_mean_100',
             'q95_roll_std_1000',
             'q99_roll_mean_1000',
             'skew',
             'std_0_to_10',
             'std_first_5000',
             'std_neg_10_to_0',
             'std_neg_2_to_2',
             'std_neg_5_to_5',
             'std_roll_mean_1000']


# In[ ]:


X_train_select = X_train[features_select].copy()
X_test_select = X_test[features_select].copy()


# In[ ]:


ttf_pred_orig, ttf_test_orig, ttf_pred_oof, model1= get_prediction(X_train_select, y_train,
                                        X_test = X_test_select)

print("Training MAE for time_to_failure is {:.7f}"          .format(np.abs(y_train['time_to_failure'] - ttf_pred_orig).mean())) 
print("OOF MAE for time_to_failure is {:.7f}"          .format(np.abs(y_train['time_to_failure'] - ttf_pred_oof).mean())) 


# In[ ]:


plt.figure(figsize=(10,4))
sns.distplot(y_train['time_to_failure'] , color="red", label="y_train")
sns.distplot(pd.DataFrame(ttf_test_orig)[0], color="skyblue", label="lgb OOF pred for y_train")
plt.legend();


# In[ ]:


taf_pred, _, taf_pred_oof, model2 = get_prediction(X_train_select, y_train_new)
    
print("Training MAE for time_after_failure is {:.7f}"      .format((y_train_new['taf'] - taf_pred).abs().mean()))
print("OOF MAE for time_after_failure is {:.7f}"          .format(np.abs(y_train_new['taf'] - taf_pred_oof).mean())) 


# Checking the correlation of the `time_after_failure` (including the estimate by LGB regressor) and the `time_to_failure`.

# In[ ]:


y_train['time_to_failure'].corr(y_train_new['taf'])


# In[ ]:


y_train['time_to_failure'].corr(pd.Series(taf_pred))


# In[ ]:


_, ax = plt.subplots(2,1, figsize=(12,8))


ax[0].plot(taf_pred_oof, color='r', label='taf prediction')
ax[0].plot(y_train_new.values, color='b', label='time_after_failure', linewidth = 2)
ax[0].set_ylabel('taf', color='orange')
ax[0].autoscale(axis='x',tight=True)
ax[0].set_title("OOF LightGBM prediction vs TAF")
ax[0].legend(loc='best');

ax[1].plot(ttf_pred_oof, color='orange', label='ttf prediction')
ax[1].plot(y_train['time_to_failure'], color='b', label='time_to_failure', linewidth = 2)
ax[1].set_ylabel('ttf', color='r')
ax[1].autoscale(axis='x',tight=True)
ax[1].set_title("OOF LightGBM prediction vs TTF")
ax[1].legend(loc='best');


# # Visualization of important features

# Not surprising that the most important ones are all FFT-based features.

# In[ ]:


important_feature_index = np.argsort(model1.feature_importances_)[::-1]
cols = X_train.columns[important_feature_index[:100]]
cols[:20]


# In[ ]:


cols_top8 = cols[:8]
_, ax1 = plt.subplots(4, 2, figsize=(20, 20))
ax1 = ax1.reshape(-1)

for i, col in enumerate(cols_top8):
    ax1[i].plot(X_train[col], color='blue')
    ax1[i].set_title(col)
    ax1[i].set_ylabel(col, color='b')

    ax2 = ax1[i].twinx()
    ax2.plot(y_train['time_to_failure'], color='g', linewidth=2)
    ax2.set_ylabel('time_to_failure', color='g')
    ax2.legend([col, 'time_to_failure'], loc= 'upper right')
    ax2.grid(False)


# ## Adding `time_after_failure` estimate to training

# In[ ]:


X_train['taf_estimate'] = taf_pred_oof
ttf_pred,_,ttf_pred_oof, model3 = get_prediction(X_train, y_train)

print("\nTraining MAE for time_to_failure is {:.7f}"      .format(np.abs(y_train['time_to_failure'] - ttf_pred).mean()))
print("OOF MAE for time_to_failure is {:.7f}"          .format(np.abs(y_train['time_to_failure'] - ttf_pred_oof).mean())) 


# In[ ]:


important_feature_index = np.argsort(model3.feature_importances_)[::-1]
X_train.columns[important_feature_index[:10]]


# It is not surprising after adding `time_after_failure` estimate to the features, it becomes the most important one...

# In[ ]:


plt.figure(figsize=(10,4))
sns.distplot(y_train['time_to_failure'] , color="red", label="y_train")
sns.distplot(pd.DataFrame(ttf_pred_oof)[0], color="skyblue", label="lgb pred for y_train")
plt.title("TTF vs LGB OOF prediction after 1 iteration of bootstrap")
plt.legend();


# # Bootstrapping effectivity test
# 
# Reload `X_train` and `X_test`, and do a manual split `X_cv` to test whether the bootstrapping works or not.

# In[ ]:


X_train = X_train_select.copy()
X_test = X_test_select.copy()


# In[ ]:


X_bsp_train = X_train[1200:]
X_bsp_cv = X_train[:1200]
y_bsp_train = y_train[1200:]
y_bsp_cv = y_train[:1200]
y_bsp_train_new = y_train_new[1200:]
y_bsp_cv_new = y_train_new[:1200]


# In[ ]:


n_fold = 6
random_state = 802


# ## Step 0: get reference time_to_failure testing error

# In[ ]:


ttf_est_train, ttf_est_cv, ttf_est_oof, _ = get_prediction(X_bsp_train, 
                                            y_bsp_train, 
                                            X_test=X_bsp_cv, 
                                            n_fold=n_fold,
                                            random_state=random_state)

print("TTF CV MAE: {:.7f}"      .format(np.abs(y_bsp_cv['time_to_failure'] - ttf_est_cv).mean())) 
print("TTF OOF MAE: {:.7f}"      .format(np.abs(y_bsp_train['time_to_failure'] - ttf_est_oof).mean()))


# ## Step 1: estimate time_after_failure for CV and train

# In[ ]:


taf_est_train, taf_est_cv, taf_est_oof, _ = get_prediction(X_bsp_train, 
                                             y_bsp_train_new, 
                                             X_test=X_bsp_cv, 
                                             n_fold=n_fold,
                                             random_state=random_state)


# ## Step 2: estimate time_to_failure for test and train

# In[ ]:


X_bsp_train = X_bsp_train.assign(taf_estimate = taf_est_oof);
X_bsp_cv = X_bsp_cv.assign(taf_estimate = taf_est_cv);


# In[ ]:


ttf_est_train, ttf_est_cv, ttf_est_oof, model_trial  = get_prediction(X_bsp_train, y_bsp_train,
                                                X_test=X_bsp_cv, 
                                                n_fold=n_fold,random_state=random_state)

print("TTF CV MAE after adding TAF to training features: {:.7f}"      .format(np.abs(y_bsp_cv['time_to_failure'] - ttf_est_cv).mean())) 
print("TTF OOF MAE after adding TAF to training features: {:.7f}"      .format(np.abs(y_bsp_train['time_to_failure'] - ttf_est_oof).mean())) 


# ## Feature importance for CV set

# In[ ]:


from eli5 import show_weights
from eli5.sklearn import PermutationImportance
from IPython.display import display


# In[ ]:


perm = PermutationImportance(model_trial, random_state=1).fit(X_bsp_cv,y_bsp_cv)
display(show_weights(perm, feature_names = X_bsp_cv.columns.tolist()))


# ## Step 3: iteratively updating ttf and taf like a leapfrog scheme
# (This is problematic) Now we can add the estimate of `time_to_failure` based on the `time_after_failure` estimate to the training features. When estimating `time_to_failure`, we drop `time_to_failure`'s estimate feature, and vice versa for `time_after_failure`. In this way, we always estimate one another in a bootstrapping fashion.

# In[ ]:


X_bsp_train['ttf_estimate'] = ttf_est_oof
X_bsp_cv['ttf_estimate'] = ttf_est_cv


# In[ ]:


num_bootstrap = 2
coeff_bsp = [0.4, 0.6] # coeff for [old, new]

for b in tqdm_notebook(range(num_bootstrap)):
    
    print("\nBootstrapping iteration {0}.".format(b+1))
    
    # reset time_after_failure values
    random_state = b
    taf_est_train, taf_est_cv, taf_est_oof, model_taf =     get_prediction(X_bsp_train.drop(columns=['taf_estimate']), 
                   y_bsp_train_new, 
                   X_test=X_bsp_cv.drop(columns=['taf_estimate']), 
                   n_fold=n_fold,random_state=random_state)
    
    print("TAF training OOF MAE: {:.7f}"      .format(np.abs(y_bsp_train_new['taf'] - taf_est_oof).mean())) 
    print("TAF CV MAE: {:.7f}"      .format(np.abs(y_bsp_cv_new['taf'] - taf_est_cv).mean())) 
    print("Mean abs change in TAF feature for CV: {:.7f}"      .format(np.abs(X_bsp_cv['taf_estimate'] - taf_est_cv).mean())) 
    
    # update the time_after_failure feature
    X_bsp_train['taf_estimate'] = coeff_bsp[0]*X_bsp_train['taf_estimate'] + coeff_bsp[1]*taf_est_oof
    X_bsp_cv['taf_estimate'] = coeff_bsp[0]*X_bsp_cv['taf_estimate'] + coeff_bsp[1]*taf_est_cv
    
    # reset time_to_failure value
    ttf_est_train, ttf_est_cv, ttf_est_oof, model =     get_prediction(X_bsp_train.drop(columns=['ttf_estimate']), 
                   y_bsp_train, 
                   X_test=X_bsp_cv.drop(columns=['ttf_estimate']), 
                   n_fold=n_fold,random_state=random_state)
    
    print("TTF training OOF MAE: {:.7f}"      .format(np.abs(y_bsp_train['time_to_failure'] - ttf_est_oof).mean())) 
    print("TTF CV MAE: {:.7f}"      .format(np.abs(y_bsp_cv['time_to_failure'] - ttf_est_cv).mean())) 
    print("Mean abs change in TTF feature for CV: {:.7f}"      .format(np.abs(X_bsp_cv['ttf_estimate'] - ttf_est_cv).mean())) 
    
    # updating the time_to_failure feature in order to bootstrap time_after_failure feature
    X_bsp_train['ttf_estimate'] = coeff_bsp[0]*X_bsp_train['ttf_estimate'] + coeff_bsp[1]*ttf_est_oof
    X_bsp_cv['ttf_estimate'] = coeff_bsp[0]*X_bsp_cv['ttf_estimate'] + coeff_bsp[1]*ttf_est_cv


# In[ ]:


important_feature_index = np.argsort(model.feature_importances_)[::-1]
X_bsp_train.columns[important_feature_index[:20]]


# # Observation
# 
# Even as the second target for bootstrapping becomes the most important features, the MAE is getting worse and worse.

# In[ ]:


_, ax = plt.subplots(2,1, figsize=(12,8))
ax[0].plot(taf_est_cv, color='r', label='taf prediction')
ax[0].plot(y_bsp_cv_new['taf'], color='b', label='time_after_failure', linewidth = 2)
ax[0].set_ylabel('taf', color='orange')
ax[0].set_title("LightGBM CV prediction vs TAF")
ax[0].legend(loc='best');

ax[1].plot(ttf_est_cv, color='orange', label='ttf prediction')
ax[1].plot(y_bsp_cv['time_to_failure'], color='b', label='time_to_failure', linewidth = 2)
ax[1].set_ylabel('ttf', color='r')
ax[1].set_title("LightGBM CV prediction vs TTF")
ax[1].legend(loc='best');


# # Bootstrapping for actual testing data

# In[ ]:


X_train = train_features_[features_select]
X_test = test_features_[features_select]


# In[ ]:


print("Final bootstrapping is performed based on {} features.".format(X_train.shape[-1]))


# In[ ]:


n_fold = 6
random_state = 1127

params = {'bagging_fraction': 0.51,
         'boosting': 'gbdt',
         'feature_fraction': 0.9,
         'lambda_l1': 2,
         'lambda_l2': 0.03,
         'learning_rate': 0.1,
         'max_bin': 48,
         'max_depth': 8,
         'metric': 'MAE',
         'min_data_in_bin': 57,
         'min_data_in_leaf': 11,
         'min_gain_to_split': 0.53,
         'num_leaves': 83,
         'objective': custom_objective,
         'subsample': 0.55}


# In[ ]:


# Step 1: get time_after_failure estimate
taf_est_train, taf_est_test, taf_est_oof, _ = get_prediction(X_train, y_train_new, 
                                               X_test=X_test, 
                                               n_fold=n_fold,random_state=random_state)


# In[ ]:


# Step 2: 
X_train['taf_estimate'] = taf_est_oof
X_test['taf_estimate'] = taf_est_test

ttf_est_train, ttf_est_test, ttf_est_oof, _   = get_prediction(X_train, y_train, 
                                                X_test=X_test, 
                                                n_fold=n_fold,random_state=random_state)
ttf_est_test_bsp1 = ttf_est_test
print("TTF OOF MAE after adding TAF to training features: {:.7f}"      .format((np.abs(y_train['time_to_failure'] - ttf_est_oof)).mean())) 


# In[ ]:


# Step 3
X_train['ttf_estimate'] = ttf_est_oof
X_test['ttf_estimate'] = ttf_est_test


# In[ ]:


# Step 3:
num_bootstrap = 20
coeff_bsp = [0.4, 0.6] # coeff for [old, new]

for b in tqdm_notebook(range(num_bootstrap)):
    
    print("\nBootstrapping iteration {0}.".format(b+1))
    random_state = b
    # reset time_after_failure values
    taf_est_train, taf_est_test, taf_est_oof, _ =     get_prediction(X_train.drop(columns=['taf_estimate']), 
                   y_train_new, 
                   X_test=X_test.drop(columns=['taf_estimate']), 
                   n_fold=n_fold,random_state=random_state)
    
    print("TAF training OOF MAE: {:.7f}"      .format(np.abs(y_train_new['taf'] - taf_est_oof).mean())) 
#     print("Mean abs change in TAF feature for training: {:.7f}"\
#       .format(np.abs(X_train['taf_estimate'] - taf_est_train).mean())) 
#     print("Correlation between TAF estimate and time_to_failure: {:.5f}"\
#           .format(y_train['time_to_failure'].corr(pd.Series(taf_est_train))))
    
    # update the time_after_failure feature
    X_train['taf_estimate'] = coeff_bsp[0]*X_train['taf_estimate'] + coeff_bsp[1]*taf_est_oof
    X_test['taf_estimate'] = coeff_bsp[0]*X_test['taf_estimate'] + coeff_bsp[1]*taf_est_test
    
    # reset time_to_failure value
    ttf_est_train, ttf_est_test, ttf_est_oof, model =     get_prediction(X_train.drop(columns=['ttf_estimate']), 
                   y_train, 
                   X_test=X_test.drop(columns=['ttf_estimate']), 
                   n_fold=n_fold,random_state=random_state)
    
    print("TTF training OOF MAE: {:.7f}"      .format((np.abs(y_train['time_to_failure'] - ttf_est_oof)).mean())) 
#     print("Mean abs change in TTF feature for training: {:.7f}"\
#       .format((np.abs(X_train['ttf_estimate'] - ttf_est_train)).mean())) 
#     print("Correlation between TTF estimate and time_after_failure: {:.5f}"\
#           .format(y_train_new['taf'].corr(pd.Series(ttf_est_train))))
    
    # updating the time_to_failure feature in order to bootstrap time_after_failure feature
    X_train['ttf_estimate'] = coeff_bsp[0]*X_train['ttf_estimate'] + coeff_bsp[1]*ttf_est_oof
    X_test['ttf_estimate'] = coeff_bsp[0]*X_test['ttf_estimate'] + coeff_bsp[1]*ttf_est_test


# # Conclusion:
# After several iteration of bootstrapping, the leakage is severe.

# In[ ]:


plt.figure(figsize=(10,4))
sns.distplot(y_train['time_to_failure'] , color="red", label="time_to_failure")
sns.distplot(pd.DataFrame(ttf_est_oof)[0], color="skyblue", label="LGB bootstrap OOF training pred")
# sns.distplot(pd.DataFrame(ttf_est_train)[0], label="LGB bootstrap training pred")
plt.legend();


# In[ ]:


plt.figure(figsize=(10,4))
sns.distplot(pd.DataFrame(ttf_est_test)[0] , color="skyblue", 
             label="Bootstrapped prediction for X_test ")
sns.distplot(pd.DataFrame(ttf_est_test_bsp1)[0] , color="orange", 
             label="After adding taf estimate to feature once")
sns.distplot(pd.DataFrame(ttf_test_orig)[0] , color="green", 
             label="Original prediction for X_test")
plt.legend();


# In[ ]:


plt.figure(figsize=(20, 6))
plt.plot(y_train.values, color='b', label='time_to_failure', linewidth = 2)
plt.plot(ttf_est_oof, color='r', label='LGB estimate')
plt.legend(loc='best')
plt.autoscale(axis='x',tight=True)
plt.title('TTF vs LGB OOF prediction after {0} iteration of bootstrap'.format(num_bootstrap));


# In[ ]:


submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv', 
                         index_col='seg_id')
submission_nonbsp = submission.copy()
submission_bsp1 = submission.copy() # only bootstrapping once
submission['time_to_failure'] = ttf_est_test
submission_bsp1['time_to_failure'] = ttf_est_test_bsp1
submission_nonbsp['time_to_failure'] = ttf_test_orig
submission.to_csv('submission.csv')
submission_nonbsp.to_csv('submission_nonbsp.csv')
submission_bsp1.to_csv('submission_bsp1.csv')

