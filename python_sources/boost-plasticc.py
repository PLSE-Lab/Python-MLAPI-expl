#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import Libraries
import pandas as pd
import numpy as np
from numba import jit
import tsfresh
from tsfresh.feature_extraction import extract_features
import sys
import gc; gc.enable()
import time

from datetime import datetime
from functools import partial
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

import os
print(os.listdir("../input"))


# In[ ]:


@jit
def haversine_plus(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees) from
    #https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    # Convert decimal degrees to Radians:
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    # Haversine Formula:
    dlon = np.subtract(lon2, lon1)
    dlat = np.subtract(lat2, lat1)

    a = np.add(np.power(np.sin(np.divide(dlat, 2)), 2),
               np.multiply(np.cos(lat1),
                           np.multiply(np.cos(lat2),
                                       np.power(np.sin(np.divide(dlon, 2)), 2))))

    haversine = np.multiply(2, np.arcsin(np.sqrt(a)))
    return {
        'haversine': haversine,
        'latlon1': np.subtract(np.multiply(lon1, lat1), np.multiply(lon2, lat2)),
    }


@jit
def process_flux(df):
    flux_ratio_sq = np.power(df['flux'].values / df['flux_err'].values, 2.0)

    df_flux = pd.DataFrame({
        'flux_ratio_sq': flux_ratio_sq,
        'flux_by_flux_ratio_sq': df['flux'].values * flux_ratio_sq
        },
        index=df.index
    )

    return pd.concat([df, df_flux], axis=1)


@jit
def process_flux_agg(df):
    flux_w_mean = df['flux_by_flux_ratio_sq_sum'].values / df['flux_ratio_sq_sum'].values
    flux_diff = df['flux_max'].values - df['flux_min'].values

    df_flux_agg = pd.DataFrame({
        'flux_w_mean': flux_w_mean,
        'flux_diff1': flux_diff,
        'flux_diff2': flux_diff / df['flux_mean'].values,
        'flux_diff3': flux_diff / flux_w_mean,
        },
        index=df.index
    )

    return pd.concat([df, df_flux_agg], axis=1)


AGGS = {
    'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'detected': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'flux_ratio_sq': ['sum', 'skew'],
    'flux_by_flux_ratio_sq': ['sum', 'skew'],
}

# tsfresh features
FCP = {
    'flux': {
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None,
        'mean_change': None,
        'mean_abs_change': None,
        'length': None,
    },
    'flux_by_flux_ratio_sq': {
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None,
    },
    'flux_passband': {
        'fft_coefficient': [
            {'coeff': 0, 'attr': 'abs'},
            {'coeff': 1, 'attr': 'abs'}
        ],
        'kurtosis': None,
        'skewness': None,
    },
    'mjd': {
        'maximum': None,
        'minimum': None,
        'mean_change': None,
        'mean_abs_change': None,
    },
}


def featurize(df, df_meta, n_jobs=4):
    """
    Extracting Features from train set
    Features from olivier's kernel https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data
    Very smart and powerful feature that is generously given here https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    Per passband features with tsfresh library.
    fft features added to capture periodicity https://www.kaggle.com/c/PLAsTiCC-2018/discussion/70346#415506
    """

    df = process_flux(df)

    # agg features
    aggs = AGGS
    # tsfresh features
    fcp = FCP

    agg_df = df.groupby('object_id').agg(aggs)
    agg_df.columns = ['{}_{}'.format(k, agg) for k in aggs.keys() for agg in aggs[k]]
    agg_df = process_flux_agg(agg_df)  # new feature to play with tsfresh

    # Add more features with
    agg_df_ts_flux_passband = extract_features(df,
                                               column_id='object_id',
                                               column_sort='mjd',
                                               column_kind='passband',
                                               column_value='flux',
                                               default_fc_parameters=fcp['flux_passband'],
                                               n_jobs=n_jobs
                                               )

    agg_df_ts_flux = extract_features(df,
                                      column_id='object_id',
                                      column_value='flux',
                                      default_fc_parameters=fcp['flux'], n_jobs=n_jobs)

    agg_df_ts_flux_by_flux_ratio_sq = extract_features(df,
                                                       column_id='object_id',
                                                       column_value='flux_by_flux_ratio_sq',
                                                       default_fc_parameters=fcp['flux_by_flux_ratio_sq'],
                                                       n_jobs=n_jobs)

    # Add smart feature that is suggested here https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    # dt[detected==1, mjd_diff:=max(mjd)-min(mjd), by=object_id]
    df_det = df[df['detected'] == 1].copy()
    agg_df_mjd = extract_features(df_det,
                                  column_id='object_id',
                                  column_value='mjd',
                                  default_fc_parameters=fcp['mjd'], n_jobs=n_jobs)
    agg_df_mjd['mjd_diff_det'] = agg_df_mjd['mjd__maximum'].values - agg_df_mjd['mjd__minimum'].values
    del agg_df_mjd['mjd__maximum'], agg_df_mjd['mjd__minimum']

    agg_df_ts_flux_passband.index.rename('object_id', inplace=True)
    agg_df_ts_flux.index.rename('object_id', inplace=True)
    agg_df_ts_flux_by_flux_ratio_sq.index.rename('object_id', inplace=True)
    agg_df_mjd.index.rename('object_id', inplace=True)
    agg_df_ts = pd.concat([agg_df,
                           agg_df_ts_flux_passband,
                           agg_df_ts_flux,
                           agg_df_ts_flux_by_flux_ratio_sq,
                           agg_df_mjd], axis=1).reset_index()

    result = agg_df_ts.merge(right=df_meta, how='left', on='object_id')
    return result


def process_meta(filename):
    meta_df = pd.read_csv(filename)

    meta_dict = dict()
    # distance
    meta_dict.update(haversine_plus(meta_df['ra'].values, meta_df['decl'].values, meta_df['gal_l'].values, meta_df['gal_b'].values))

    meta_dict['hostgal_photoz_certain'] = np.multiply(
        meta_df['hostgal_photoz'].values,
        np.exp(meta_df['hostgal_photoz_err'].values))

    meta_df = pd.concat([meta_df, pd.DataFrame(meta_dict, index=meta_df.index)], axis=1)
    return meta_df

def multi_weighted_logloss(y_true, y_preds, classes, class_weights):
    """
    refactor from
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weights[k] for k in sorted(class_weights.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss


def lgbm_multi_weighted_logloss(y_true, y_preds):
    """
    refactor from
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # Taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    galactic_classes = [6, 16, 53, 65, 92]
    extragalactic_classes = [15, 42, 52, 62, 64, 67, 88, 90, 95]
    galactic_classes_weights = {c: 1 for c in galactic_classes}
    extragalactic_classes_weights = {c: 1 for c in extragalactic_classes}
    extragalactic_classes_weights.update({c: 2 for c in [64, 15]})
    if len(y_preds) == 5 * y_true.shape[0]:
        classes = galactic_classes
        class_weights = galactic_classes_weights
    else:
        classes = extragalactic_classes
        class_weights = extragalactic_classes_weights

    loss = multi_weighted_logloss(y_true, y_preds, classes, class_weights)
    return 'wloss', loss, False


def xgb_multi_weighted_logloss(y_predicted, y_true, classes, class_weights):
    loss = multi_weighted_logloss(y_true.get_label(), y_predicted, classes, class_weights)
    return 'wloss', loss


# In[ ]:


galactic_classes = [6, 16, 53, 65, 92]
extragalactic_classes = [15, 42, 52, 62, 64, 67, 88, 90, 95]

def predict_chunk(df_, clfs_gal_, clfs_ext_, meta_, features):
    # processing all features
    full_test = featurize(df_, meta_)
    full_test.fillna(0, inplace=True)

    galactic_cut = full_test['hostgal_photoz'] == 0
    gal_test = full_test[galactic_cut]
    ext_test = full_test[~galactic_cut]

    # predictions
    preds_gal = None
    if not gal_test.empty:
        for clf in clfs_gal_:
            if preds_gal is None:
                preds_gal = clf.predict_proba(gal_test[features])
            else:
                preds_gal += clf.predict_proba(gal_test[features])

        preds_gal = preds_gal / len(clfs_gal_)

        preds_99_gal = np.ones(preds_gal.shape[0])
        for i in range(preds_gal.shape[1]):
            preds_99_gal *= (1 - preds_gal[:, i])

        # Create DataFrame from predictions
        preds_gal = pd.DataFrame(preds_gal,
                                 columns=['class_{}'.format(s) for s in clfs_gal_[0].classes_])
        assert preds_gal.shape[0] == gal_test.shape[0], 'len of preds={}, test={}'.format(preds_gal.shape[0], gal_test.shape[0])
        preds_gal['object_id'] = gal_test['object_id'].values
        for c in ['class_{}'.format(s) for s in extragalactic_classes]:
            preds_gal.insert(0, c, 0.0)
        preds_gal['class_99'] = 0.017 * preds_99_gal / np.mean(preds_99_gal)

    preds_ext = None
    if not ext_test.empty:
        for clf in clfs_ext_:
            if preds_ext is None:
                preds_ext = clf.predict_proba(ext_test[features])
            else:
                preds_ext += clf.predict_proba(ext_test[features])

        preds_ext = preds_ext / len(clfs_ext_)

        preds_99_ext = np.ones(preds_ext.shape[0])
        for i in range(preds_ext.shape[1]):
            preds_99_ext *= (1 - preds_ext[:, i])

        # Create DataFrame from predictions
        preds_ext = pd.DataFrame(preds_ext,
                                 columns=['class_{}'.format(s) for s in clfs_ext_[0].classes_])
        assert preds_ext.shape[0] == ext_test.shape[0], 'len of preds={}, test={}'.format(preds_ext.shape[0], ext_test.shape[0])
        preds_ext['object_id'] = ext_test['object_id'].values
        for c in ['class_{}'.format(s) for s in galactic_classes]:
            preds_ext.insert(0, c, 0.0)
        preds_ext['class_99'] = 0.17 * preds_99_ext / np.mean(preds_99_ext)

    preds_df_ = pd.concat([preds_gal, preds_ext], ignore_index=True, sort=False)

    return preds_df_


def process_test(clfs_gal, clfs_ext,
                 features,
                 filename='predictions.csv',
                 chunks=5000000):
    start = time.time()

    meta_test = process_meta('../input/test_set_metadata.csv')

    remain_df = None
    for i_c, df in enumerate(pd.read_csv('../input/test_set.csv', chunksize=chunks, iterator=True)):
        # Check object_ids
        # I believe np.unique keeps the order of group_ids as they appear in the file
        unique_ids = np.unique(df['object_id'])

        new_remain_df = df.loc[df['object_id'] == unique_ids[-1]].copy()
        if remain_df is None:
            df = df.loc[df['object_id'].isin(unique_ids[:-1])]
        else:
            df = pd.concat([remain_df, df.loc[df['object_id'].isin(unique_ids[:-1])]], axis=0)
        # Create remaining samples df
        remain_df = new_remain_df

        preds_df = predict_chunk(df_=df,
                                 clfs_gal_=clfs_gal,
                                 clfs_ext_=clfs_ext,
                                 meta_=meta_test,
                                 features=features)

        if i_c == 0:
            preds_df.to_csv(filename, header=True, mode='a', index=False)
        else:
            preds_df.to_csv(filename, header=False, mode='a', index=False)

        del preds_df
        gc.collect()
        print('{:15d} done in {:5.1f} minutes'.format(
            chunks * (i_c + 1), (time.time() - start) / 60), flush=True)

    # Compute last object in remain_df
    preds_df = predict_chunk(df_=remain_df,
                             clfs_gal_=clfs_gal,
                             clfs_ext_=clfs_ext,
                             meta_=meta_test,
                             features=features)

    preds_df.to_csv(filename, header=False, mode='a', index=False)
    return



def lgbm_modeling_cross_validation(params,
                                   full_train,
                                   y,
                                   classes,
                                   class_weights,
                                   id,
                                   part,
                                   nr_fold=5,
                                   random_state=99,
                                   ):
    # Compute weights
    w = y.value_counts()
    weights = {i: np.sum(w) / w[i] for i in w.index}

    clfs = []
    importances = pd.DataFrame()
    folds = StratifiedKFold(n_splits=nr_fold,
                            shuffle=True,
                            random_state=random_state)

    oof_preds = np.zeros((len(full_train), np.unique(y).shape[0]))

    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]
        val_x, val_y = full_train.iloc[val_], y.iloc[val_]

        clf = LGBMClassifier(**params)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric=lgbm_multi_weighted_logloss,
            verbose=100,
            early_stopping_rounds=50,
            sample_weight=trn_y.map(weights)
        )
        clfs.append(clf)

        oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
        print('no {}-fold loss: {}'.format(fold_ + 1,
                                           multi_weighted_logloss(val_y, oof_preds[val_, :],
                                                                  classes, class_weights)))

        imp_df = pd.DataFrame({
            'feature': full_train.columns,
            'gain': clf.feature_importances_,
            'fold': [fold_ + 1] * len(full_train.columns),
        })
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

    score = multi_weighted_logloss(y_true=y, y_preds=oof_preds,
                                   classes=classes, class_weights=class_weights)
    print('MULTI WEIGHTED LOG LOSS: {:.5f}'.format(score))

    oof_preds_pd = pd.DataFrame(data=oof_preds, columns=['class_{}'.format(s) for s in classes])
    pd.concat([id, oof_preds_pd], axis=1).to_csv('lgbm_train_oof_preds_{}.csv'.format(part), index=False)
    df_importances = save_importances(importances_=importances)
    df_importances.to_csv('lgbm_importances_{}.csv'.format(part), index=False)

    return clfs, score

def save_importances(importances_):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    return importances_


# In[ ]:


xgb_params = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'silent': True,
    'num_class': 14,

    'booster': 'gbtree',
    'n_jobs': 4,
    'n_estimators': 1000,
    'tree_method': 'hist',
    'grow_policy': 'lossguide',
    'base_score': 0.25,
    'max_depth': 7,
    'max_delta_step': 2,  # default=0
    'learning_rate': 0.03,
    'max_leaves': 11,
    'min_child_weight': 64,
    'gamma': 0.1,  # default=
    'subsample': 0.7,
    'colsample_bytree': 0.68,
    'reg_alpha': 0.01,  # default=0
    'reg_lambda': 10.,  # default=1
    'seed': 1234

}

lgbm_params = {
    'device': 'cpu',
    'objective': 'multiclass',
    'num_class': 14,
    'boosting_type': 'gbdt',
    'n_jobs': -1,
    'max_depth': 7,
    'n_estimators': 500,
    'subsample_freq': 2,
    'subsample_for_bin': 5000,
    'min_data_per_group': 100,
    'max_cat_to_onehot': 4,
    'cat_l2': 1.0,
    'cat_smooth': 59.5,
    'max_cat_threshold': 32,
    'metric_freq': 10,
    'verbosity': -1,
    'metric': 'multi_logloss',
    'xgboost_dart_mode': False,
    'uniform_drop': False,
    'colsample_bytree': 0.5,
    'drop_rate': 0.173,
    'learning_rate': 0.0267,
    'max_drop': 5,
    'min_child_samples': 10,
    'min_child_weight': 100.0,
    'min_split_gain': 0.1,
    'num_leaves': 7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.00023,
    'skip_drop': 0.44,
    'subsample': 0.75
}


# In[ ]:


def train_model(full_train, classes, class_weights, part):
    if 'target' in full_train:
        y = full_train['target']
        del full_train['target']

    if 'object_id' in full_train:
        object_id = full_train['object_id']
        del full_train['object_id']
        del full_train['hostgal_specz']
        del full_train['ra'], full_train['decl'], full_train['gal_l'], full_train['gal_b']
        del full_train['ddf']

    eval_func = partial(lgbm_modeling_cross_validation,
                        full_train=full_train,
                        y=y,
                        classes=classes,
                        class_weights=class_weights,
                        id=object_id,
                        part=part,
                        nr_fold=5,
                        random_state=99,
                        )

    lgbm_params.update({'n_estimators': 2000})

    # modeling from CV
    clfs, score = eval_func(lgbm_params)

    return clfs, score


# In[ ]:


def main(argc, argv):
    meta_train = process_meta('../input/training_set_metadata.csv')

    train = pd.read_csv('../input/training_set.csv')
    full_train = featurize(train, meta_train)

    # Taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    galactic_classes_weights = {c: 1 for c in galactic_classes}
    extragalactic_classes_weights = {c: 1 for c in extragalactic_classes}
    extragalactic_classes_weights.update({c: 2 for c in [64, 15]})

    full_train.fillna(0, inplace=True)
    galactic_cut = full_train['hostgal_photoz'] == 0

    clfs_gal, score_gal = train_model(full_train[galactic_cut],
                            galactic_classes, galactic_classes_weights, 'gal')
    clfs_ext, score_ext = train_model(full_train[~galactic_cut],
                            extragalactic_classes, extragalactic_classes_weights, 'ext')

    filename = 'subm_{:.6f}_{:.6f}_{}.csv'.format(
            score_gal, score_ext,
            datetime.now().strftime('%Y-%m-%d-%H-%M'))
    print('save to {}'.format(filename))

    # TEST
    if 'target' in full_train:
        y = full_train['target']
        del full_train['target']

    if 'object_id' in full_train:
        object_id = full_train['object_id']
        del full_train['object_id']
        del full_train['hostgal_specz']
        del full_train['ra'], full_train['decl'], full_train['gal_l'], full_train['gal_b']
        del full_train['ddf']
    process_test(clfs_gal, clfs_ext,
                 features=full_train.columns,
                 filename=filename,
                 chunks=5000000)

    z = pd.read_csv(filename)
    print("Shape BEFORE grouping: {}".format(z.shape))
    z = z.groupby('object_id').mean()
    print("Shape AFTER grouping: {}".format(z.shape))
    z.to_csv('single_{}'.format(filename), index=True)


# In[ ]:


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)


# In[ ]:




