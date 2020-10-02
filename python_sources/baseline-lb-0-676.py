# -*- coding:utf8 -*-
# demo version: 1.0
# 

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn import metrics
import pandas as pd
import numpy as np
import gc
import time
import sys
import datetime
import warnings
warnings.filterwarnings("ignore")



# after carefully invistigation, I have updated the following dtypes
# identifiers are probably be categorical data
# also, one may convert `Census_InternalBatteryNumberOfCharges` using some techs like `log1p`
# to generate its alternative variable `Census_InternalBatteryNumberOfCharges_log1p`
#
dtypes = {
    'MachineIdentifier': 'category',
    'ProductName': 'category',
    'EngineVersion': 'category',
    'AppVersion': 'category',
    'AvSigVersion': 'category',
    'IsBeta': 'int8',
    'RtpStateBitfield': 'category',
    'IsSxsPassiveMode': 'int8',
    'DefaultBrowsersIdentifier': 'float16',
    'AVProductStatesIdentifier': 'float32',
    'AVProductsInstalled': 'float16',
    'AVProductsEnabled': 'float16',
    'HasTpm': 'int8',
    'CountryIdentifier': 'category',
    'CityIdentifier': 'category',
    'OrganizationIdentifier': 'category',
    'GeoNameIdentifier': 'category',
    'LocaleEnglishNameIdentifier': 'category',
    'Platform': 'category',
    'Processor': 'category',
    'OsVer': 'category',
    'OsBuild': 'category',
    'OsSuite': 'category',
    'OsPlatformSubRelease': 'category',
    'OsBuildLab': 'category',
    'SkuEdition': 'category',
    'IsProtected': 'category',
    'AutoSampleOptIn': 'int8',
    'PuaMode': 'category',
    'SMode': 'float16',
    'IeVerIdentifier': 'float16',
    'SmartScreen': 'category',
    'Firewall': 'float16',
    'UacLuaenable': 'float32',
    'Census_MDC2FormFactor': 'category',
    'Census_DeviceFamily': 'category',
    'Census_OEMNameIdentifier': 'float16',
    'Census_OEMModelIdentifier': 'float32',
    'Census_ProcessorCoreCount': 'float16',
    'Census_ProcessorManufacturerIdentifier': 'float16',
    'Census_ProcessorModelIdentifier': 'float16',
    'Census_ProcessorClass': 'category',
    'Census_PrimaryDiskTotalCapacity': 'float32',
    'Census_PrimaryDiskTypeName': 'category',
    'Census_SystemVolumeTotalCapacity': 'float32',
    'Census_HasOpticalDiskDrive': 'int8',
    'Census_TotalPhysicalRAM': 'float32',
    'Census_ChassisTypeName': 'category',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches': 'float16',
    'Census_InternalPrimaryDisplayResolutionHorizontal': 'float16',
    'Census_InternalPrimaryDisplayResolutionVertical': 'float16',
    'Census_PowerPlatformRoleName': 'category',
    'Census_InternalBatteryType': 'category',
    'Census_InternalBatteryNumberOfCharges': 'float32',
    'Census_OSVersion': 'category',
    'Census_OSArchitecture': 'category',
    'Census_OSBranch': 'category',
    'Census_OSBuildNumber': 'int16',
    'Census_OSBuildRevision': 'int32',
    'Census_OSEdition': 'category',
    'Census_OSSkuName': 'category',
    'Census_OSInstallTypeName': 'category',
    'Census_OSInstallLanguageIdentifier': 'float16',
    'Census_OSUILocaleIdentifier': 'int16',
    'Census_OSWUAutoUpdateOptionsName': 'category',
    'Census_IsPortableOperatingSystem': 'int8',
    'Census_GenuineStateName': 'category',
    'Census_ActivationChannel': 'category',
    'Census_IsFlightingInternal': 'float16',
    'Census_IsFlightsDisabled': 'float16',
    'Census_FlightRing': 'category',
    'Census_ThresholdOptIn': 'float16',
    'Census_FirmwareManufacturerIdentifier': 'float16',
    'Census_FirmwareVersionIdentifier': 'float32',
    'Census_IsSecureBootEnabled': 'int8',
    'Census_IsWIMBootEnabled': 'float16',
    'Census_IsVirtualDevice': 'float16',
    'Census_IsTouchEnabled': 'int8',
    'Census_IsPenCapable': 'int8',
    'Census_IsAlwaysOnAlwaysConnectedCapable': 'float16',
    'Wdft_IsGamer': 'float16',
    'Wdft_RegionIdentifier': 'float16',
    'HasDetections': 'int8'
}

COLS = 'MachineIdentifier,ProductName,EngineVersion,AppVersion,AvSigVersion,IsBeta,RtpStateBitfield,IsSxsPassiveMode,DefaultBrowsersIdentifier,AVProductStatesIdentifier,AVProductsInstalled,AVProductsEnabled,HasTpm,CountryIdentifier,CityIdentifier,OrganizationIdentifier,GeoNameIdentifier,LocaleEnglishNameIdentifier,Platform,Processor,OsVer,OsBuild,OsSuite,OsPlatformSubRelease,OsBuildLab,SkuEdition,IsProtected,AutoSampleOptIn,PuaMode,SMode,IeVerIdentifier,SmartScreen,Firewall,UacLuaenable,Census_MDC2FormFactor,Census_DeviceFamily,Census_OEMNameIdentifier,Census_OEMModelIdentifier,Census_ProcessorCoreCount,Census_ProcessorManufacturerIdentifier,Census_ProcessorModelIdentifier,Census_ProcessorClass,Census_PrimaryDiskTotalCapacity,Census_PrimaryDiskTypeName,Census_SystemVolumeTotalCapacity,Census_HasOpticalDiskDrive,Census_TotalPhysicalRAM,Census_ChassisTypeName,Census_InternalPrimaryDiagonalDisplaySizeInInches,Census_InternalPrimaryDisplayResolutionHorizontal,Census_InternalPrimaryDisplayResolutionVertical,Census_PowerPlatformRoleName,Census_InternalBatteryType,Census_InternalBatteryNumberOfCharges,Census_OSVersion,Census_OSArchitecture,Census_OSBranch,Census_OSBuildNumber,Census_OSBuildRevision,Census_OSEdition,Census_OSSkuName,Census_OSInstallTypeName,Census_OSInstallLanguageIdentifier,Census_OSUILocaleIdentifier,Census_OSWUAutoUpdateOptionsName,Census_IsPortableOperatingSystem,Census_GenuineStateName,Census_ActivationChannel,Census_IsFlightingInternal,Census_IsFlightsDisabled,Census_FlightRing,Census_ThresholdOptIn,Census_FirmwareManufacturerIdentifier,Census_FirmwareVersionIdentifier,Census_IsSecureBootEnabled,Census_IsWIMBootEnabled,Census_IsVirtualDevice,Census_IsTouchEnabled,Census_IsPenCapable,Census_IsAlwaysOnAlwaysConnectedCapable,Wdft_IsGamer,Wdft_RegionIdentifier,HasDetections'.split(',')

N = 4000000
TRAINFILE='../input/train.csv'
TESTFILE='../input/test.csv'

# load data
df_train = pd.read_csv(TRAINFILE, dtype=dtypes, nrows=N, skiprows=1, header=None, names=COLS)
# df_test = pd.read_csv(TESTFILE, dtype=dtypes, skiprows=1, header=None, names=COLS[:-1])

print(df_train.shape)

# model buiding
numeric_types = ['int8', 'int16', 'int32', 'float16', 'float32']
categorical_columns = [k for k, v in dtypes.items() if v == 'category']

df_train['Census_InternalBatteryNumberOfCharges_log1p'] = df_train.Census_InternalBatteryNumberOfCharges.apply(pd.np.log1p)
df_train.drop(columns='Census_InternalBatteryNumberOfCharges', inplace=True)

# df_test['Census_InternalBatteryNumberOfCharges_log1p'] = df_test.Census_InternalBatteryNumberOfCharges.apply(pd.np.log1p)
# df_test.drop(columns='Census_InternalBatteryNumberOfCharges', inplace=True)


# settings
param = {
    'num_leaves': 60,
    'min_data_in_leaf': 60,
    'objective': 'binary',
    'max_depth': -1,
    'learning_rate': 0.1,
    "boosting": "gbdt",
    "feature_fraction": 0.8,
    "bagging_freq": 1,
    "bagging_fraction": 0.8,
    "bagging_seed": 11,
    "metric": 'auc',
    "lambda_l1": 0.1,
    "random_state": 13,
    "verbosity": -1
}
max_iter = 5
folds = KFold(n_splits=5, shuffle=True, random_state=10)
oof = np.zeros(df_train.shape[0])

columns = df_train.columns[:-2].tolist() + ['Census_InternalBatteryNumberOfCharges_log1p']
target = df_train['HasDetections']
df_train.drop(columns='HasDetections',inplace=True)
gc.collect()


features = columns
results = []
mids = []
# predictions = None
df_results = pd.DataFrame()
feature_importance_df = pd.DataFrame()

#
start_time = time.time()
score = [0 for _ in range(folds.n_splits)]

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, target.values)):
    print("fold #{}".format(fold_))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][features],
                           label=target.iloc[trn_idx],
                           categorical_feature=categorical_columns
                           )
    val_data = lgb.Dataset(df_train.iloc[val_idx][features],
                           label=target.iloc[val_idx],
                           categorical_feature=categorical_columns
                           )

    num_round = 600
    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=100,
                    early_stopping_rounds=300)

    oof[val_idx] = clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    #
    bt = time.clock()
    for df_test in pd.read_csv(TESTFILE, sep=',', skiprows=1, chunksize=100000, header=None, dtype=dtypes, names=COLS[:-1]):
        # df_results = pd.concat([df_results, df_test[['MachineIdentifier']]])
        if fold_==0:
            [mids.append(_id) for _id in df_test['MachineIdentifier'].values if _id not in mids]
        df_test['Census_InternalBatteryNumberOfCharges_log1p'] = df_test.Census_InternalBatteryNumberOfCharges.apply(pd.np.log1p)
        results.append(clf.predict(df_test[features], num_iteration=clf.best_iteration) / min(folds.n_splits, max_iter))
        # break
    if fold_==0:
        predictions = np.hstack(results)
    else:
        predictions += np.hstack(results)
    results = []

    print("time elapsed: {}s".format((time.time() - start_time)))
    score[fold_] = metrics.roc_auc_score(target.iloc[val_idx], oof[val_idx])
    if fold_ == max_iter - 1:
        break

if (folds.n_splits == max_iter):
    print("CV score: {:<8.5f}".format(metrics.roc_auc_score(target, oof)))
else:
    print("CV score: {:<8.5f}".format(sum(score) / max_iter / (max_iter/folds.n_splits)))

# importance
cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
best_features.to_csv('best_feats_demo.csv', index=False)

print('saving predictions ...')
with open('submission_base.csv', 'w') as f:
        f.write('{},{}\n'.format('MachineIdentifier', 'HasDetections'))
        for mid, score in zip(mids, predictions):
            f.write('{},{}\n'.format(mid, score))
            
# sub_df = pd.DataFrame({"MachineIdentifier": machineIdds['MachineIdentifier'].values})
# df_results["HasDetections"] = predictions
# print(df_results[:10])
# df_results.to_csv('submission_base_demo.csv', index=False)
