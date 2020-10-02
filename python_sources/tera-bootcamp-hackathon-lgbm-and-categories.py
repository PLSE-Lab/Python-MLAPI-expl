#!/usr/bin/env python
# coding: utf-8

# __IMPORTING__

# In[ ]:


import pandas as pd
import numpy as np
import shap
from tqdm import tqdm

from lightgbm import LGBMClassifier, plot_metric, plot_tree, create_tree_digraph
from sklearn.model_selection import GridSearchCV

from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (mean_squared_error, accuracy_score)

from sklearn.model_selection import train_test_split
import shap
import random

#from plotting import (multiple_histograms_plot, plot_confusion_matrix, plot_roc)
#from evaluation import predictions_hist, confusion_matrix_report, grid_search_report
#from auto_lgbm import find_n_estimators, grid_search


# In[ ]:


total_lines = 8921484
skip_rows = np.arange(1, total_lines, 1)
skip_rows = random.sample(list(skip_rows), total_lines - 1000000)
df_orig = pd.read_csv("../input/microsoft-malware-prediction/train.csv", skiprows= skip_rows) 


# __EDA__
# 
# For EDA we analysed feature by feature, researching their meanings and selecting the ones appeared to have a higher impact on a malware infection.

# __FEATURE ENGINEERING__

# In[ ]:


df = df_orig.drop(['Census_FlightRing',
                  'Census_PrimaryDiskTypeName',
                  'Census_IsFlightingInternal',
                  'GeoNameIdentifier',
                  'Census_MDC2FormFactor',
                  'OsPlatformSubRelease',
                  'Census_ActivationChannel',
                  'Census_OSSkuName',
                  'Census_OSBranch',
                  'Census_OSArchitecture',
                  'Census_InternalBatteryNumberOfCharges',
                  'Census_PowerPlatformRoleName',
                  'OsBuildLab',
                  'MachineIdentifier',
                  'PuaMode', 
                  'Census_ProcessorClass', 
                  'DefaultBrowsersIdentifier', 
                  'Census_InternalBatteryType'], axis=1)


# In[ ]:


pd.set_option('display.max_columns',100)
df.head()


# In[ ]:


df.dropna()


# In[ ]:


df.loc[:,'AppVersion'] = df['AppVersion'].astype('category')
df.loc[:,'AvSigVersion'] = df['AvSigVersion'].astype('category')
df.loc[:,'EngineVersion'] = df['EngineVersion'].astype('category')
df.loc[:,'UacLuaenable'] = df['UacLuaenable'].astype('category')
df.loc[:,'ProductName'] = df['ProductName'].astype('category')
df.loc[:,'Processor'] = df['Processor'].astype('category')
df.loc[:,'Platform'] = df['Platform'].astype('category')
df.loc[:,'SkuEdition'] = df['SkuEdition'].astype('category')
df.loc[:,'LocaleEnglishNameIdentifier'] = df['LocaleEnglishNameIdentifier'].astype('category')
df.loc[:,'OrganizationIdentifier'] = df['OrganizationIdentifier'].astype('category')
df.loc[:,'CityIdentifier'] = df['CityIdentifier'].astype('category')
df.loc[:,'SmartScreen'] = df['SmartScreen'].astype('category')
df.loc[:,'Census_DeviceFamily'] = df['Census_DeviceFamily'].astype('category')
df.loc[:,'Census_OSInstallTypeName'] = df['Census_OSInstallTypeName'].astype('category')
df.loc[:,'Census_OSEdition'] = df['Census_OSEdition'].astype('category')
df.loc[:,'OsVer'] = df['OsVer'].astype('category')
df.loc[:,'Census_OSVersion'] = df['Census_OSVersion'].astype('category')
df.loc[:,'Census_OEMNameIdentifier'] = df['Census_OEMNameIdentifier'].astype('category')
df.loc[:,'Census_OEMModelIdentifier'] = df['Census_OEMModelIdentifier'].astype('category')
df.loc[:,'Census_ProcessorManufacturerIdentifier'] = df['Census_ProcessorManufacturerIdentifier'].astype('category')
df.loc[:,'Census_ProcessorModelIdentifier'] = df['Census_ProcessorModelIdentifier'].astype('category')
df.loc[:,'Census_ChassisTypeName'] = df['Census_ChassisTypeName'].astype('category')
df.loc[:,'Census_OSBuildNumber'] = df['Census_OSBuildNumber'].astype('category')
df.loc[:,'Census_OSInstallLanguageIdentifier'] = df['Census_OSInstallLanguageIdentifier'].astype('category')
df.loc[:,'Census_OSUILocaleIdentifier'] = df['Census_OSUILocaleIdentifier'].astype('category')
df.loc[:,'Census_FirmwareManufacturerIdentifier'] = df['Census_FirmwareManufacturerIdentifier'].astype('category')
df.loc[:,'Census_FirmwareVersionIdentifier'] = df['Census_FirmwareVersionIdentifier'].astype('category')
df.loc[:,'CountryIdentifier'] = df['CountryIdentifier'].astype('category')


# In[ ]:


le = LabelEncoder()
grade_intlabels = le.fit_transform(df['Census_OSWUAutoUpdateOptionsName'])
df.loc[:,'Census_OSWUAutoUpdateOptionsName'] = grade_intlabels


# In[ ]:


le = LabelEncoder()
grade_intlabels = le.fit_transform(df['Census_GenuineStateName'])
df.loc[:,'Census_GenuineStateName'] = grade_intlabels


# In[ ]:


pd.set_option('display.max_columns',100)
df.head()


# __TRAINING THE MODEL__

# In[ ]:


X = df.drop(columns=['HasDetections'])
y = df['HasDetections']


# In[ ]:


X_cat = df.drop(columns=['HasDetections'])

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y, test_size=0.2, 
                                                                    random_state=0)


# In[ ]:


validation_size = 0.25

X_dev, X_val, y_dev, y_val = train_test_split(X_train_cat, y_train_cat, 
                                              test_size=validation_size, 
                                              random_state=0)


# In[ ]:


lgbm = LGBMClassifier(n_estimators=5000,
                         class_weight='balanced', random_state=0)


# In[ ]:


_ = lgbm.fit(X_dev, y_dev, eval_set=(X_val, y_val), early_stopping_rounds = 50)


# In[ ]:


y_pred_proba = lgbm.predict_proba(X_test_cat)[:, 1]

confusion_matrix_report(y_test_cat, y_pred_proba, thres=0.625)
# In[ ]:


_ = plot_roc(y_test_cat, y_pred_proba)


# In[ ]:


max_n_estimators = 5000
early_stopping_rounds = 50
learning_rates = [0.01, 0.03, 0.1]


# In[ ]:


lgbm_lr = LGBMClassifier(n_estimators=max_n_estimators, 
                         class_weight='balanced', random_state=0)


# In[ ]:


results = pd.DataFrame(columns=['learning_rate', 'best_n_estimators', 'best_log_loss'])

for learning_rate in tqdm(learning_rates):
    lgbm_lr.learning_rate = learning_rate
    
    lgbm_lr.fit(X_dev, y_dev, eval_set=(X_val, y_val), 
                early_stopping_rounds=early_stopping_rounds, verbose=False)
    
    results = results.append({'learning_rate': learning_rate, 
                              'best_n_estimators': lgbm_lr.best_iteration_,
                              'best_log_loss': lgbm_lr.best_score_['valid_0']['binary_logloss']},
                             ignore_index=True)


# In[ ]:


results['learning_rate'] = results['learning_rate']
results['best_n_estimators'] = results['best_n_estimators'].astype(int)

results


# In[ ]:


param_grids = [
    {'learning_rate': [0.01], 'n_estimators': [760]},
    {'learning_rate': [0.03], 'n_estimators': [276]},
    {'learning_rate': [0.1], 'n_estimators': [78]}
]


# In[ ]:


scoring = 'neg_log_loss'


# In[ ]:


grid_search_cv = GridSearchCV(lgbm_lr, param_grid = param_grids, scoring = scoring,
                             cv=5, verbose=2)


# In[ ]:


grid_search_cv.fit(X_train_cat, y_train_cat, verbose=False)


# __ANALYSING THE MODEL__

# In[ ]:


y_pred_proba_cv = grid_search_cv.predict_proba(X_test_cat)[:, 1]


# In[ ]:


confusion_matrix_report(y_test_cat, y_pred_proba_cv, thres=0.625)


# In[ ]:


_ = plot_roc(y_test_cat, y_pred_proba_cv)


# In[ ]:


X_train_sample = X_train_cat.sample(1_000, random_state=0)


# In[ ]:


explainer = shap.TreeExplainer(model=lgbm).shap_values(X_train_sample)


# In[ ]:


shap.summary_plot(explainer, X_train_sample, plot_type='bar')


# __IMPORTING TEST DATASET__

# In[ ]:


df_test = pd.read_csv("../input/microsoft-malware-prediction/test.csv")


# __FEATURE ENGINEERING TO TEST DATASET__

# In[ ]:


df_predict = df_test.drop(['Census_FlightRing',
                   'Census_PrimaryDiskTypeName',
                   'Census_IsFlightingInternal',
                   'GeoNameIdentifier',
                   'Census_MDC2FormFactor',
                   'OsPlatformSubRelease',
                   'Census_ActivationChannel',
                   'Census_OSSkuName',
                  'Census_OSBranch',
                   'Census_OSArchitecture',
                   'Census_InternalBatteryNumberOfCharges',
                   'Census_PowerPlatformRoleName',
                   'OsBuildLab',
                   'MachineIdentifier',
                   'PuaMode', 
                   'Census_ProcessorClass', 
                   'DefaultBrowsersIdentifier', 
                   'Census_InternalBatteryType'], axis=1)


# In[ ]:


df_predict.loc[:,'AppVersion'] = df['AppVersion'].astype('category')
df_predict.loc[:,'AvSigVersion'] = df['AvSigVersion'].astype('category')
df_predict.loc[:,'EngineVersion'] = df['EngineVersion'].astype('category')
df_predict.loc[:,'UacLuaenable'] = df['UacLuaenable'].astype('category')
df_predict.loc[:,'ProductName'] = df['ProductName'].astype('category')
df_predict.loc[:,'Processor'] = df['Processor'].astype('category')
df_predict.loc[:,'Platform'] = df['Platform'].astype('category')
df_predict.loc[:,'SkuEdition'] = df['SkuEdition'].astype('category')
df_predict.loc[:,'LocaleEnglishNameIdentifier'] = df['LocaleEnglishNameIdentifier'].astype('category')
df_predict.loc[:,'OrganizationIdentifier'] = df['OrganizationIdentifier'].astype('category')
df_predict.loc[:,'CityIdentifier'] = df['CityIdentifier'].astype('category')
df_predict.loc[:,'SmartScreen'] = df['SmartScreen'].astype('category')
df_predict.loc[:,'Census_DeviceFamily'] = df['Census_DeviceFamily'].astype('category')
df_predict.loc[:,'Census_OSInstallTypeName'] = df['Census_OSInstallTypeName'].astype('category')
df_predict.loc[:,'Census_OSEdition'] = df['Census_OSEdition'].astype('category')
df_predict.loc[:,'OsVer'] = df['OsVer'].astype('category')
df_predict.loc[:,'Census_OSVersion'] = df['Census_OSVersion'].astype('category')
df_predict.loc[:,'Census_OEMNameIdentifier'] = df['Census_OEMNameIdentifier'].astype('category')
df_predict.loc[:,'Census_OEMModelIdentifier'] = df['Census_OEMModelIdentifier'].astype('category')
df_predict.loc[:,'Census_ProcessorManufacturerIdentifier'] = df['Census_ProcessorManufacturerIdentifier'].astype('category')
df_predict.loc[:,'Census_ProcessorModelIdentifier'] = df['Census_ProcessorModelIdentifier'].astype('category')
df_predict.loc[:,'Census_ChassisTypeName'] = df['Census_ChassisTypeName'].astype('category')
df_predict.loc[:,'Census_OSBuildNumber'] = df['Census_OSBuildNumber'].astype('category')
df_predict.loc[:,'Census_OSInstallLanguageIdentifier'] = df['Census_OSInstallLanguageIdentifier'].astype('category')
df_predict.loc[:,'Census_OSUILocaleIdentifier'] = df['Census_OSUILocaleIdentifier'].astype('category')
df_predict.loc[:,'Census_FirmwareManufacturerIdentifier'] = df['Census_FirmwareManufacturerIdentifier'].astype('category')
df_predict.loc[:,'Census_FirmwareVersionIdentifier'] = df['Census_FirmwareVersionIdentifier'].astype('category')
df_predict.loc[:,'CountryIdentifier'] = df['CountryIdentifier'].astype('category')


# In[ ]:


le = LabelEncoder()
grade_intlabels = le.fit_transform(df['Census_OSWUAutoUpdateOptionsName'])
df_predict.loc[:,'Census_OSWUAutoUpdateOptionsName'] = grade_intlabels


# In[ ]:


le = LabelEncoder()
grade_intlabels = le.fit_transform(df['Census_GenuineStateName'])
df.loc[:,'Census_GenuineStateName'] = grade_intlabels


# __PREDICTING__

# In[ ]:


y_pred = grid_search_cv.predict(X_test_cat)

print(accuracy_score(y_test_cat, y_pred)*100)


# __MAKING A DATASET WITH PREDICIONS__

# In[ ]:


df_testid = pd.read_csv('../input/microsoft-malware-prediction/test.csv',usecols=[0])


# In[ ]:


df_pred = pd.DataFrame(y_pred)


# In[ ]:


df_submission = pd.concat([df_testid, df_pred], axis=1, join='inner', ignore_index=False, keys=None,
          levels=None, names=None, verify_integrity=False, copy=True)


# In[ ]:


df_submission[0]

df_tudo.set_axis(['id', 'inf'], axis=1, inplace= True)


# In[ ]:


len(df_submission[df_submission['inf'] == 1]['id'])


# In[ ]:


df_submission[df_submission['inf'] == 1]['id']


# In[ ]:


df_submission.to_csv('submission.csv', index=False)

