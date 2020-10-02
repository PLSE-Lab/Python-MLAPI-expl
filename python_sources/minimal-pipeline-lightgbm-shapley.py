#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import warnings
warnings.simplefilter("ignore")

import os
print(os.listdir("../input"))


# In[ ]:


# Using tip from Shruti_Iyyer kernel
# Large Data Loading Trick with MS-Malware data
# https://www.kaggle.com/shrutimechlearn/large-data-loading-trick-with-ms-malware-data

dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int8',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }


# In[ ]:


target="HasDetections"
submission_id_col="MachineIdentifier"

seed_split=1 
test_size=1/3
seed_train=100


# In[ ]:


top_features=["SmartScreen","AVProductsInstalled","EngineVersion","AVProductStatesIdentifier","Census_TotalPhysicalRAM","Wdft_IsGamer",
"Census_PrimaryDiskTotalCapacity","AppVersion","Census_InternalPrimaryDiagonalDisplaySizeInInches","Census_OSInstallTypeName"]


# In[ ]:


import random

df_kaggle_train = pd.read_csv(
         '../input/train.csv',
         dtype=dtypes,
         usecols=top_features+[submission_id_col,target]
)


# In[ ]:


df_kaggle_train.shape


# In[ ]:


df_kaggle_test = pd.read_csv('../input/test.csv', dtype=dtypes,usecols=top_features+[submission_id_col])


# In[ ]:


df_kaggle_test.shape


# In[ ]:


from sklearn.model_selection import train_test_split

# Split X,y
y= df_kaggle_train[target].values
df_kaggle_train.drop(columns=target,inplace=True)

# Split kaggle train, reserve internal hold out test set
X_train, X_test, y_train,y_test = train_test_split(df_kaggle_train,y, 
                                                   test_size=test_size, random_state=seed_split,stratify =y)


# In[ ]:


X_train.drop(columns=["MachineIdentifier"],inplace=True)


# In[ ]:


from  sklearn.impute import SimpleImputer
from sklearn_pandas import DataFrameMapper
import sklearn.preprocessing as pp

nums=[ ([c],[SimpleImputer(strategy="constant",fill_value=0)]) for c in X_train.select_dtypes(include=[np.number])]
cats=[ ([c],[SimpleImputer(fill_value="NA",strategy="constant"), 
             pp.OrdinalEncoder(categories=[list(
                 set(list(["NA"])+list(df_kaggle_train[c].unique())+list(df_kaggle_test[c].unique()))
                 #set(list(["NA"])+list(df_kaggle_train[c].unique()))
             )])
            ]) for c in X_train.select_dtypes(include=["category"])]
mapper=DataFrameMapper(nums+cats,df_out=True)


# In[ ]:


from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

pipeline=Pipeline([('featurize', mapper),("clf",LGBMClassifier(random_state=seed_train))])


# In[ ]:


pipeline.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import roc_auc_score

y_pred_train=pipeline.predict_proba(X_train)[:,1]
print("train score",roc_auc_score(y_score=y_pred_train,y_true=y_train))


# In[ ]:


y_pred_test=pipeline.predict_proba(X_test)[:,1]
print("test score",roc_auc_score(y_score=y_pred_test,y_true=y_test))


# In[ ]:


import shap

N_ROWS=10000
feature_names=mapper.transformed_names_
X_explanation_rows=pipeline.named_steps["featurize"].transform(X_train.sample(N_ROWS))

# use fast shapley tree explainer
shap_explainer=shap.TreeExplainer(pipeline.named_steps["clf"])
shap_values = shap_explainer.shap_values(X_explanation_rows)


# In[ ]:


df_shap_values=pd.DataFrame(shap_values,columns=feature_names)
feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0)[:-1])
feature_order = feature_order[-min(len(X_explanation_rows.columns), len(feature_order)):]
shap_feature_rank=X_explanation_rows.columns[list(reversed(feature_order))]
df_shap_importance=pd.DataFrame(dict(feature=shap_feature_rank,shap_rank=range(1,len(shap_feature_rank)+1)))
df_shap_importance


# In[ ]:


shap.summary_plot(shap_values, X_explanation_rows,feature_names=feature_names,max_display=100)


# In[ ]:


TOP_FEATURES=50
for i_feature in df_shap_importance.feature[:TOP_FEATURES]:
    shap.dependence_plot(i_feature,shap_values, X_explanation_rows)


# In[ ]:


# Full train fit
pipeline.fit(df_kaggle_train,y)


# In[ ]:


full_pipeline=pipeline
y_pred_submission=full_pipeline.predict_proba(df_kaggle_test)[:,1]


# In[ ]:


# Prepare submission
df_submission=pd.DataFrame({submission_id_col:df_kaggle_test[submission_id_col],target:y_pred_submission})
df_submission.head()


# In[ ]:


# Check predictions 
df_submission[target].hist()
print("y mean:",np.mean(y))
print("y submission mean:",df_submission[target].mean())


# In[ ]:


df_submission.to_csv(f"submission.csv",index=False)
print("Done!")


# In[ ]:




