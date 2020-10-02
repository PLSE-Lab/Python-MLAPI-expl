#!/usr/bin/env python
# coding: utf-8

# ### **Purpose of this notebook** : To gain model insights
# ---
# ### **Methods used to serve the purpose** : Permutation Importance, Partial Plots and SHAP values
# ---
# ### **Prerequisites** : Since a model is necessary for doing the above, we'll quickly do feature engineering and train a model based on [this](https://www.kaggle.com/harmeggels/random-forest-feature-importances/notebook) popular notebook. Credits to this notebook for useful information.
# ----

# ### **Notebook contents** 
# 
# * [Necessary Imports](#0)
# * [Load Data and Feature Engineering](#1)
# * [Train Model](#2)
# * [**Derive Model Insights**](#3)
#     * [**Permutation Importance**](#3.1)
#     * [**Partial Plots**](#3.2)
#     * [**SHAP Values**](#3.3)

# ### **Please upvote if you find this kernel useful**. 

# <a id="0"></a> <br>
# ## **Necessary Imports**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, forest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
import scipy
import re

# Permutation Importance
import eli5
from eli5.sklearn import PermutationImportance

# Partial Plots
from pdpbox import pdp, get_dataset, info_plots

# Package used to calculate SHAP Values
import shap


# <a id="1"></a> <br>
# ## **Load Data and Feature Engineering**

# As explained in [this](https://www.kaggle.com/theoviel/load-the-totality-of-the-data) notebook, we set the data types(example : switching from float64 to float32) of variables/features to reduce memory usage

# In[ ]:


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

get_ipython().run_line_magic('time', "train = pd.read_csv('../input/train.csv', nrows=1000000, usecols=dtypes.keys(), low_memory=False)")

#display(train.describe(include='all').T)


# In[ ]:


col = ['EngineVersion', 'AppVersion', 'AvSigVersion', 'OsBuildLab', 'Census_OSVersion']
for c in col:
    for i in range(6):
        train[c + str(i)] = train[c].map(lambda x: re.split('\.|-', str(x))[i] if len(re.split('\.|-', str(x))) > i else -1)
        try:
            train[c + str(i)] = pd.to_numeric(train[c + str(i)])
        except:
            #print(f'{c + str(i)} cannot be casted to number')
            pass
            
train['HasExistsNotSet'] = train['SmartScreen'] == 'ExistsNotSet'
#In the competition details, a strong time component was indicated. 
#At this point, I am not aware of any columns which show this time component, so lets for now split our validation set based on the index
def split_train_val_set(X, Y, n):
    if n < 1: n=int(len(X.index) * n)
    return X.iloc[:n], X.iloc[n:], Y.iloc[:n], Y.iloc[n:]

#We prepare the training data by replacing the category variables with the category codes 
#and replacing the nan values in the numerical columns with the median
for col, val in train.items():
    if pd.api.types.is_string_dtype(val): 
        train[col] = val.astype('category').cat.as_ordered()
        train[col] = train[col].cat.codes
    elif pd.api.types.is_numeric_dtype(val) and val.isnull().sum() > 0:
        train[col] = val.fillna(val.median())

X, Y = train.drop('HasDetections', axis=1), train['HasDetections']
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
#X_train, X_val, Y_train, Y_val = split_train_val_set(X, Y, n=0.1)
X_train.head(5)

#To be able to test the models rapidly, we create a function to print the scores of the model.
def print_score(m):
    res = [roc_auc_score(m.predict(X_train), Y_train), roc_auc_score(m.predict(X_val), Y_val), 
           m.score(X_train, Y_train), m.score(X_val, Y_val)
          ]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
    
#As in the fastai course, the rf_samples can be reduced to allow for faster repetition cycles. 
#We also immediately create a reset function to check the model performance on the entire dataset.
def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples: forest.check_random_state(rs).randint(0, n_samples, n))
    
def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))
    
set_rf_samples(50000)


# In[ ]:


train[:5]


# In[ ]:


train.describe()


# <a id="2"></a> <br>
# ## **Train Model**

# In[ ]:


model = RandomForestClassifier(n_estimators=100, min_samples_leaf=50, max_features=0.5, n_jobs=-1, oob_score=False)
get_ipython().run_line_magic('time', 'model.fit(X_train, Y_train)')

print_score(model)


# <a id="3"></a> <br>
# ## **Derive Model insights**

# * Q) How in general model insights help?
# * *A) Debugging(ex: leakage problems), better feature engineering , direct future data collection and inform human decision making.*
# ---
# 
# * Q) How in this Microsoft malware prediction competition does model insights help?
# * *A) Better feature engineering*

# <a id="3.1"></a> <br>
# ### **Permutation Importance**
# Permutation Importance help us understand what features have the biggest impact on predictions

# In[ ]:


perm = PermutationImportance(model, random_state=1).fit(X_val, Y_val)
eli5.show_weights(perm, feature_names = X_val.columns.tolist())


# For now, let us dig more into the top 3 original features(AVProductStatesIdentifier, AVProductsInstalled, AvSigVersion) as indicated by Permutation Importance.

# <a id="3.2"></a> <br>
# ### **Partial Plots**
# Partial plots help us understand how a feature affects predictions.

# In[ ]:


feat_names = ['AVProductStatesIdentifier', 'AVProductsInstalled', 'AvSigVersion']

for feat_name in feat_names:
    pdp_dist = pdp.pdp_isolate(model=model, dataset=X_val, model_features=X_val.columns.tolist(), feature=feat_name)
    pdp.pdp_plot(pdp_dist, feat_name)
    plt.show()


# * **AVProductStatesIdentifier** -  ID for the specific configuration of a user's antivirus software 
#     * Chances of malware detection considerably increase from a value of ~23000 till ~48000. Thereafter, though chances decrease, these machine are still not as protected as the ones with a value < 23000. Having a value < 23000 for this feature seems ideal for a machine not to be affected by malware
#     * Additional stats : As can be seen in train.describe() above, min value = 6 and max value = 70492. 
#     
# * **AVProductsInstalled** - NA in description
#     * Having a value >= 2 lowers prediction of malware detection.
# 
# * **AvSigVersion** - Defender state information e.g. 1.217.1014.0 
#     * To be filled in

# Let's also check the interaction between the top 2 features. We'll make use of 2D interactive plots for this.

# In[ ]:


inter1  =  pdp.pdp_interact(model=model, dataset=X_val, model_features=X_val.columns.tolist(), features=['AVProductStatesIdentifier', 'AVProductsInstalled'])

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['AVProductStatesIdentifier', 'AVProductsInstalled'], plot_type='contour')
plt.show()


# This graph shows predictions for any combination of Goals Scored and Distance covered.
# 
# When AVProductsInstalled >= 2, the impact of AVProductStatesIdentifier on the prediction is not as much as it would have had when AVProductsInstalled < 2

# <a id="3.3"></a> <br>
# ### **SHAP Values**
#  SHAP values break down a single prediction to show the impact of each feature

# In[ ]:


row_to_show = 17
data_for_prediction = X_val.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

model.predict_proba(data_for_prediction_array)


# The machine is **83.38%** likely to get infected by malware

# In[ ]:


# Create object that can calculate shap values
explainer = shap.TreeExplainer(model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)

shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)


# **Feature values causing increased predictions are in pink, and their visual size shows the magnitude of the feature's effect. Feature values decreasing the prediction are in blue**

# **SHAP Summary Plot:** SHAP values all of validation data samples and not just a single row

# In[ ]:


# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(X_val)

# Make plot. Index of [1] is explained in text below.
shap.summary_plot(shap_values[1], X_val)


# **More to come. Stay tuned!**
