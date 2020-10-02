#!/usr/bin/env python
# coding: utf-8

# # <center> Vowpal Wabbit starter
# ## <center> Training while reading 
#     
# ![](https://habrastorage.org/webt/je/do/29/jedo293npvm0uytxuwx-4goid2e.jpeg)
# 
# 
# In this kernel, we train a model just on the fly while reading data. If you are in doubt how it's even possible - take a look at [this tutorial](https://www.kaggle.com/kashnitsky/vowpal-wabbit-tutorial-blazingly-fast-learning) on Vowpal Wabbit. We'll skip basic EDA and feature engineering (for that you can pick any kernel, ex. [this one]([this EDA](https://www.kaggle.com/artgor/is-this-malware-eda-fe-and-lgb-updated)). We are not going to beat cool baselines with this model, but it's a nice and fast starter.  

# In[ ]:


import math
import pandas as pd
from datetime import datetime
from vowpalwabbit import pyvw


# **Read feature names from the header of the test set.**

# In[ ]:


with open('../input/test.csv') as f:
    # skip header
    feature_names = f.readline().strip().split(',')


# **We'll drop features with too many unique values, too many missing values, and too imbalanced value distribution. Motivated by [this EDA](https://www.kaggle.com/artgor/is-this-malware-eda-fe-and-lgb-updated) by Andrew Lukyanenko.**

# In[ ]:


too_many_unique_vals = ['MachineIdentifier',
                        'Census_FirmwareVersionIdentifier',
                        'Census_OEMModelIdentifier',
                        'CityIdentifier'
                       ]
too_many_nas = ['PuaMode',
                'Census_ProcessorClass',
                'DefaultBrowsersIdentifier',
                'Census_IsFlightingInternal',
                'Census_InternalBatteryType',
                'Census_ThresholdOptIn',
                'Census_IsWIMBootEnabled'
               ]

too_imbalanced = ['Census_IsFlightsDisabled',
                  'Census_IsAlwaysOnAlwaysConnectedCapable',
                  'AVProductsEnabled',
                  'IsProtected',
                  'RtpStateBitfield',
                  'Census_IsVirtualDevice',
                  'Census_IsPortableOperatingSystem',
                  'Census_IsPenCapable',
                  'Census_FlightRing',
                  'OsVer',
                  'IsBeta',
                  'Platform',
                  'AutoSampleOptIn',
                  'Census_DeviceFamily',
                  'ProductName'
                 ]


# Let's figure out ids of numeric and categorical features that we'll used for prediction. Inspired by [this post](https://www.kaggle.com/c/microsoft-malware-prediction/discussion/75396) by Aditya Soni.

# In[ ]:


numeric_column_ids = [
    38,  # Census_ProcessorCoreCount
    42,  # Census_PrimaryDiskTotalCapacity
    44,  # Census_SystemVolumeTotalCapacity
    46,  # Census_TotalPhysicalRAM
    48,  # Census_InternalPrimaryDiagonalDisplaySizeInInches
    49,  # Census_InternalPrimaryDisplayResolutionHorizontal
    50,  # Census_InternalPrimaryDisplayResolutionVertical
    53   # Census_InternalBatteryNumberOfCharges  
]


# In[ ]:


categorical_column_ids = [i for i, feat_name in zip(range(len(feature_names)), feature_names) 
                          if (feat_name not in too_many_unique_vals + too_many_nas + too_imbalanced
                          and i not in numeric_column_ids)]


# Thus we have 48 categorical features and 8 numeric ones.

# In[ ]:


len(categorical_column_ids), len(numeric_column_ids)


# The following function converts a string to Vowpal Wabbit format. Take a look at [this tutorial](https://www.kaggle.com/kashnitsky/vowpal-wabbit-tutorial-blazingly-fast-learning) on VW to understand the format. 

# In[ ]:


def to_vw(line, categ_column_ids, num_column_ids, column_names, train=True):
    """
    Converts a string to VW format.
    
    :param line: a string with comma-separated feature values, str
    :param categ_column_ids: ids of categorical features, list
    :param num_column_ids: ids of numeric features, list
    :param column_names: column (or feature) names to use (both categorical and numeric), list
    :param train: whether the line belongs to a training set
    :return: processed line, str
    """
    values = line.strip().split(',')
    # VW treats '|' and ':' as special symbols, so jnust in case we'll replace them
    for i in range(len(values)):
        values[i] = values[i].replace('|', '').replace(':', '')
    label = '-1'
    if train:
        label, values = values[-1], values[:-1] 
        # in case of binary classification, VW eats labels 1 and -1, so 1 -> 1, 0 -> -1
        label = str(2 * int(label) - 1)
    
    # for categorical features, we fill in missing values with 'unk'
    for i in categ_column_ids:
        if not values[i]:
            values[i] = 'unk'
            
    # for numeric features, we fill in missing values with '-1'
    for i in num_column_ids:
        if values[i] == '':
            values[i] = '-1'
    
    categ_vw = ' '.join(['{}={}'.format(column_names[i], values[i])
                           for i in categ_column_ids])
    # we apply log1p transformation to numeric features
    numeric_vw = ' '.join(['{}:{}'.format(column_names[i],round(math.log(1 + float(values[i]) + 1e-10)))
                           for i in num_column_ids])
    
    new_line = label + ' |num ' + numeric_vw + ' |cat ' + categ_vw
    return new_line


# **Let's see how this function processes the first line from the test set.** 

# In[ ]:


line = '0000010489e3af074adeac69c53e555e,win8defender,1.1.15400.5,4.18.1810.5,1.281.501.0,0,7,0,,53447,1,1,1,43,58552,18,53,42,windows10,x64,10.0.0.0,15063,768,rs2,15063.0.amd64fre.rs2_release.170317-1834,Home,1,0,,,108,,1,1,Notebook,Windows.Desktop,2689,30661,4,5,3063,,488386,SSD,123179,0,8192,Notebook,15.5,1920,1080,Mobile,,8,10.0.15063.1387,amd64,rs2_release,15063,1387,Core,CORE,Reset,37,158,AutoInstallAndRebootAtMaintenanceTime,0,IS_GENUINE,OEM:DM,,0,Retail,,807,8554,1,,0,0,0,0,0,7'


# In[ ]:


to_vw(line, categorical_column_ids, numeric_column_ids, feature_names, train=False)


# **Reading training data and training Vowpal Wabbit on the fly.**

# In[ ]:


vw = pyvw.vw(b=28, random_seed=17, loss_function='logistic', passes=3, learning_rate=0.7, k=True, c=True, 
             link='logistic', quiet=True)
with open('../input/train.csv') as f:
    # skip header
    f.readline()
    start_time = datetime.now()
    for i, line in enumerate(f):
        # print when the next 1 mln examples is processed
        if i % 1e5 == 0: print("{}\t{} passed.".format(i, datetime.now() - start_time))
        # training Vowpal Wabbit with current example
        vw.learn(to_vw(line, categorical_column_ids, numeric_column_ids, feature_names, train=True))


# **Reading test data and making predictions on the fly.**

# In[ ]:


predictions = []
with open('../input/test.csv') as f:
    # skip header
    f.readline()
    start_time = datetime.now()
    for i, line in enumerate(f):
        # print when the next 1 mln examples is processed
        if i % 1e5 == 0: print("{}\t{} passed.".format(i, datetime.now() - start_time))
        # add Vowpal Wabbit prediction for the current example
        predictions.append(vw.predict(to_vw(line, categorical_column_ids, numeric_column_ids, feature_names, train=False)))


# **Form the submission file**

# In[ ]:


subm_df = pd.read_csv('../input/sample_submission.csv', index_col='MachineIdentifier')
subm_df['HasDetections'] = predictions
subm_df.to_csv('submission.csv', header=True)


# **What was skipped here and what can be improved**
# * train/test split: local validation gave showed ROC AUC 0.70838 for 30% holdout set
# * hyperparam tuning: this was done with [vw-hyperopt](https://github.com/VowpalWabbit/vowpal_wabbit/blob/master/utl/vw-hyperopt.py), take a look at [this Hyperopt tutorial](https://www.kaggle.com/ilialar/hyperparameters-tunning-with-hyperopt)
# * feature engineering: explore other kernels, come up with good features, add them, and see your AUC rise! 
# * blending: well, this solutions doesn;t result in a high AUC but try to blend this model predictions with some others
