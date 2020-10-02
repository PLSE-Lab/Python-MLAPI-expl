#!/usr/bin/env python
# coding: utf-8

# 
# ![](http://www.portofquincy.org/wp-content/uploads/2015/08/Microsoft_logo_2012_modified.svg_.png)
# 

# ****Do we have something interesting in the dataset?
# 
# What i've done:
# 
# 1. Loaded a dataset with explicit types
# 2. Target and ID columns demistifyied :)
# 3. Categorical columns. Quick look
# 
# Plan to do next:
# 
# 4. Manage columns with versions Either:
#     - separate out major, minor, fixes to the new features
#     - cast them to integer and look if the order makes sense
#     
# 5. Checkout a test set distribution
# 
# 6. Other types of columns. Quick look
# 
# 7. Downcast float and int types more when reading a train file (see attachment)

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


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


# In[ ]:


TARGET = 'HasDetections'
ID     = 'MachineIdentifier'

TRAIN_FILE = "../input/train.csv"


# In[ ]:


train = pd.read_csv(TRAIN_FILE, dtype=dtypes)


# Reduced to 7.8G from 19G memory usage thank's to the explicit data types.
# And to be more concrete, thank's [Theo Viel](https://www.kaggle.com/theoviel) for that! ([original kaggle kernel](https://www.kaggle.com/theoviel/load-the-totality-of-the-data))
# 
# First of all, let's take a look at the target distribution

# In[ ]:


sns.countplot(train['HasDetections']);


# As we can see, it's distributed nearly perfect 50/50
# 
# Moving forward
# 
# I'd like to be sure that we have one row per ID in this dataset.
# 
# Let's leave this assumption here and take a look at the category variables

# In[ ]:


train['MachineIdentifier'].nunique() == train.shape[0]


# Now i'd like to choose some integer value as a threshold to decouple categorical features, which we can analyze from others I guess **30 unique values** might be reasonable

# In[ ]:


nunique_values = train.nunique(axis=0)
cat_cols = nunique_values[nunique_values < 30].index


# And finally, let's plot bar charts to understand some correlations and distributions.
# 
# I'll do that **with respect to target** metric (which is '**HasDetections**')

# In[ ]:


from math import floor

def plot_slice_of_cols(start, end, cols_dict, train_df):
    switcher = 1
    fig, axes = plt.subplots(8, 2, figsize=(15,30))
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    for i, col in enumerate(cols_dict[start:end]):
        switcher ^= 1
        spl = sns.countplot(col, hue='HasDetections', data=train_df, ax=axes[floor(i/2)][switcher])
        spl.set_xticklabels(spl.get_xticklabels(),rotation=30)


# In[ ]:


plot_slice_of_cols(0, 15, cat_cols, train)


# So the first look:
# 
# * *ProductName*, *IsBeta*, *IsSxsPassiveMode*, *HasTpm*, *Platform*, *AutoSampleOptIn*: **Rubbish** Could be **dropped** as for me. Target metric distributed mostly equally
# * Other features seem to be weak.
# * It may be worth to consider features like: *AvProductsInstalled*, *AvProductsEnabled* (but it's obvious ;))
# 
# What is also interesting. I'm quite confused about a feature '*IsProtected*'... Can somebody who use windows help me to understand what's that ? Thank you in advance!

# In[ ]:


plot_slice_of_cols(15, 30, cat_cols, train)


# Here *SmartScreen* may be worth to dig...

# In[ ]:





# In[ ]:


plot_slice_of_cols(30, 45, cat_cols, train)


# As a conclusion about this categorical features, i may say that we don't have a very strong features among them.
# 
# However, it's interesting to dive deeper into some other features (45/~80 only seen for now)
# 
# **TO BE CONTINUED**

# **Attachment**: Columns may be downcasted more (not tested yet, so not reported to Theo Vial)
# 2 problems could be present: None values among integers, and test set distribution 
# 
# * RtpStateBitfield      :float16 --> int8
# * AVProductsInstalled   :float16 --> int8
# * AVProductsEnabled     :float16 --> int8
# * Wdft_RegionIdentifier :float16 --> int8

# In[ ]:




