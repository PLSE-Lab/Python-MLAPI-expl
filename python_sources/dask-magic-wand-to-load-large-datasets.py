#!/usr/bin/env python
# coding: utf-8

# ![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/07/dask-feat-850x478.jpg)

# **Note:**  
# Kindly upvote the kernel if you find it useful. Suggestions are always welome. Let me know your thoughts in the comment if any.

# **Data reading**  
# As the data volume is huge for Microsoft Malware Prediction Competition (https://www.kaggle.com/c/microsoft-malware-prediction), I am going to test Dask this time to read the data files.  Below, are the list of steps that I will follow before using Dash to read the data.
# 
# * I will load objects as categories  
# * Binary values are switched to int8  
# * Binary values with missing values are switched to float16 (int does not understand nan)  
# * 64 bits encoding are all switched to 32, or 16 of possible  
# 
# Thanks to Theo Viel for the amazing data reading technique. Below, is the link of the technique used to read large data files.
# 
# https://www.kaggle.com/theoviel/load-the-totality-of-the-data  

# In[ ]:


#Datatypes definition
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


# **Data reading using dask**

# In[ ]:


get_ipython().run_cell_magic('time', '', "import dask\nimport dask.dataframe as dd\ntrain_df = dd.read_csv('../input/microsoft-malware-prediction/train.csv', dtype=dtypes)\ntest_df = dd.read_csv('../input/microsoft-malware-prediction/test.csv', dtype=dtypes)")


# **Training Data Sample**

# In[ ]:


train_df.head()


# **Testing Data Sample**

# In[ ]:


test_df.head()


# **Bingo.... Voila... Data is loaded into the environment within blink of an eye!!!**  

# Now, with the training and testing datasets loaded into the environment which means only portion of the data is loaded. So, In order to loead the complete dataset the dash dataframes should be converted to pandas dataframe.

# **Dash Dataframe to Pandas Dataframe**

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df = train_df.compute()  \ntest_df = test_df.compute()')


# In[ ]:


print(train_df.shape)
print(test_df.shape)


# **Bingo.... Voila... Now Complete data is loaded into the environment!!!**  
# 
# **Happy Coding**  
# 
# **Thanks to Dask**  
