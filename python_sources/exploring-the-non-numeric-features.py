#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df_train = pd.read_csv('../input/train.csv',nrows=500000,low_memory=True)
# Not the complete dataset, but the number ratios should be representative of all the data
df_train.head()


# In[ ]:


df_train = df_train.select_dtypes('object')
df_train.shape


# ### Let's investigate these 30 'object' features

# In[ ]:


df_train.columns


# ### MachineIdentifier -- Just the individual machine ids. Not much more than an identifier.
# ### ProductName, EngineVersion, AppVersion, AvSigVersion -- All give defender state informations

# In[ ]:


df_train['MachineIdentifier'][:5] # Can we glean info from this?


# In[ ]:


print(df_train['ProductName'].isna().sum(),df_train['EngineVersion'].isna().sum(),      df_train['AppVersion'].isna().sum(),df_train['AvSigVersion'].isna().sum())


# ### None of them have any missing values

# In[ ]:


print(len(df_train['ProductName'].value_counts().unique()),len(df_train['EngineVersion'].value_counts().unique()),     len(df_train['AppVersion'].value_counts().unique()),len(df_train['AvSigVersion'].value_counts().unique()))


# ### How do we deal with the ones with so many values? These are clearly categorical variables. 

# In[ ]:


df_train['ProductName'].value_counts()


# In[ ]:


df_train['EngineVersion'].value_counts()[:10]


# In[ ]:


df_train['AppVersion'].value_counts()[:10]


# In[ ]:


df_train['AvSigVersion'].value_counts()[:10]


# In[ ]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(20,20))
sns.countplot(x = 'ProductName',
              data = df_train,
              order = df_train['ProductName'].value_counts().index,ax=ax1)
sns.countplot(x = 'EngineVersion',
              data = df_train,
              order = df_train['EngineVersion'].value_counts().iloc[:5].index,ax=ax2)
sns.countplot(x = 'AppVersion',
              data = df_train,
              order = df_train['AppVersion'].value_counts().iloc[:5].index,ax=ax3)
sns.countplot(x = 'AvSigVersion',
              data = df_train,
              order = df_train['AvSigVersion'].value_counts().iloc[:5].index,ax=ax4);


# ### Platform , Processor, OsVer, OsPlatformSubRelease, OsBuildLab, SkuEdition, PuaMode, SmartScreen
# * Platform - - Calculates platform name (of OS related properties and processor property)
# * Processor -- This is the process architecture of the installed operating system
# * OsVer -- Version of the current operating system
# * OsPlatformSubRelease -- Returns the OS Platform sub-release (Windows Vista, Windows 7, Windows 8, TH1, TH2)
# * OsBuildLab -- Build lab that generated the current OS. Example: 9600.17630.amd64fre.winblue_r7.150109-2022
# * SkuEdition -- The goal of this feature is to use the Product Type defined in the MSDN to map to a 'SKU-Edition' name that is useful in population reporting. The valid Product Type are defined in %sdxroot%\data\windowseditions.xml. This API has been used since Vista and Server 2008, so there are many Product Types that do not apply to Windows 10. The 'SKU-Edition' is a string value that is in one of three classes of results. The design must handle each class.
# * PuaMode -- Pua Enabled mode from the service (https://www.tenforums.com/tutorials/32236-enable-disable-windows-defender-pua-protection-windows-10-a.html)
# * SmartScreen -- This is the SmartScreen enabled string value from registry. This is obtained by checking in order, HKLM\SOFTWARE\Policies\Microsoft\Windows\System\SmartScreenEnabled and HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\SmartScreenEnabled. If the value exists but is blank, the value "ExistsNotSet" is sent in telemetry.

# In[ ]:


df_train['Platform'].value_counts() # Huh, whatever happened to Windows 95


# In[ ]:


df_train['Processor'].value_counts()


# ### Did you know what arm64 was?  I didn't. (https://en.wikipedia.org/wiki/ARM_architecture)

# In[ ]:


df_train['OsVer'].value_counts()[:10]


# In[ ]:


df_train['OsPlatformSubRelease'].value_counts()


# In[ ]:


df_train['OsBuildLab'].value_counts()[:10]


# In[ ]:


df_train['SkuEdition'].value_counts()


# In[ ]:


df_train['PuaMode'].value_counts() # The rest are probably nans => PuaMode = off


# In[ ]:


df_train['PuaMode'].isna().sum()


# In[ ]:


df_train['SmartScreen'].value_counts()


# In[ ]:


df_train['SmartScreen'].isna().sum() # Perhaps we should impute "ExistsNotSet" for these


# In[ ]:


print(df_train['Platform'].isna().sum(),df_train['Processor'].isna().sum(),df_train['SkuEdition'].isna().sum())


# In[ ]:


print(df_train['OsPlatformSubRelease'].isna().sum(),df_train['OsBuildLab'].isna().sum(),df_train['OsVer'].isna().sum())


# In[ ]:


df_train[df_train['OsBuildLab'].isnull()]


# ### If need be, we can impute the OsBuildLab missing value with the most frequent value '17134.1.amd64fre.rs4_release.180410-1804 '

# In[ ]:


print(len(df_train['Platform'].value_counts().unique()),len(df_train['Processor'].value_counts().unique()),     len(df_train['OsVer'].value_counts().unique()),len(df_train['OsPlatformSubRelease'].value_counts().unique()),     len(df_train['OsBuildLab'].value_counts().unique()),len(df_train['SkuEdition'].value_counts().unique()),     len(df_train['PuaMode'].value_counts().unique()),len(df_train['SmartScreen'].value_counts().unique()))


# ### Probably feasible to one-hot encode for all of them except OsBuildLab

# In[ ]:


fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2,4, figsize=(20,20))
sns.countplot(x = 'Platform',
              data = df_train,
              order = df_train['Platform'].value_counts().index,ax=ax1)
sns.countplot(x = 'Processor',
              data = df_train,
              order = df_train['Processor'].value_counts().index,ax=ax2)
sns.countplot(x = 'OsVer',
              data = df_train,
              order = df_train['OsVer'].value_counts().iloc[:5].index,ax=ax3)
sns.countplot(x = 'OsPlatformSubRelease',
              data = df_train,
              order = df_train['OsPlatformSubRelease'].value_counts().iloc[:5].index,ax=ax4);
sns.countplot(x = 'OsBuildLab',
              data = df_train,
              order = df_train['OsBuildLab'].value_counts().iloc[:3].index,ax=ax5);
sns.countplot(x = 'SkuEdition',
              data = df_train,
              order = df_train['SkuEdition'].value_counts().iloc[:5].index,ax=ax6);
ax5.set_xticklabels(['17134.1.amd64fre','16299.431.amd64fre','16299.15.amd64fre'], rotation=-30);
sns.countplot(x = 'PuaMode',
              data = df_train,
              order = df_train['PuaMode'].value_counts().index,ax=ax7);
sns.countplot(x = 'SmartScreen',
              data = df_train,
              order = df_train['SmartScreen'].value_counts().iloc[:5].index,ax=ax8);
ax8.set_xticklabels( df_train['SmartScreen'].value_counts().iloc[:5].index,rotation=-30);


# ### Census_MDC2FormFactor, Census_DeviceFamily, Census_ProcessorClass, Census_PrimaryDiskTypeName, Census_ChassisTypeName, Census_PowerPlatformRoleName ,Census_InternalBatteryType, Census_OSVersion
# 
# * Census_MDC2FormFactor - A grouping based on a combination of Device Census level hardware characteristics. The logic used to define Form Factor is rooted in business and industry standards and aligns with how people think about their device. (Examples: Smartphone, Small Tablet, All in One, Convertible...)
# * Census_DeviceFamily - AKA DeviceClass. Indicates the type of device that an edition of the OS is intended for. Example values: Windows.Desktop, Windows.Mobile, and iOS.Phone
# * Census_ProcessorClass - A classification of processors into high/medium/low. Initially used for Pricing Level SKU. No longer maintained and updated
# * Census_PrimaryDiskTypeName - Friendly name of Primary Disk Type - HDD or SSD
# * Census_ChassisTypeName - Retrieves a numeric representation of what type of chassis the machine has. A value of 0 means xx
# * Census_PowerPlatformRoleName - Indicates the OEM preferred power management profile. This value helps identify the basic form factor of the device
# * Census_InternalBatteryType - NA - ??
# * Census_OSVersion - Numeric OS version Example - 10.0.10130.0

# In[ ]:


df_train['Census_MDC2FormFactor'].value_counts()


# In[ ]:


df_train['Census_DeviceFamily'].value_counts()


# In[ ]:


df_train['Census_ProcessorClass'].value_counts() # So lots of nans -- probably for the newer models


# In[ ]:


df_train['Census_PrimaryDiskTypeName'].value_counts() # Unknown and Unspecified are probably the same?


# In[ ]:


df_train['Census_ChassisTypeName'].value_counts()[:10]


# In[ ]:


df_train['Census_PowerPlatformRoleName'].value_counts()


# In[ ]:


df_train['Census_InternalBatteryType'].value_counts()[:10]


# In[ ]:


df_train['Census_OSVersion'].value_counts()[:5]


# In[ ]:


cols = ['Census_MDC2FormFactor', 'Census_DeviceFamily',
       'Census_ProcessorClass', 'Census_PrimaryDiskTypeName',
       'Census_ChassisTypeName', 'Census_PowerPlatformRoleName',
       'Census_InternalBatteryType', 'Census_OSVersion']
for col in cols:
    print(col, "::::: NAN values: {}".format(df_train[col].isna().sum()),'::::: Unique vals: {}'.format(len(df_train[col].value_counts().unique())))


# ### Census_ProcessorClass NaNs unclear what to use, might need to remove this feature. Census_InternalBatteryType probably just replace with 'lion'. Census_PrimaryDiskTypeName probably group NaNs, Unknowns and Unspecifieds into one category. 

# In[ ]:


fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2,4, figsize=(20,20))
sns.countplot(x = 'Census_MDC2FormFactor',
              data = df_train,
              order = df_train['Census_MDC2FormFactor'].value_counts().iloc[:5].index,ax=ax1)
ax1.set_xticklabels(df_train['Census_MDC2FormFactor'].value_counts().iloc[:5], rotation=-30);
sns.countplot(x = 'Census_DeviceFamily',
              data = df_train,
              order = df_train['Census_DeviceFamily'].value_counts().index,ax=ax2)
sns.countplot(x = 'Census_ProcessorClass',
              data = df_train,
              order = df_train['Census_ProcessorClass'].value_counts().iloc[:5].index,ax=ax3)
sns.countplot(x = 'Census_PrimaryDiskTypeName',
              data = df_train,
              order = df_train['Census_PrimaryDiskTypeName'].value_counts().iloc[:5].index,ax=ax4);
sns.countplot(x = 'Census_ChassisTypeName',
              data = df_train,
              order = df_train['Census_ChassisTypeName'].value_counts().iloc[:5].index,ax=ax5);
sns.countplot(x = 'Census_PowerPlatformRoleName',
              data = df_train,
              order = df_train['Census_PowerPlatformRoleName'].value_counts().iloc[:5].index,ax=ax6);
ax6.set_xticklabels(df_train['Census_PowerPlatformRoleName'].value_counts().iloc[:5], rotation=-30);
sns.countplot(x = 'Census_InternalBatteryType',
              data = df_train,
              order = df_train['Census_InternalBatteryType'].value_counts().iloc[:5].index,ax=ax7);
sns.countplot(x = 'Census_OSVersion',
              data = df_train,
              order = df_train['Census_OSVersion'].value_counts().iloc[:5].index,ax=ax8);
ax8.set_xticklabels( df_train['Census_OSVersion'].value_counts().iloc[:5].index,rotation=-30);


# ### Census_OSArchitecture, Census_OSBranch, Census_OSEdition,Census_OSSkuName, Census_OSInstallTypeName, Census_OSWUAutoUpdateOptionsName, Census_GenuineStateName,Census_ActivationChannel,Census_FlightRing
# * Census_OSArchitecture - Architecture on which the OS is based. Derived from OSVersionFull. Example - amd64
# * Census_OSBranch - Branch of the OS extracted from the OsVersionFull. Example - OsBranch = fbl_partner_eeap where OsVersion = 6.4.9813.0.amd64fre.fbl_partner_eeap.140810-0005
# * Census_OSEdition - Edition of the current OS. Sourced from HKLM\Software\Microsoft\Windows NT\CurrentVersion@EditionID in registry. Example: Enterprise
# * Census_OSSkuName - OS edition friendly name (currently Windows only)
# * Census_OSInstallTypeName - Friendly description of what install was used on the machine i.e. clean
# * Census_OSWUAutoUpdateOptionsName - Friendly name of the WindowsUpdate auto-update settings on the machine.
# * Census_GenuineStateName - Friendly name of OSGenuineStateID. 0 = Genuine
# * Census_ActivationChannel - Retail license key or Volume license key for a machine.
# * Census_FlightRing - The ring that the device user would like to receive flights for. This might be different from the ring of the OS which is currently installed if the user changes the ring after getting a flight from a different ring.

# In[ ]:


df_train['Census_OSArchitecture'].value_counts()


# In[ ]:


df_train['Census_OSBranch'].value_counts()[:5]


# In[ ]:


df_train['Census_OSEdition'].value_counts()[:5]


# In[ ]:


df_train['Census_OSSkuName'].value_counts()[:5]


# In[ ]:


df_train['Census_OSInstallTypeName'].value_counts()


# In[ ]:


df_train['Census_OSWUAutoUpdateOptionsName'].value_counts()


# In[ ]:


df_train['Census_GenuineStateName'].value_counts() # Probably important in predicting Malware infection probability


# In[ ]:


df_train['Census_ActivationChannel'].value_counts()


# In[ ]:


df_train['Census_FlightRing'].value_counts()


# In[ ]:


cols = ['Census_OSArchitecture', 'Census_OSBranch', 'Census_OSEdition',
       'Census_OSSkuName', 'Census_OSInstallTypeName',
       'Census_OSWUAutoUpdateOptionsName', 'Census_GenuineStateName',
       'Census_ActivationChannel', 'Census_FlightRing']
for col in cols:
    print(col, "::::: NAN values: {}".format(df_train[col].isna().sum()),'::::: Unique vals: {}'.format(len(df_train[col].value_counts().unique())))


# ### None of these have any nans

# In[ ]:


fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3,3, figsize=(20,20))
sns.countplot(x = 'Census_OSArchitecture',
              data = df_train,
              order = df_train['Census_OSArchitecture'].value_counts().iloc[:5].index,ax=ax1)
# ax1.set_xticklabels(df_train['Census_MDC2FormFacto'].value_counts().iloc[:5], rotation=-30);
sns.countplot(x = 'Census_OSBranch',
              data = df_train,
              order = df_train['Census_OSBranch'].value_counts().iloc[:5].index,ax=ax2)
ax2.set_xticklabels(df_train['Census_OSBranch'].value_counts().iloc[:5], rotation=-30);
sns.countplot(x = 'Census_OSEdition',
              data = df_train,
              order = df_train['Census_OSEdition'].value_counts().iloc[:5].index,ax=ax3)
ax3.set_xticklabels(df_train['Census_OSEdition'].value_counts().iloc[:5], rotation=-30);
sns.countplot(x = 'Census_OSSkuName',
              data = df_train,
              order = df_train['Census_OSSkuName'].value_counts().iloc[:5].index,ax=ax4);
ax4.set_xticklabels(df_train['Census_OSSkuName'].value_counts().iloc[:5], rotation=-30);
sns.countplot(x = 'Census_OSInstallTypeName',
              data = df_train,
              order = df_train['Census_OSInstallTypeName'].value_counts().iloc[:5].index,ax=ax5);
sns.countplot(x = 'Census_OSWUAutoUpdateOptionsName',
              data = df_train,
              order = df_train['Census_OSWUAutoUpdateOptionsName'].value_counts().iloc[:5].index,ax=ax6);
ax6.set_xticklabels(df_train['Census_OSWUAutoUpdateOptionsName'].value_counts().iloc[:5], rotation=-30);
sns.countplot(x = 'Census_GenuineStateName',
              data = df_train,
              order = df_train['Census_GenuineStateName'].value_counts().iloc[:5].index,ax=ax7);
sns.countplot(x = 'Census_ActivationChannel',
              data = df_train,
              order = df_train['Census_ActivationChannel'].value_counts().iloc[:6].index,ax=ax8);
ax8.set_xticklabels( df_train['Census_ActivationChannel'].value_counts().iloc[:6].index,rotation=-30);
sns.countplot(x = 'Census_FlightRing',
              data = df_train,
              order = df_train['Census_FlightRing'].value_counts().iloc[:5].index,ax=ax9);


# ### So we've done a rudimentary EDA of the 'object' features in the dataframe. There is a lot of information here, and I will need to think deeper on this. 

# ### Hope this helps people starting to look into this. There is a lot to discover here! I will write up an EDA on the numerical features soon (if no one has done it as of then). 

# In[ ]:




