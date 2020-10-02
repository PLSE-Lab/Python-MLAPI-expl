#!/usr/bin/env python
# coding: utf-8

# # Complete EDA for the Microsoft Malware Prediction Competition
# 
# ** There is a lot of features in the data, including some complicated one. I do basic EDA here **
# 
# Enjoy!

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import operator 

sns.set_style('whitegrid')


# ## Loading Data
# I only load the train data here, I'll load the test data when needed. I use the types chosen here : https://www.kaggle.com/theoviel/load-the-totality-of-the-data/edit

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
        'HasDetections':                                        'int8',}


# In[ ]:


train_df = pd.read_csv('../input/train.csv', dtype=dtypes, nrows=1000000)


# In[ ]:


train_df.info()


# That's a lot of features...

# In[ ]:


train_df.head()


# ## Defender
# 
# - ProductName - Defender state information e.g. win8defender 
# - EngineVersion - Defender state information e.g. 1.1.12603.0 
# - AppVersion - Defender state information e.g. 4.9.10586.0 
# - AvSigVersion - Defender state information e.g. 1.217.1014.0 
# - IsBeta - Defender state information e.g. false 

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='ProductName', hue='HasDetections', data=train_df)
plt.yscale('log')
plt.title('Which Defenders are infected', size=15)
plt.show()


# The two dominating softwares are mse (Microsoft Security Essentials) and w8defender, which seem to have the same repartition.

# In[ ]:


df = train_df[train_df["IsBeta"] == 1]

plt.figure(figsize=(15,10))
sns.countplot(x='ProductName', hue='HasDetections', data=df)
plt.title('Beta defenders', size=15)
plt.show()


# This feature looks useless because there are too little softwares in beta. 
# 
# About the version features, I don't know what to do with them yet.

# ### Rtp State Bitfield
# We don't have any infos on this one. 
# Rtp means Real-time transport protocol, it is a network protocol for delivering audio and video over IP networks.
# 
# I had no idea about Bitfield, so thanks wikipedia : 
# > A bit field is a data structure used in computer programming. It consists of a number of adjacent computer memory locations which have been allocated to hold a sequence of bits, stored so that any single bit or group of bits within the set can be addressed. A bit field is most commonly used to represent integral types of known, fixed bit-width.

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='RtpStateBitfield', hue='HasDetections', data=train_df)
plt.yscale('log')
plt.title('RtpStateBitfield influence', size=15)
plt.show()


# Interesting feature, because it seems better to have less bit field, a model can definitely learn from this.

# ### Sxs Passive Mode
# 
# The only thing I found about Sxs is that it a memory card made by Sony. Anyways, it could have a passive mode I guess.

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='IsSxsPassiveMode', hue='HasDetections', data=train_df)
plt.yscale('log')
plt.title('Which Defenders are infected', size=15)
plt.show()


# 0 gives no info, but a passive Sxs is definitely interesting.

# ## Default browser
# This one can definitely be useful, be the problem is that this feature is numerical and takes a lot of values so nothing can really be said about it.

# In[ ]:


print("Number of browser ids :", len(df['DefaultBrowsersIdentifier'].unique()))


# ## Antivirus
# - AVProductStatesIdentifier - ID for the specific configuration of a user's antivirus software 
# - AVProductsInstalled - NA 
# - AVProductsEnabled - NA 
# - IsProtected - This is a calculated field derived from the Spynet Report's AV Products field. Returns: a. TRUE if there is at least one active and up-to-date antivirus product running on this machine. b. FALSE if there is no active AV product on this machine, or if the AV is active, but is not receiving the latest updates. c. null if there are no Anti Virus Products in the report. Returns: Whether a machine is protected. 

# In[ ]:


print("Number of antivirus ids :", len(df['AVProductStatesIdentifier'].unique()))


# #### Installed

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='AVProductsInstalled', hue='HasDetections', data=train_df)
plt.title('The more AntiVirus installed...', size=15)
plt.yscale('log')
plt.show()


# **... The better**
# 
# But I guess they should be enabled ?

# #### Enabled

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='AVProductsEnabled', hue='HasDetections', data=train_df)
plt.title('Enable your AntiVirus ?', size=15)
plt.yscale('log')
plt.show()


# #### Protected devices
# This feature is approximately a summary of the previous ones

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='IsProtected', hue='HasDetections', data=train_df)
plt.title('Protected devices', size=15)
plt.show()


# *Wait, not protected devices have a lower ratio of detections ?*
# This feature is questionnable then..

# ## Trusted Platform Module (TPM)
# 
#  It is an international standard for a secure cryptoprocessor. (thanks wikipedia, again)

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='HasTpm', hue='HasDetections', data=train_df)
plt.title('TPM influence', size=15)
plt.yscale('log')
plt.show()


# Nothing much about this one, does not seem to have any impact.

# ## Location
# 
# - CountryIdentifier - ID for the country the machine is located in 
# - CityIdentifier - ID for the city the machine is located in 
# - OrganizationIdentifier - ID for the organization the machine belongs in, organization ID is mapped to both specific companies and broad industries 
# - GeoNameIdentifier - ID for the geographic region a machine is located in 
# - LocaleEnglishNameIdentifier - English name of Locale ID of the current user 
# - Wdft_RegionIdentifier - NA
# 
# Again, those are only Ids so it is hard to do any EDA on those, but they definitely play a role.

# In[ ]:


print("Number of country ids :", len(train_df['CountryIdentifier'].unique()))
print("Number of city ids :", len(train_df['CityIdentifier'].unique()))
print("Number of organization ids :", len(train_df['OrganizationIdentifier'].unique()))
print("Number of region ids :", len(train_df['GeoNameIdentifier'].unique()))
print("Number of locale english name ids :", len(train_df['LocaleEnglishNameIdentifier'].unique()))
print("Number of region ids (wdft) :", len(train_df['Wdft_RegionIdentifier'].unique()))


# For example, let us plot the proportion of infected ids in some countries.

# In[ ]:


ratios = {}

for c in train_df['CountryIdentifier'].unique():
    df = train_df[train_df['CountryIdentifier'] == c]
    ratios[c] = sum(df['HasDetections']) / len(df)


# In[ ]:


data = pd.DataFrame({"Country Id": list(ratios.keys()), "Infection Ratio": list(ratios.values())}).sample(50).sort_values(by='Infection Ratio')
order = list(data['Country Id'])[::-1]

plt.figure(figsize=(15,10))
sns.barplot(x="Country Id", y="Infection Ratio", data=data, order=order)
plt.title('Proportion of infected samples for some countries', size=15)
plt.xticks(rotation=-45)
plt.show()


# ## Operating system
# - Platform - Calculates platform name (of OS related properties and processor property) 
# - Census_DeviceFamily - AKA DeviceClass. Indicates the type of device that an edition of the OS is intended for. Example values: Windows.Desktop, Windows.Mobile, and iOS.Phone 
# - Processor - This is the process architecture of the installed operating system 
# - OsVer - Version of the current operating system 
# - OsBuild - Build of the current operating system 
# - OsSuite - Product suite mask for the current operating system. 
# - OsPlatformSubRelease - Returns the OS Platform sub-release (Windows Vista, Windows 7, Windows 8, TH1, TH2) 
# - OsBuildLab - Build lab that generated the current OS. Example: 9600.17630.amd64fre.winblue_r7.150109-2022 

# #### OS

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='Platform', hue='HasDetections', data=train_df)
plt.title('Impact of the Operation System', size=15)
plt.yscale('log')
plt.show()


# Windows 8 seems to be the OS that has the worst ratio. Expected ? Probably, yes. I'd like to see the stats of older OS (vista / XP)

# #### Os Family

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='Census_DeviceFamily', hue='HasDetections', data=train_df)
plt.title('UacLua Influence', size=15)
plt.yscale('log')
plt.show()


# Not much to say here. 

# #### Processor

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='Platform', hue='Processor', data=train_df)
plt.title('Architectures of the different OS', size=15)
plt.yscale('log')
plt.show()


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='Processor', hue='HasDetections', data=train_df)
plt.title('x64 vs x86', size=15)
plt.yscale('log')
plt.show()


# In[ ]:


print("Number of os versions :", len(train_df['OsVer'].unique()))
print("Number of os builds :", len(train_df['OsBuild'].unique()))
print("Number of os suites :", len(train_df['OsSuite'].unique()))
print("Number of os platform subreleases :", len(train_df['OsPlatformSubRelease'].unique()))
print("Number of os build labs :", len(train_df['OsBuildLab'].unique()))


# #### Version

# In[ ]:


order = ['10.0.0.0', '10.0.0.1', '10.0.1.0', '10.0.1.44', '10.0.2.0', '10.0.21.0', '10.0.3.0', 
         '10.0.3.80', '10.0.32.0', '10.0.32.72', '10.0.4.0', '10.0.5.0', '10.0.5.18', '10.0.7.0', '10.0.80.0', 
         '6.1.0.0', '6.1.1.0', '6.1.3.0', '6.3.0.0', '6.3.1.0', '6.3.3.0', '6.3.4.0']

plt.figure(figsize=(15,10))
sns.countplot(x='OsVer', hue='Platform', data=train_df, order=order)
plt.title('Version of the OS', size=15)
plt.yscale('log')
plt.xticks(rotation=-45)
plt.show()


# Version seems useless, because the information is the same as for the OS feature, except for a dozen of outliers. We almost have the following :
# - w7 = 6.1.1.0
# - w8 = 6.3.0.0
# - w10 = 10.0.0.0
# - w2016 = 10.0.0.0
# 

# #### Suite

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='OsSuite', hue='Platform', data=train_df)
plt.title('Suites of different OS', size=15)
plt.yscale('log')
plt.show()


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='OsSuite', hue='HasDetections', data=train_df)
plt.title('Influence of the Suite', size=15)
plt.yscale('log')
plt.show()


# Way more interesting than the version feature, even though I don't really know what the suite is.

# #### OS Subrelease

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='OsPlatformSubRelease', hue='Platform', data=train_df)
plt.title('Subreleases for different OS', size=15)
plt.yscale('log')
plt.show()


# This one can give some infos about w10 devices.

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='OsPlatformSubRelease', hue='HasDetections', data=train_df)
plt.title('Influence of the os subreleases', size=15)
plt.yscale('log')
plt.show()


# ### Stock Keeping Unit 
# >  The goal of this feature is to use the Product Type defined in the MSDN to map to a 'SKU-Edition' name that is useful in population reporting. The valid Product Type are defined in %sdxroot%\data\windowseditions.xml. This API has been used since Vista and Server 2008, so there are many Product Types that do not apply to Windows 10. The 'SKU-Edition' is a string value that is in one of three classes of results. The design must hand each class.

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='SkuEdition', hue='HasDetections', data=train_df)
plt.title('Different SKUs', size=15)
plt.yscale('log')
plt.show()


# This feature is really interesting because it has a lot of meaning behind it. Even though its name is a bit weird, different categories are easily understandable

# ### Auto Sample Opt
#  - This is the SubmitSamplesConsent value passed in from the service, available on CAMP 9+ 

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='AutoSampleOptIn', hue='HasDetections', data=train_df)
plt.title('Influence of auto sample', size=15)
plt.yscale('log')
plt.show()


# I don't know what this is, but as everybody seems to have it off, it is probably not that useful.

# ### PUA mode
# - Pua Enabled mode from the service 
# 
# I guess PUA stands for Private Use Areas

# In[ ]:


train_df['PuaMode'] = train_df['PuaMode'].cat.add_categories(['off'])
train_df['PuaMode'] = train_df[['PuaMode']].fillna('off')


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='PuaMode', hue='HasDetections', data=train_df)
plt.title('Pua Mode influence', size=15)
plt.yscale('log')
plt.show()


# Same as the previous one.

# ### Store Mode
#  - SMode - This field is set to true when the device is known to be in 'S Mode', as in, Windows 10 S mode, where only Microsoft Store apps can be installed 

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='SMode', hue='HasDetections', data=train_df)
plt.title('Store Mode influence', size=15)
plt.yscale('log')
plt.show()


# It is way harder to get a malware with only app from the Microsoft store, so this feature is interesting.

# ### IeVerIdentifier
# Could send for Internet Explorer Version, but I'm just guessing.

# In[ ]:


print("Number of ie versions :", len(train_df['IeVerIdentifier'].unique()))


# ### Smart Screen parameter
# - SmartScreen - This is the SmartScreen enabled string value from registry. This is obtained by checking in order, HKLM\SOFTWARE\Policies\Microsoft\Windows\System\SmartScreenEnabled and HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\SmartScreenEnabled. If the value exists but is blank, the value "ExistsNotSet" is sent in telemetry
# 
# Smart Screen is a software supposed to protect users from Malwares.

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='SmartScreen', hue='HasDetections', data=train_df)
plt.title('Smart Screen influence', size=15)
plt.yscale('log')
plt.show()


# Values need cleaning, but it appears to be better to turn SmartScreen on. Good to know.

# ### Firewall
# 
# - Firewall - This attribute is true (1) for Windows 8.1 and above if windows firewall is enabled, as reported by the service. 

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='Firewall', hue='HasDetections', data=train_df)
plt.title('Firewall Influence', size=15)
plt.yscale('log')
plt.show()


# Most people have their firewall on, but it appears that it does not have a lot of influence on malwares. 

# ### User Account Control Limited User Account  (UacLua)
# UacLuaenable - This attribute reports whether or not the "administrator in Admin Approval Mode" user type is disabled or enabled in UAC. The value reported is obtained by reading the regkey HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System\EnableLUA. 

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='UacLuaenable', hue='HasDetections', data=train_df)
plt.title('UacLua Influence', size=15)
plt.yscale('log')
plt.show()


# Not very meaningful.

# ## Device
# 
# - Census_MDC2FormFactor - A grouping based on a combination of Device Census level hardware characteristics. The logic used to define Form Factor is rooted in business and industry standards and aligns with how people think about their device. (Examples: Smartphone, Small Tablet, All in One, Convertible...) 
# - Census_OEMNameIdentifier - NA 
# - Census_OEMModelIdentifier - NA 

# In[ ]:


print("Number of names ids :", len(train_df['Census_OEMNameIdentifier'].unique()))
print("Number of model ids :", len(train_df['Census_OEMModelIdentifier'].unique()))


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='Census_MDC2FormFactor', hue='HasDetections', data=train_df)
plt.title('UacLua Influence', size=15)
plt.yscale('log')
plt.show()


# This one is interesting. Some devices are more targeted by malwares.

# ## Processor
# - Census_ProcessorCoreCount - Number of logical cores in the processor 
# - Census_ProcessorManufacturerIdentifier - NA 
# - Census_ProcessorModelIdentifier - NA 
# - Census_ProcessorClass - A classification of processors into high/medium/low. Initially used for Pricing Level SKU. No longer maintained and updated 

# In[ ]:


print("Number of processor manufacturer ids :", len(train_df['Census_ProcessorManufacturerIdentifier'].unique()))
print("Number of processor model ids :", len(train_df['Census_ProcessorModelIdentifier'].unique()))


# The processor has nothing to do with malwares, but it can tell us a bit about the user of the computer. Let us look at the classes.

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='Census_ProcessorClass', hue='HasDetections', data=train_df, order=['low', 'mid', 'high'])
plt.title('Different precessor tiers', size=15)
plt.show()


# I like this one.

# ## Storage
# 
# - Census_PrimaryDiskTotalCapacity - Amount of disk space on primary disk of the machine in MB 
# - Census_PrimaryDiskTypeName - Friendly name of Primary Disk Type - HDD or SSD 
# - Census_SystemVolumeTotalCapacity - The size of the partition that the System volume is installed on in MB 
# 

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='Census_PrimaryDiskTypeName', hue='HasDetections', data=train_df)
plt.title('Type of primary disk influence', size=15)
plt.show()


# SSD are better than HDD, so it makes sense.

# In[ ]:


ratios = {}
volumes = [50000, 100000, 500000, 1000000, 1000000000]

for i, vol in enumerate(volumes[1:]):
    df = train_df[train_df['Census_PrimaryDiskTotalCapacity'] <= vol]
    df = df[df['Census_PrimaryDiskTotalCapacity'] >= volumes[i]]
    ratios[vol] = sum(df['HasDetections']) / len(df)
    
data = pd.DataFrame({"Disk Volume": list(ratios.keys()), "Infection Ratio": list(ratios.values())}).sort_values(by="Disk Volume")


# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot(x="Disk Volume", y="Infection Ratio", data=data)
plt.title('Proportion of infected samples for different disk sizes', size=15)
plt.xticks(range(0, 4), ["<100 000", "[100 000; 500 000]", "[500 000; 1 000 000]", ">1 000 000"], rotation=-85)
plt.show()


# Same as with the processor, a bigger disk implies more usuage, and especially more downloading.

# 
# ### Disk Drive
# - Census_HasOpticalDiskDrive - True indicates that the machine has an optical disk drive (CD/DVD)

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='Census_HasOpticalDiskDrive', hue='HasDetections', data=train_df)
plt.title('Type of primary disk influence', size=15)
plt.show()


# Recent devices don't have disk drive anymore. Assuming that older devices tend to be more likely to have malwares, this feature is useful.

# ### Ram
# - Census_TotalPhysicalRAM - Retrieves the physical RAM in MB 

# In[ ]:


ratios = {}
volumes = [0, 2000, 4000, 6000, 8000, 12000, 16000, 1000000]

for i, vol in enumerate(volumes[1:]):
    df = train_df[train_df['Census_TotalPhysicalRAM'] <= vol]
    df = df[df['Census_TotalPhysicalRAM'] >= volumes[i]]
    ratios[vol] = sum(df['HasDetections']) / len(df)
    
data = pd.DataFrame({"ram": list(ratios.keys()), "Infection Ratio": list(ratios.values())}).sort_values(by="ram")


# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot(x="ram", y="Infection Ratio", data=data)
plt.title('Proportion of infected samples for different RAM sizes', size=15)
plt.xticks(range(0, 8), ["<2Gb", "[2Gb; 4Gb]", "[4Gb; 6Gb]", "[6Gb; 8Gb]", "[8Gb; 12Gb]", "[12Gb; 16Gb]", "> 16Gb",], rotation=-85)
plt.show()


# Same thing as with storage space, basically.

# ## Other Setup infos : (TO DO)
# 
# Chassis :
# - Census_ChassisTypeName - Retrieves a numeric representation of what type of chassis the machine has. A value of 0 means xx 
# 
# Display : 
# - Census_InternalPrimaryDiagonalDisplaySizeInInches - Retrieves the physical diagonal length in inches of the primary display 
# - Census_InternalPrimaryDisplayResolutionHorizontal - Retrieves the number of pixels in the horizontal direction of the internal display. 
# - Census_InternalPrimaryDisplayResolutionVertical - Retrieves the number of pixels in the vertical direction of the internal display 
# 
# Battery :
# - Census_PowerPlatformRoleName - Indicates the OEM preferred power management profile. This value helps identify the basic form factor of the device 
# - Census_InternalBatteryType - NA 
# - Census_InternalBatteryNumberOfCharges - NA 
# - Census_IsAlwaysOnAlwaysConnectedCapable - Retreives information about whether the battery enables the device to be AlwaysOnAlwaysConnected . 
# 
# Os (again ?) :
# - Census_OSVersion - Numeric OS version Example - 10.0.10130.0 
# - Census_OSArchitecture - Architecture on which the OS is based. Derived from OSVersionFull. Example - amd64 
# - Census_OSBranch - Branch of the OS extracted from the OsVersionFull. Example - OsBranch = fbl_partner_eeap where OsVersion = 6.4.9813.0.amd64fre.fbl_partner_eeap.140810-0005 
# - Census_OSBuildNumber - OS Build number extracted from the OsVersionFull. Example - OsBuildNumber = 10512 or 10240 
# - Census_OSBuildRevision - OS Build revision extracted from the OsVersionFull. Example - OsBuildRevision = 1000 or 16458 
# - Census_OSEdition - Edition of the current OS. Sourced from HKLM\Software\Microsoft\Windows NT\CurrentVersion@EditionID in registry. Example: Enterprise 
# - Census_OSSkuName - OS edition friendly name (currently Windows only) 
# - Census_OSInstallTypeName - Friendly description of what install was used on the machine i.e. clean 
# - Census_OSInstallLanguageIdentifier - NA 
# - Census_OSUILocaleIdentifier - NA 
# - Census_OSWUAutoUpdateOptionsName - Friendly name of the WindowsUpdate auto-update settings on the machine. 
# - Census_IsPortableOperatingSystem - Indicates whether OS is booted up and running via Windows-To-Go on a USB stick. 
# - Census_GenuineStateName - Friendly name of OSGenuineStateID. 0 = Genuine 
# 
# Firmware : 
# - Census_FirmwareManufacturerIdentifier - NA 
# - Census_FirmwareVersionIdentifier - NA 
# 
# Boot : 
# - Census_IsSecureBootEnabled - Indicates if Secure Boot mode is enabled. 
# - Census_IsWIMBootEnabled - NA 
# - Census_IsVirtualDevice - Identifies a Virtual Machine (machine learning model) 
# 
# Touch Screen:
# - Census_IsTouchEnabled - Is this a touch device ? 
# - Census_IsPenCapable - Is the device capable of pen input ? 
# 
# Others : 
# - Census_ActivationChannel - Retail license key or Volume license key for a machine. 
# - Census_IsFlightingInternal - NA 
# - Census_IsFlightsDisabled - Indicates if the machine is participating in flighting. 
# - Census_FlightRing - The ring that the device user would like to receive flights for. This might be different from the ring of the OS which is currently installed if the user changes the ring after getting a flight from a different ring. 
# - Census_ThresholdOptIn - NA 

# ##### Let us consider two last things : 
# 
# ### Gamers
# - Wdft_IsGamer - Indicates whether the device is a gamer device or not based on its hardware combination. 

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='Wdft_IsGamer', hue='HasDetections', data=train_df)
plt.title("Gamers' Malwares", size=15)
plt.show()


# Gamers do get more malwares. Maybe because they spend more time on their computer and tend to download more. This sums up the observations on the processor, storage and ram.

# ## Target repartition

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(train_df['HasDetections'])
plt.show()


# > "Perfectly balanced, as all things should be"

# ### To be continued ...
# #### Thanks for reading ! Any feedback is appreciated.
