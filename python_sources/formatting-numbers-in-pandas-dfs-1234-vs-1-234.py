#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pathlib

import pandas as pd


# In[ ]:


def read_data(data_path):
    # dtypes are borrowed from https://www.kaggle.com/konradb/shrinking-the-data
    dtypes = {
        "MachineIdentifier": "category",
        "ProductName": "category",
        "EngineVersion": "category",
        "AppVersion": "category",
        "AvSigVersion": "category",
        "IsBeta": "int8",
        "RtpStateBitfield": "float16",
        "IsSxsPassiveMode": "int8",
        "DefaultBrowsersIdentifier": "float16",
        "AVProductStatesIdentifier": "float32",
        "AVProductsInstalled": "float16",
        "AVProductsEnabled": "float16",
        "HasTpm": "int8",
        "CountryIdentifier": "int16",
        "CityIdentifier": "float32",
        "OrganizationIdentifier": "float16",
        "GeoNameIdentifier": "float16",
        "LocaleEnglishNameIdentifier": "int16",
        "Platform": "category",
        "Processor": "category",
        "OsVer": "category",
        "OsBuild": "int16",
        "OsSuite": "int16",
        "OsPlatformSubRelease": "category",
        "OsBuildLab": "category",
        "SkuEdition": "category",
        "IsProtected": "float16",
        "AutoSampleOptIn": "int8",
        "PuaMode": "category",
        "SMode": "float16",
        "IeVerIdentifier": "float16",
        "SmartScreen": "category",
        "Firewall": "float16",
        "UacLuaenable": "float32",
        "Census_MDC2FormFactor": "category",
        "Census_DeviceFamily": "category",
        "Census_OEMNameIdentifier": "float16",
        "Census_OEMModelIdentifier": "float32",
        "Census_ProcessorCoreCount": "float16",
        "Census_ProcessorManufacturerIdentifier": "float16",
        "Census_ProcessorModelIdentifier": "float16",
        "Census_ProcessorClass": "category",
        "Census_PrimaryDiskTotalCapacity": "float32",
        "Census_PrimaryDiskTypeName": "category",
        "Census_SystemVolumeTotalCapacity": "float32",
        "Census_HasOpticalDiskDrive": "int8",
        "Census_TotalPhysicalRAM": "float32",
        "Census_ChassisTypeName": "category",
        "Census_InternalPrimaryDiagonalDisplaySizeInInches": "float16",
        "Census_InternalPrimaryDisplayResolutionHorizontal": "float16",
        "Census_InternalPrimaryDisplayResolutionVertical": "float16",
        "Census_PowerPlatformRoleName": "category",
        "Census_InternalBatteryType": "category",
        "Census_InternalBatteryNumberOfCharges": "float32",
        "Census_OSVersion": "category",
        "Census_OSArchitecture": "category",
        "Census_OSBranch": "category",
        "Census_OSBuildNumber": "int16",
        "Census_OSBuildRevision": "int32",
        "Census_OSEdition": "category",
        "Census_OSSkuName": "category",
        "Census_OSInstallTypeName": "category",
        "Census_OSInstallLanguageIdentifier": "float16",
        "Census_OSUILocaleIdentifier": "int16",
        "Census_OSWUAutoUpdateOptionsName": "category",
        "Census_IsPortableOperatingSystem": "int8",
        "Census_GenuineStateName": "category",
        "Census_ActivationChannel": "category",
        "Census_IsFlightingInternal": "float16",
        "Census_IsFlightsDisabled": "float16",
        "Census_FlightRing": "category",
        "Census_ThresholdOptIn": "float16",
        "Census_FirmwareManufacturerIdentifier": "float16",
        "Census_FirmwareVersionIdentifier": "float32",
        "Census_IsSecureBootEnabled": "int8",
        "Census_IsWIMBootEnabled": "float16",
        "Census_IsVirtualDevice": "float16",
        "Census_IsTouchEnabled": "int8",
        "Census_IsPenCapable": "int8",
        "Census_IsAlwaysOnAlwaysConnectedCapable": "float16",
        "Wdft_IsGamer": "float16",
        "Wdft_RegionIdentifier": "float16",
        "HasDetections": "int8",
    }
    return pd.read_csv(data_path, dtype=dtypes)


# In[ ]:


DATA_PATH = pathlib.Path("../input")


# By default, pandas does not apply any number formats when DataFrames are displayed. As shown below, long integers are printed without any commas:

# In[ ]:


train_data = read_data(DATA_PATH / "train.csv")


# In[ ]:


train_data.agg(['size', 'count', 'nunique']).transpose().head(10)


# Applying a proper format can greatly enhance readability of the output. For instance, if the numbers of four or more digits are displayed with commas, we can easily recognize their orders of magnitude at a glace.
# 
# Surely pandas has options for customizing number formats, right? Well, turns out that the answer is: yes and no. Pandas does indeed expose an easy to use settings for [float formatting](https://pandas.pydata.org/pandas-docs/stable/options.html), but there is no corresponding option for integers (as far as I know). Fortunately, in Python almost everytihng can be customized with a little bit of monkey-patching. `CustomPandasDisplayOptions` class defined below allows to set custom formats for both floats and integers. All you need to do is to paste the following cell into your Kaggle kernel, or Jupyter notebook:

# In[ ]:


class CustomPandasDisplayOptions:
    import pandas.io.formats.format as pf

    _INT_FORMAT = "{:,d}".format
    _FLOAT_FORMAT = "{:,}".format

    @classmethod
    def enable(cls):
        """Sets custom options."""
        class _IntArrayFormatter(cls.pf.GenericArrayFormatter):
            def _format_strings(self):
                formatter = self.formatter or cls._INT_FORMAT
                fmt_values = [formatter(x) for x in self.values]
                return fmt_values
        # Save the original options for later.
        cls.orig_int_format = cls.pf.IntArrayFormatter
        cls.orig_float_format = pd.options.display.float_format
        # Set custom options.
        cls.pf.IntArrayFormatter = _IntArrayFormatter
        pd.options.display.float_format = cls._FLOAT_FORMAT

    @classmethod
    def disable(cls):
        """Restores the original options."""
        cls.pf.IntArrayFormatter = cls.orig_int_format
        pd.options.display.float_format = cls.orig_float_format


CustomPandasDisplayOptions.enable()


# From now on, all floats and integers should be formatted nicely:

# In[ ]:


train_data.agg(['size', 'count', 'nunique']).transpose().head(10)


# If you want, you can always disable the custom fomratting rules and revert back to the original settings:

# In[ ]:


CustomPandasDisplayOptions.disable()


# In[ ]:


train_data.agg(['size', 'count', 'nunique']).transpose().head(10)


# Hope that's useful!
