#!/usr/bin/env python
# coding: utf-8

# # Sentinel 2 Atmospheric Correction in Google Earth Engine

# ### Clone Repo

# In[ ]:


get_ipython().system('git clone https://github.com/ridhodwidharmawan/gee-atmcorr-S2')


# In[ ]:


cd /kaggle/working/gee-atmcorr-S2/bin/


# ### Authorize Google Earth Engine API

# In[ ]:


import ee
ee.Authenticate()


# ### Installing Py6s using conda-forge channel
# Its reccomended using package from conda-forge, don't use pip

# In[ ]:


get_ipython().system('conda install -c conda-forge py6s -y')


# ### Import modules 
# Import require module and initialize Earth Engine

# In[ ]:


from Py6S import *
import datetime
import math
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()),'bin'))
from atmospheric import Atmospheric

ee.Initialize()


# ### Location Coordinate of Study Area
# Define the location that you are looking for.

# In[ ]:


# This is centroid of Yogyakarta City
geom = ee.Geometry.Point(110.363524, -7.782469)


# ### Choose clearest image in a year 
# The following code will grab the first scene of clearest image in a year (Example 2018)

# In[ ]:


# The first Sentinel 2 image
S2 = ee.Image(
  ee.ImageCollection('COPERNICUS/S2')
    .filterBounds(geom)
    .filterDate('2018-01-01', '2018-12-31')
    .sort('CLOUDY_PIXEL_PERCENTAGE')
    .first()
  )

# top of atmosphere reflectance
toa = S2.divide(10000)


# ### Get Metadata

# In[ ]:


info = S2.getInfo()['properties']
scene_date = datetime.datetime.utcfromtimestamp(info['system:time_start']/1000)# i.e. Python uses seconds, EE uses milliseconds
solar_z = info['MEAN_SOLAR_ZENITH_ANGLE']
date = ee.Date(scene_date.strftime("%Y-%m-%d"))


# ### Atmospheric constituents

# In[ ]:


h2o = Atmospheric.water(geom,date).getInfo()
o3 = Atmospheric.ozone(geom,date).getInfo()
aot = Atmospheric.aerosol(geom,date).getInfo()


# ### Calculate target altitude in km

# In[ ]:


SRTM = ee.Image('CGIAR/SRTM90_V4')# Shuttle Radar Topography mission covers *most* of the Earth
alt = SRTM.reduceRegion(reducer = ee.Reducer.mean(),geometry = geom.centroid()).get('elevation').getInfo()
km = alt/1000 # i.e. Py6S uses units of kilometers


# ### 6S object
# 
# The backbone of Py6S is the 6S (i.e. SixS) class. It allows you to define the various input parameters, to run the radiative transfer code and to access the outputs which are required to convert radiance to surface reflectance.

# In[ ]:


# Instantiate
s = SixS()

# Atmospheric constituents
s.atmos_profile = AtmosProfile.UserWaterAndOzone(h2o,o3)
s.aero_profile = AeroProfile.Continental
s.aot550 = aot

# Earth-Sun-satellite geometry
s.geometry = Geometry.User()
s.geometry.view_z = 0               # always NADIR (I think..)
s.geometry.solar_z = solar_z        # solar zenith angle
s.geometry.month = scene_date.month # month and day used for Earth-Sun distance
s.geometry.day = scene_date.day     # month and day used for Earth-Sun distance
s.altitudes.set_sensor_satellite_level()
s.altitudes.set_target_custom_altitude(km)


# ### Spectral Response functions

# Py6S uses the Wavelength class to handle the wavelength(s) associated with a given channel (a.k.a. waveband). This might be a single scalar value (e.g. a central wavelength) or, if known, possibly the spectral response function of the waveband. The Sentinel 2 spectral response functions are provided with Py6S (as well as those of a number of missions). For more details please see the [docs](http://py6s.readthedocs.io/en/latest/params.html#wavelengths) or the (comment-rich) [source code](https://github.com/robintw/Py6S/blob/master/Py6S/Params/wavelength.py)

# In[ ]:


def spectralResponseFunction(bandname):
    """
    Extract spectral response function for given band name
    """

    bandSelect = {
        'B1':PredefinedWavelengths.S2A_MSI_01,
        'B2':PredefinedWavelengths.S2A_MSI_02,
        'B3':PredefinedWavelengths.S2A_MSI_03,
        'B4':PredefinedWavelengths.S2A_MSI_04,
        'B5':PredefinedWavelengths.S2A_MSI_05,
        'B6':PredefinedWavelengths.S2A_MSI_06,
        'B7':PredefinedWavelengths.S2A_MSI_07,
        'B8':PredefinedWavelengths.S2A_MSI_08,
        'B8A':PredefinedWavelengths.S2A_MSI_8A,
        'B9':PredefinedWavelengths.S2A_MSI_09,
        'B10':PredefinedWavelengths.S2A_MSI_10,
        'B11':PredefinedWavelengths.S2A_MSI_11,
        'B12':PredefinedWavelengths.S2A_MSI_12,
        }
    
    return Wavelength(bandSelect[bandname])


# ### TOA Reflectance to Radiance
# 
# Sentinel 2 data is provided as top-of-atmosphere reflectance. Lets convert this to at-sensor radiance for the atmospheric correction.*
# 
# \*<sub>You *can* atmospherically corrected directly from TOA reflectance. However, I suggest radiance for a couple of reasons.
#   Firstly, it is more intuitive. Instead of *spherical albedo* (which I suspect is more of a mathematical convenience than a physical property) you can use solar irradiance, transmissivity, path radiance, etc. Secondly, Py6S seems to be more geared towards converting from radiance to SR</sup>
# 
# 
# 
# 

# In[ ]:


def toa_to_rad(bandname):
    """
    Converts top of atmosphere reflectance to at-sensor radiance
    """
    
    # solar exoatmospheric spectral irradiance
    ESUN = info['SOLAR_IRRADIANCE_'+bandname]
    solar_angle_correction = math.cos(math.radians(solar_z))
    
    # Earth-Sun distance (from day of year)
    doy = scene_date.timetuple().tm_yday
    d = 1 - 0.01672 * math.cos(0.9856 * (doy-4))# http://physics.stackexchange.com/questions/177949/earth-sun-distance-on-a-given-day-of-the-year
   
    # conversion factor
    multiplier = ESUN*solar_angle_correction/(math.pi*d**2)

    # at-sensor radiance
    rad = toa.select(bandname).multiply(multiplier)
    
    return rad


# ### Radiance to Surface Reflectance
# 
# Reflected sunlight can be described as follows (wavelength dependence is implied):
# 
# $ L = \tau\rho(E_{dir} + E_{dif})/\pi + L_p$
# 
# where L is at-sensor radiance, $\tau$ is transmissivity, $\rho$ is surface reflectance, $E_{dir}$ is direct solar irradiance, $E_{dif}$ is diffuse solar irradiance and $L_p$ is path radiance. There are five unknowns in this equation, 4 atmospheric terms ($\tau$, $E_{dir}$, $E_{dif}$ and $L_p$) and surface reflectance. The 6S radiative transfer code is used to solve for the atmospheric terms, allowing us to solve for surface reflectance.
# 
# $ \rho = \pi(L - L_p) / \tau(E_{dir} + E_{dif}) $

# In[ ]:


def surface_reflectance(bandname):
    """
    Calculate surface reflectance from at-sensor radiance given waveband name
    """
    
    # run 6S for this waveband
    s.wavelength = spectralResponseFunction(bandname)
    s.run()
    
    # extract 6S outputs
    Edir = s.outputs.direct_solar_irradiance             #direct solar irradiance
    Edif = s.outputs.diffuse_solar_irradiance            #diffuse solar irradiance
    Lp   = s.outputs.atmospheric_intrinsic_radiance      #path radiance
    absorb  = s.outputs.trans['global_gas'].upward       #absorption transmissivity
    scatter = s.outputs.trans['total_scattering'].upward #scattering transmissivity
    tau2 = absorb*scatter                                #total transmissivity
    
    # radiance to surface reflectance
    rad = toa_to_rad(bandname)
    ref = rad.subtract(Lp).multiply(math.pi).divide(tau2*(Edir+Edif))
    
    return ref


# ### Atmospheric Correction

# In[ ]:


# Use this if you only need surface reflectance of rgb channel
b = surface_reflectance('B2')
g = surface_reflectance('B3')
r = surface_reflectance('B4')
ref = r.addBands(g).addBands(b)

# # Calculate surface reflectance for all wavebands
# output = S2.select('QA60')
# for band in ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']:
#     print(band)
#     output = output.addBands(surface_reflectance(band))


# ### Display results

# In[ ]:


from IPython.display import display, Image

region = geom.buffer(5000).bounds().getInfo()['coordinates']
channels = ['B4','B3','B2']

original = Image(url=toa.select(channels).getThumbUrl({
                'region':region,
                'min':0,
                'max':0.15
                }))

corrected = Image(url=ref.select(channels).getThumbUrl({
                'region':region,
                'min':0,
                'max':0.15
                }))

display(original, corrected)


# ### Export to Asset

# In[ ]:


# # set some properties for export
dateString = scene_date.strftime("%Y-%m-%d")
ref = ref.set({'satellite':'Sentinel 2',
              'fileID':info['system:index'],
              'date':dateString,
              'aerosol_optical_thickness':aot,
              'water_vapour':h2o,
              'ozone':o3})


# In[ ]:


# Define YOUR Google Earth Engine assetID 
# in my case it was something like this..
assetID = 'users/ridhodwid/S2A_SR_yogya_'+dateString


# In[ ]:


# # export
export = ee.batch.Export.image.toAsset(    image=ref,
    description='sentinel2_atmcorr_export',
    assetId = assetID,
    region = region)

# # uncomment to run the export
export.start() 

