#!/usr/bin/env python
# coding: utf-8

# This short Notebook serves as introduction to other Notebooks aimed to analyze specific aspects of the Emissions Factors.
# 
# <h2>Activity data: electricity generated in Power Plants</h2>
# I began working with the provided file, gppd_120_pr.csv. After some analysis, available in Notebook https://www.kaggle.com/ajulian/gppd-120-pr-csv-and-the-capacity-factor, I realized the electricity generation information was not realistic, so I decided to complement the file with another source.
# 
# I could find EIA-923 monthly data for each Power Plant for the first half of rolling year (Jul 2018 - Dec 2018), both about power generation and fuel consumption. Combined with standard Emissions Factors for fuel consumption, from EPA's eGRID, I could calculate for the second half of year 2018:
# - a reference emissions level of 31,007 NOx ton
# - a reference Emissions Factor for electricity generation of 2.114 ton/GWh, or 4.228 lb/MWh. 
# This analysis is detailed in Notebook https://www.kaggle.com/ajulian/eia-923-input-nox-emissions-and-ef-reference.
# 
# Unfortunately, I could not find the same information for Jan 2019 - Jun 2019, since it probably will not be available in EIA-923 for Puerto Rico before summer 2020.
# 
# <h2>Emissions information</h2>
# Global Emissions information must be taken from the provided SP5 dataset in the band "tropospheric vertical column density", since is the result of splitting the vertical NO2 column (after inferring it from the slant column and having into account things such as the Air Mass Factor. Good news we do not have to deal with it).
# 
# However, it is easy to realize there are other NO2 emission sources such as mobile fuel combustion from vehicles, at least. In order to characterize other emission sources and other gases I performed a brief analysis in Notebook https://www.kaggle.com/ajulian/activities-and-ghg-precursor-gases. This way I could obtain an estimate of emissions due to vehicles of 32,161 NOx ton per year.
# 
# <h2>Emissions factors</h2>
# Finally, I wrote another Notebook, https://www.kaggle.com/ajulian/remote-sensing-and-emissions-factors, where the EE NO2 dataset from S5P is analyzed and a framework is set to obtain Emissions Factors; it uses the wind information (which is represented with arrows, easy to check) to estimate where the Power Plant plume is.
# 
# Unfortunately, due to last minute problems, I only could calculate monthly Emissions Factors for Aguirre.
# 
# <h2>Possible improvements</h2>
# One of the ideas I did not have time to check (explained in https://www.kaggle.com/ajulian/activities-and-ghg-precursor-gases) is combining NO2 analysis with other GHG precursors, such as SO2 which is not produced by vehicles and could allow to isolate emissions of Power Plants in urban areas. By the way, the framework allows using SO2 as datasource.
# 
# Regarding the vehicle emissions estimation, I finally didn't had a clear idea about how to use it.
# 
# Finally there is much room to improve on image processing and tracking, to locate the plume more precisely.
# 
# <h2>Extension to other geographical areas</h2>
# Since the framework is based on EE, it can be used in another areas, as long as there is electricity generation information.

# In[ ]:




