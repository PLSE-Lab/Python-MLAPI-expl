#!/usr/bin/env python
# coding: utf-8

# This is my fourth kernel series contribution to <a href="https://www.kaggle.com/c/ds4g-environmental-insights-explorer/overview"> DS4G: Environmental Insights Explorer </a> to analyze the impacts of air pollution at a particular Geographical area and location
# 
# In this kernel, the Analysis report for conditions that would result in a higher/lower emissions factor for the sub-national region is detailed.
# 
# **Contents**
# * 1. <a href="#keyHighlights">Key Higlights</a>
# * 2. <a href="#annualhistorical">Deep exploration on conditions that would result in a higher/lower emissions factor</a> 
# 
#     * <a href="#countyAnalysis">Analysis by County sub national region polygons</a> 
#     * <a href="#100mile">Analysis within 100 mile buffer area from power plant</a>
#     * <a href="#50mile">Analysis within 50 mile buffer area from power plant</a>
# 

# <p id="keyHighlights" style="color:#703bdb; font-size:20px">Key Highlights</p>
# <hr size="30">
# * Performed analysis by considering multiple sub national regions such as county boundaries, 100 mile buffer area from power plant and 50 mile buffer area from power plant
# * Analysis are limited to power plants with Nuclear, Oil, Gas and Coal
# * Analysis says, emission factor is high in Vieques county with average emission factor of  28, followed by Guayanilla county. Weather conditions during this period will be detailed below
# * Extending the Analysis to smaller sub national region, prove that power plant Vieques EPP in Vieques county with average emission factor of 0.25 in 50 miles buffer area and average emission factor of 0.81 in 100 miles buffer area. Weather conditions during this period will be detailed below
# * Analysis identified smaller the geographical area accurate will be emission factor result
# 
# 
# 

# <p id="annualhistorical" style="color:#703bdb; font-size:20px">Deep exploration of annual historical emissions</p>
# <hr size="30">
#  All the analysis are first done at high level and then extendend to different sub region level. The main idea of this analysis is to measure following points :
#  
# * Identifying which Sub National region have high emissions and detail about weather condition?
# * During the period of year, does emission factor behave same across all sub national regions and and detail about weather condition
# * What type of power plant has emission factor has high and reasons?
# 
# 

# <p id="countyAnalysis" style="color:#703bdb; font-size:20px">Analysis by County sub national region polygons</p>
# <hr size="30">
# 
# For analysis purpose, we have considered below bands from GLDAS and GFS satelite imagery by verifying the data across mulitple bands. Example: min and max value of some bands is very high and it's not giving accurate result, hence we considered below bands for analysis which was giving decent result.
# 
# **Note: **The bands to be considered for analysis can be add any time or remove by changing below parameter in configuration file
# {
#     "plot": true
# }
# 
# **From GFS: **
# 
# Temperature 2m above ground (temp_2m_ag)
# 
# Specific humidity 2m above ground (spHum_2m_ag)
# 
# Precipitable water for entire atmosphere (precip)
# 
# **From GLDAS:**
# 
# Pressure (pSurf)
# 
# Heat flux (qG)
# 
# Wind speed (wind)
# 
# Analysis says, emission factor is high in Vieques county with average emission factor of  28, followed by Guayanilla county with average emission factor of 5. Analysis noticed, the emission factor as high during period Mar-2019 to June-2019 but higest recorded is during June 2019. While comparing weather conditions during this period from GLDAS and GFS satelite imagery it looks as below
# 
# **Temperature 2m above ground, Specific humidity 2m above ground, Precipitable water for entire atmosphere**  - These factors are least in the month of Feb-2019 and gradually increased while reaching June-2019. These are very low during the month of March but emission factor is going high, hence will be reducing concentration on this for now.
# 
# **Pressure** - This was high before Feb-2019 and started decreasing while reaching Feb-2019, while emission factor started increasing. study represents Pressure is inversly proportional to emission factor
# 
# Heat flux - While this is increasing emission factor is increasing along with this. This will be one of the important factor to be considered to control emission factor
# 
# **Wind Speed** - It's not showing much relationship with emission factor, hence we can ignore this.
# 
# In the month of May-2019, it looks like emission factor dropped little bit, while Heat flux is still high. In this condition, Precipitable water for entire atmosphere is dropped, while Temperature 2m above ground and Specific humidity 2m above ground in steady state. Hence, this is one of the factor that can be considered to control emission factor.
# 
# **Emission Factor Vs Weather:** 
# 
# when emission factor is high, how the values of GDAL & GFS looks like
# 
# ![emission%20high.PNG](attachment:emission%20high.PNG)
# 
# 
# ![weather_high.PNG](attachment:weather_high.PNG)
# 
# 
# 
# 
# when emission factor is low, how the values of GDAL & GFS looks like
# 
# ![emissionlow.PNG](attachment:emissionlow.PNG)
# 
# ![weather_low.PNG](attachment:weather_low.PNG)
# 
# 
# The plot has been done by scaling the values to make all observation overlap each other
# 
# 
# 
# ![Vieques.png](attachment:Vieques.png)
# 

# The output path for this data is available at below location
# 
# /kaggle/workspace/output/County/
# 
# Plots are availble at below location
# 
# /kaggle/workspace/output/County/Plot/WeatherAndEmission/

# <p id="100mile" style="color:#703bdb; font-size:20px">Analysis within 100 mile buffer area from power plant</p>
# <hr size="30">
# 
# Analysis says, emission factor is high power plant Vieques EPP power plant with average emission factor of  0.81, followed by Vega Baja power plant with average emission factor of 0.11. Analysis noticed, the emission factor as high during period Mar-2019 to June-2019 but higest recorded is during June 2019. While comparing weather conditions during this period from GLDAS and GFS satelite imagery it looks as below
# 
# **Temperature 2m above ground, Specific humidity 2m above ground, Precipitable water for entire atmosphere**  - These factors are least in the month of Feb-2019 and gradually increased while reaching June-2019. These are very low during the month of March but emission factor is going high, hence will be reducing concentration on this for now.
# 
# **Pressure** - This was high before Feb-2019 and started decreasing while reaching Feb-2019, while emission factor started increasing. study represents Pressure is inversly proportional to emission factor
# 
# Heat flux - While this is increasing emission factor is increasing along with this. This will be one of the important factor to be considered to control emission factor
# 
# **Wind Speed** - It's not showing much relationship with emission factor, hence we can ignore this.
# 
# In the month of May-2019, it looks like emission factor dropped little bit, while Heat flux is still high. In this condition, Precipitable water for entire atmosphere is dropped, while Temperature 2m above ground and Specific humidity 2m above ground in steady state. Hence, this is one of the factor that can be considered to control emission factor.
# 
# **Emission Factor Vs Weather:** 
# 
# when emission factor is high, how the values of GDAL & GFS looks like
# 
# ![emission_high_100mile.PNG](attachment:emission_high_100mile.PNG)
# 
# 
# 
# 
# ![weather_100mile_high.PNG](attachment:weather_100mile_high.PNG)
# 
# 
# 
# 
# when emission factor is low, how the values of GDAL & GFS looks like
# ![emissionlow_100mile.PNG](attachment:emissionlow_100mile.PNG)
# 
# ![weather_100mile_low.PNG](attachment:weather_100mile_low.PNG)
# 
# 
# The plot has been done by scaling the values to make all observation overlap each other
# 
# ![Vieques%20EPP.png](attachment:Vieques%20EPP.png)
# 

# <p id="50mile" style="color:#703bdb; font-size:20px">Analysis within 50 mile buffer area from power plant</p>
# <hr size="30">
# 
# 
# <a href="https://www.kaggle.com/nagabilwanth/4-analysis-for-higher-lower-emissions-continued">Continued here</a>
