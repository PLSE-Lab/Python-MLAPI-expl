#!/usr/bin/env python
# coding: utf-8

# This is my fifth kernel series contribution to <a href="https://www.kaggle.com/c/ds4g-environmental-insights-explorer/overview"> DS4G: Environmental Insights Explorer </a> to analyze the impacts of air pollution at a particular Geographical area and location
# 
# In this kernel, the Analysis report for Month-Month average historical emission factors for the sub-national region is detailed.
# 
# **Contents**
# * 1. <a href="#keyHighlights">Key Higlights</a>
# * 2. <a href="#annualhistorical">Deep exploration of month-month average historical emissions</a> 
# 
#     * <a href="#countyAnalysis">Analysis by County sub national region polygons</a> 
#     * <a href="#100mile">Analysis within 100 mile buffer area from power plant</a>
#     * <a href="#500mile">Analysis within 50 mile buffer area from power plant</a>

# <p id="keyHighlights" style="color:#703bdb; font-size:20px">Key Highlights</p>
# <hr size="30">
# * Performed analysis by considering multiple sub national regions such as county boundaries, 100 mile buffer area from power plant and 50 mile buffer area from power plant
# * Analysis are limited to power plants with Nuclear, Oil, Gas and Coal
# * Analysis identified, emission factor is high in Vieques county with average emission factor of  28, followed by Guayanilla county.
# * Extending the Analysis to smaller sub national region, prove that power plant Vieques EPP in Vieques county with average emission factor of 0.25 in 50 miles buffer area and average emission factor of 0.81 in 100 miles buffer area.
# * By extending the analysis to month-month, we identified maximum emission factor identified on Jun-05-2019 and on Jun-21-2019
# * Analysis identified smaller the geographical area accurate will be emission factor result

# <p id="annualhistorical" style="color:#703bdb; font-size:20px">Deep exploration of month-month average historical emissions</p>
# <hr size="30">
# *  Deep diving into historical emission factor for the month of June 2019 which as been recorded as a month having highest emission factor.
# 
# * Comparing emission factor with GLDAS and GFS satelite imagery and analysing the weather

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
# Analysis says, emission factor is high in Vieques county with average emission factor of  28, followed by Guayanilla county with average emission factor of 5. Analysis noticed, the emission factor as high during period Mar-2019 to June-2019 but higest recorded is during June 2019. Will go into deeper analysis how weather conditions and emission factor looks on each day of the June-2019.
# 
# On June 4-2019 and June 25-2019 - the emission factor recorded as high
# 
# On June 7-2019 - the emission factor recorded as low
# 
# 
# While comparing weather conditions during this period from GLDAS and GFS satelite imagery it looks as below
# 
# **Temperature 2m above ground, Specific humidity 2m above ground, Precipitable water for entire atmosphere**  - While analyzing the data between months, we discussed these factors are impacting least in controlling emission factor by comparing emission factor between Feb-2019 and June-2019 but while studying the same data at granular level for each date this is showing more relationship with relase of emissions. Let us analyse more here
# 
# From the begining of May to till end of May, Specific humidity 2m above ground, Temperature 2m above ground and Precipitable were recorded on same line constantly till the end of May but suddenly they got reduced but emission factor is recorded as high. 
# 
# On June-08-2019, the emission factor is low, but temperature, specific humidity and precipitation recorded high 
# 
# Hence these are the factors that can be considered to control emission factor
# 
# **Pressure** - While analyzing the data between months, we discussed Pressure is inversly proportional to emission factor but while comparing data at granular level we  it looks like pressure is recorded the directly proportional with emissions since while pressure recorded as low and temperature, humidity in rising state the emission factor is also recorded low. Hence these are the factors that can be considered to control emission factor
# 
# Heat flux - While this is increasing emission factor is increasing along with this. This will be one of the important factor to be considered to control emission factor. It recorded same while comparing data between months and days.
# 
# **Wind Speed** - It's not showing much relationship with emission factor, hence we can ignore this.
# 
# The plot has been done by scaling the values to make all observation overlap each other
# 
# ![Vieques_June-2019.png](attachment:Vieques_June-2019.png)

# <p id="100mile" style="color:#703bdb; font-size:20px">Analysis within 100 mile buffer area from power plant</p>
# <hr size="30">
# 
# Month-Month analysis within 100 mile buffer from power plant
# 
# The plot has been done by scaling the values to make all observation overlap each other
# 
# 
# ![Vieques%20EPP_June-2019.png](attachment:Vieques%20EPP_June-2019.png)

# <p id="50mile" style="color:#703bdb; font-size:20px">Analysis within 50 mile buffer area from power plant</p>
# <hr size="30">
# 
# 
# Month-Month analysis within 50 mile buffer from power plant
# 
# The plot has been done by scaling the values to make all observation overlap each other
# 
# 
# ![Vieques%20EPP_June-2019.png](attachment:Vieques%20EPP_June-2019.png)
# 
# 

# 
# The output path for this data is available at below location
# 
# /kaggle/workspace/output/County/
# /kaggle/workspace/output/powerplant_100miles/
# /kaggle/workspace/output/powerplant_50miles/
# 
# Plots are availble at below location
# 
# /kaggle/workspace/output/County/Plot/WeatherAndEmission/month
# /kaggle/workspace/output/powerplant_100miles/Plot/WeatherAndEmission/month
# /kaggle/workspace/output/powerplant_50miles/Plot/WeatherAndEmission/month
# 

# <a href="https://www.kaggle.com/nagabilwanth/6-methodology-calc-marginal-emissions-factor" >Next: Part 6: Methodology for calculating marginal emission factor for the sub-national region</a>
