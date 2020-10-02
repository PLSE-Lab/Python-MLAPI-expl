#!/usr/bin/env python
# coding: utf-8

# This is my third kernel series contribution to <a href="https://www.kaggle.com/c/ds4g-environmental-insights-explorer/overview"> DS4G: Environmental Insights Explorer </a> to analyze the impacts of air pollution at a particular Geographical area and location
# 
# In this kernel, the Analysis report for annual historical emissions factor for the sub-national region is detailed.
# 
# **Contents**
# * 1. <a href="#keyHighlights">Key Higlights</a>
# * 2. <a href="#annualhistorical">Deep exploration of annual historical emissions</a> 
# 
#     * <a href="#overview">Overview of analysis</a>   
#     * <a href="#countyAnalysis">Analysis by County sub national region polygons</a> 
#     * <a href="#100mile">Analysis within 100 mile buffer area from power plant</a>
#     * <a href="#50mile">Analysis within 50 mile buffer area from power plant</a>
# 

# <p id="keyHighlights" style="color:#703bdb; font-size:20px">Key Highlights</p>
# <hr size="30">
# * Performed analysis by considering multiple sub national regions such as county boundaries, 100 mile buffer area from power plant and 50 mile buffer area from power plant
# * Analysis are limited to power plants with Nuclear, Oil, Gas and Coal
# * Analysis says, emission factor is high in Vieques county with average emission factor of  28, followed by Guayanilla county.
# * Extending the Analysis to smaller sub national region, prove that power plant Vieques EPP in Vieques county with average emission factor of 0.25 in 50 miles buffer area and average emission factor of 0.81 in 100 miles buffer area.
# * Analysis identified smaller the geographical area accurate will be emission factor result
# 
# 
# 

# <p id="annualhistorical" style="color:#703bdb; font-size:20px">Deep exploration of annual historical emissions</p>
# <hr size="30">
#  All the analysis are first done at high level and then extendend to different sub region level. The main idea of this analysis is to measure following points :
#  
# * Identifying which Sub National region have high emissions?
# * During the period of year, does emission factor behave same across all sub national regions
# * What type of power plant has emission factor has high?
# 
# 

# <p id="overview" style="color:#703bdb; font-size:20px">Overview of analysis</p>
# <hr size="30">
# 
# Let's look at an high level overview of the power plants in different counties. The following plot shows the counties available in different counties. The leged of the graph is explained in following representations:
# 
# * County polygon - <font color='brown'> Brown color </font>
# * 100 mile buffer polygon from power plant -<font color='blue'> Blue color </font>
# * 50 mile buffer polygon from power plant -<font color='purple'> Purple color </font>
# * Power plant -<font color='green'> Green color </font>
# * County polygon highlighted in  - <font color='yellow'> Yellow color </font> are top two counties having maximum emission factor
# 
# ![image.png](attachment:image.png)
# 

# **Inferences**
# * The Analysis to sub national region is limited to county polygons that are intersecting with power plants, since to calulate emission factor amount of power generated in that region is required.
# * The analysis are limited to power plants that use fuel type Nuclear, Oil, Gas and Coal, since the emission factor value is too high for wind, solar and hydro. This should should usually. Either the data received is something not good.
# * The emission factor is high during the month of June-2019 compared to other months
# * The emission factor is low during the month of February-2019 compared to other months

# <p id="countyAnalysis" style="color:#703bdb; font-size:20px">Analysis by County sub national region polygons</p>
# <hr size="30">
# 
# Analysis says, emission factor is high in Vieques county with average emission factor of  28, followed by Guayanilla county with emission factor of 5. It looks like the power plant data received has lot of vulnerabilities hence the calculated value of emission factor is not as expected. It's usually showing the emission factor as high for the power plants having low power generation, which is not correct I guess. To have better accuracy of emission factor we have excluded the power plants Wind, Hydro and solar from analysis. Including them back into the analysis is a configuration change. No code change is required.
# 
# To verify this we have done several analysis like multiplying emissions with fuel used per unit of power generation, which also did not worked. Finally we came to a decission to skip below fuels.
# 
# ![image.png](attachment:image.png)
# 
# 

# 
# Emission Factor below shown for Arecibo county polygon
# This data available at below path in output folder with prefix RasterEmission
# 
# /kaggle/workspace/output/County/
# 
# ![Arecibo.PNG](attachment:Arecibo.PNG)
# 
# 

# Below is the analysis report of the power plants noticed across the counties from July-2018 to June-2019.
# At below path you can find the output reports generated
# /kaggle/workspace/output/County/Plot/
# 
# ![image.png](attachment:image.png)
# 
# 
# 
# The emission factor is high during the month of June-2019 compared to other months.
# 
# Note: The values on plot has been scaled same measurement for better comparision and visibility.
# 
# ![Vieques.png](attachment:Vieques.png)

# <p id="100mile" style="color:#703bdb; font-size:20px">Analysis within 100 mile buffer area from power plant</p>
# <hr size="30">
# 
# <a href="https://www.kaggle.com/nagabilwanth/3-annual-hist-emission-continued">Continued here</a>
# 

# <p id="50mile" style="color:#703bdb; font-size:20px">Analysis within 50 mile buffer area from power plant</p>
# <hr size="30">
# 
# <a href="https://www.kaggle.com/nagabilwanth/3-annual-hist-emission-continued">Continued here</a>
