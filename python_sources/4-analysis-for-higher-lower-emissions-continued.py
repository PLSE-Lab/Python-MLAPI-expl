#!/usr/bin/env python
# coding: utf-8

# <p id="50mile" style="color:#703bdb; font-size:20px">Analysis within 50 mile buffer area from power plant</p>
# <hr size="30">
# Analysis says, emission factor is high power plant Vieques EPP power plant with average emission factor of  0.26, followed by Vega Baja power plant with average emission factor of 0.04. Analysis noticed, the emission factor as high during period Mar-2019 to June-2019 but higest recorded is during June 2019. While comparing weather conditions during this period from GLDAS and GFS satelite imagery it looks as below
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
# ![emission%20high_50mile.PNG](attachment:emission%20high_50mile.PNG)
# 
# 
# 
# 
# 
# 
# 
# ![weather%2050mile_low.PNG](attachment:weather%2050mile_low.PNG)
# 
# 
# 
# 
# when emission factor is low, how the values of GDAL & GFS looks like
# ![emission%20low_50mile.PNG](attachment:emission%20low_50mile.PNG)
# 
# ![weather%2050mile_low.PNG](attachment:weather%2050mile_low.PNG)
# 
# 
# 
# The plot has been done by scaling the values to make all observation overlap each other
# 
# ![Vieques%20EPP.jpg](attachment:Vieques%20EPP.jpg)

# <a href="https://www.kaggle.com/nagabilwanth/5-analysis-report-for-month-month-avg-historical" >Next: Part 5: Analysis report for Month-Month average historical emission factor</a>
