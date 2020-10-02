#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# *  Kaggle's present working directory is /kaggle/input which was previously ../input.
# * People, who will be making new kernels or say notebook will also be printed with the directories where their data is present.

# ### Loading both the files from the source

# In[ ]:


reservoir_levels = pd.read_csv('/kaggle/input/chennai-water-management/chennai_reservoir_levels.csv')


# In[ ]:



reservoir_rainfalls = pd.read_csv('/kaggle/input/chennai-water-management/chennai_reservoir_rainfall.csv')


# In[ ]:


reservoir_levels.head()


# In[ ]:


reservoir_rainfalls.head()


# In[ ]:


# Reading the file with the population info of the 

chennai_population =  pd.read_csv('/kaggle/input/chennai-population/chennai_population.csv')


# In[ ]:


chennai_population.head()


# ### Checking the distribution of levels for 'POONDI' reservoir 

# In[ ]:


# importing style
from matplotlib import style

# using the style as ggplot
style.use("ggplot")

plt.figure(figsize=(40,14))

plt.plot(reservoir_levels['POONDI'],'b',linewidth = 3)

plt.xlabel("dates")
plt.ylabel("levels")


# ### Distribution of rainfall for 'POONDI' with the point graph and bargraph

# In[ ]:


plt.figure(figsize=(40,14))

plt.plot(reservoir_rainfalls['POONDI'],'o')

plt.xlabel("Days from year 2004 to 2019")
plt.ylabel("Rainfall in mcft")


# ---
# ### Below graph shows that how the rainfall happens only in few months

# In[ ]:


plt.figure(figsize=(40,14))
plt.bar(list(range(reservoir_rainfalls.shape[0])),reservoir_rainfalls['POONDI'])
plt.xlabel("Days of rainfall from year 2004 to 2019")
plt.ylabel("Rainfall in mcft")


# ---
# 1.  We have the data for the reservoirs with their information for levels and rainfall.
# 2.  To better analyze the data we need to seperate the whole data on the basis of year.
#     * For that we need to convert the date column in date format and then fetch the year and month from that.
#     * Then, after that we will be creating the two columns with the names month and year in our existing datasets****      

# In[ ]:


reservoir_levels['Date'] = pd.to_datetime(reservoir_levels['Date'],format="%d-%m-%Y",dayfirst=True)

reservoir_levels['year'], reservoir_levels['month'] = reservoir_levels['Date'].dt.year, reservoir_levels['Date'].dt.month


# In[ ]:


reservoir_levels.head()


# In[ ]:


# Means, data we have till june only for the year of 2019
reservoir_levels.tail()


# In[ ]:


# Seperating the data on the basis of years
reservoir_levels_2004 = reservoir_levels[reservoir_levels.year==2004]
reservoir_levels_2005 = reservoir_levels[reservoir_levels.year==2005]
reservoir_levels_2006 = reservoir_levels[reservoir_levels.year==2006]
reservoir_levels_2007 = reservoir_levels[reservoir_levels.year==2007]
reservoir_levels_2008 = reservoir_levels[reservoir_levels.year==2008]
reservoir_levels_2009 = reservoir_levels[reservoir_levels.year==2009]
reservoir_levels_2010 = reservoir_levels[reservoir_levels.year==2010]
reservoir_levels_2011 = reservoir_levels[reservoir_levels.year==2011]
reservoir_levels_2012 = reservoir_levels[reservoir_levels.year==2012]
reservoir_levels_2013 = reservoir_levels[reservoir_levels.year==2013]
reservoir_levels_2014 = reservoir_levels[reservoir_levels.year==2014]
reservoir_levels_2015 = reservoir_levels[reservoir_levels.year==2015]
reservoir_levels_2016 = reservoir_levels[reservoir_levels.year==2016]
reservoir_levels_2017 = reservoir_levels[reservoir_levels.year==2017]
reservoir_levels_2018 = reservoir_levels[reservoir_levels.year==2018]
reservoir_levels_2019 = reservoir_levels[reservoir_levels.year==2019]


# In[ ]:


# 2004 is a leap year
reservoir_levels_2004.shape


# In[ ]:


# 2005 is not a leap year
reservoir_levels_2005.shape


# In[ ]:


# plotting for 'POONDI' reservoir for year 2004
plt.figure(figsize=(30,14))
plt.plot(reservoir_levels_2004['POONDI'],'r',linewidth = 4)

plt.xlabel("All 366 days in year 2004")
plt.ylabel("Reservoirs level in mcft")
plt.title("Line graph of resevoir level for POONDI reservoir in year 2004 in MCFT")


# Above graph shows that, for 'POONDI' reservoir water level rises at the end of the year. 
# Reason for this could be rainfall. So, we will check the rainfall in 'POONDI' reservoir for the same year.
# 
# ---

# ---
# 1. Seperating the data of reservoir rainfalls as well on the basis of the year

# In[ ]:



reservoir_rainfalls['Date'] = pd.to_datetime(reservoir_rainfalls['Date'],format="%d-%m-%Y",dayfirst=True)


reservoir_rainfalls['year'], reservoir_rainfalls['month'] = reservoir_rainfalls['Date'].dt.year, reservoir_rainfalls['Date'].dt.month


# In[ ]:


reservoir_rainfalls.head()


# In[ ]:


reservoir_rainfalls.tail()


# In[ ]:


# Seperating the reservoir rainfall data on the basis of the year

reservoir_rainfalls_2004 = reservoir_rainfalls[reservoir_rainfalls.year==2004]
reservoir_rainfalls_2005 = reservoir_rainfalls[reservoir_rainfalls.year==2005]
reservoir_rainfalls_2006 = reservoir_rainfalls[reservoir_rainfalls.year==2006]
reservoir_rainfalls_2007 = reservoir_rainfalls[reservoir_rainfalls.year==2007]
reservoir_rainfalls_2008 = reservoir_rainfalls[reservoir_rainfalls.year==2008]
reservoir_rainfalls_2009 = reservoir_rainfalls[reservoir_rainfalls.year==2009]
reservoir_rainfalls_2010 = reservoir_rainfalls[reservoir_rainfalls.year==2010]
reservoir_rainfalls_2011 = reservoir_rainfalls[reservoir_rainfalls.year==2011]
reservoir_rainfalls_2012 = reservoir_rainfalls[reservoir_rainfalls.year==2012]
reservoir_rainfalls_2013 = reservoir_rainfalls[reservoir_rainfalls.year==2013]
reservoir_rainfalls_2014 = reservoir_rainfalls[reservoir_rainfalls.year==2014]
reservoir_rainfalls_2015 = reservoir_rainfalls[reservoir_rainfalls.year==2015]
reservoir_rainfalls_2016 = reservoir_rainfalls[reservoir_rainfalls.year==2016]
reservoir_rainfalls_2017 = reservoir_rainfalls[reservoir_rainfalls.year==2017]
reservoir_rainfalls_2018 = reservoir_rainfalls[reservoir_rainfalls.year==2018]
reservoir_rainfalls_2019 = reservoir_rainfalls[reservoir_rainfalls.year==2019]


# In[ ]:


plt.figure(figsize=(30,14))
plt.plot(reservoir_rainfalls_2004['POONDI'],'b',linewidth = 4)

plt.xlabel("All 366 days in year 2004")
plt.ylabel("Rainfall in mcft")
plt.title("Line graph of rainfall for POONDI reservoir in year 2004 in MCFT")


# * Comparing the data for both, reservoir levels and rainfall of 'POONDI' in year 2004
# 

# In[ ]:


plt.figure(figsize=(27,24))

plt.subplot(221)
plt.plot(reservoir_levels_2004['POONDI'],'r', label = "reservoir level",linewidth = 4)
plt.legend()

plt.subplot(222)
plt.plot(reservoir_rainfalls_2004['POONDI'],'o',label = "rainfall",linewidth = 4)
plt.legend()

plt.title("Comparision of 'POONDI' reservoir's levels VS rainfall")


# 1. We can see that the range in which rainfall has occurred, reservoir levels have started to rise.
# 2. Though, we can observe that not exactly in the same period reservoir levels have rose in which rainfall has occurred.
# 3. But, levels of reservoir have started to rise after a short while of rainfall. It may be because rain water will take sometime to reach reservoir. Hence, we can say that reservoir is getting water directly from rainfall as well as rainfall water may have been directed from other sources as well.
# ---

# * Poondi Lake or Sathyamoorthy reservoir is the reservoir across Kotralai River in Tiruvallur district of Tamil Nadu State. It acts as the important water source for Chennai city which is 60 km away.Frequent buses are available from Chennai and Tiruvallur to reach this place.
# * Poondi Reservoir (later named as Sathyamoorthy Sagar) was constructed in 1944 across the Kosathalaiyar River or Kotralai River in Thiruvallur district with a capacity of 2573 Mcft and placed in service for intercepting and storing Kosathalaiyar River water.

# ---
# ### Looking at all years reservoir level of 'POONDI'

# In[ ]:



# Setting the figure size
plt.figure(figsize=(27,24))

plt.subplot(431)
plt.plot(list(range(reservoir_levels_2004.shape[0])),reservoir_levels_2004['POONDI'],'b',label = "year 2004",linewidth = 4)
plt.legend()

plt.subplot(432)
plt.plot(list(range(reservoir_levels_2005.shape[0])),reservoir_levels_2005['POONDI'],'b',label = "year 2005",linewidth = 4)
plt.legend()

plt.subplot(433)
plt.plot(list(range(reservoir_levels_2006.shape[0])),reservoir_levels_2006['POONDI'],'b',label = "year 2006",linewidth = 4)
plt.legend()

plt.subplot(434)
plt.plot(list(range(reservoir_levels_2007.shape[0])),reservoir_levels_2007['POONDI'],'b',label = "year 2007",linewidth = 4)
plt.legend()

plt.subplot(435)
plt.plot(list(range(reservoir_levels_2008.shape[0])),reservoir_levels_2008['POONDI'],'b',label = "year 2008",linewidth = 4)
plt.legend()

plt.subplot(436)
plt.plot(list(range(reservoir_levels_2009.shape[0])),reservoir_levels_2009['POONDI'],'b',label = "year 2009",linewidth = 4)
plt.legend()

plt.subplot(437)
plt.plot(list(range(reservoir_levels_2010.shape[0])),reservoir_levels_2010['POONDI'],'b',label = "year 2010",linewidth = 4)
plt.legend()

plt.subplot(438)
plt.plot(list(range(reservoir_levels_2011.shape[0])),reservoir_levels_2011['POONDI'],'b',label = "year 2011",linewidth = 4)
plt.legend()

plt.subplot(439)
plt.plot(list(range(reservoir_levels_2012.shape[0])),reservoir_levels_2012['POONDI'],'b',label = "year 2012",linewidth = 4)
plt.legend()


# In[ ]:


plt.figure(figsize=(27,24))

plt.subplot(431)
plt.plot(list(range(reservoir_levels_2013.shape[0])),reservoir_levels_2013['POONDI'],'b',label = "year 2013",linewidth = 4)
plt.legend()

plt.subplot(432)
plt.plot(list(range(reservoir_levels_2014.shape[0])),reservoir_levels_2014['POONDI'],'b',label = "year 2014",linewidth = 4)
plt.legend()

plt.subplot(433)
plt.plot(list(range(reservoir_levels_2015.shape[0])),reservoir_levels_2015['POONDI'],'b',label = "year 2015",linewidth = 4)
plt.legend()

plt.subplot(434)
plt.plot(list(range(reservoir_levels_2016.shape[0])),reservoir_levels_2016['POONDI'],'b',label = "year 2016",linewidth = 4)
plt.legend()

plt.subplot(435)
plt.plot(list(range(reservoir_levels_2017.shape[0])),reservoir_levels_2017['POONDI'],'b',label = "year 2017",linewidth = 4)
plt.legend()

plt.subplot(436)
plt.plot(list(range(reservoir_levels_2018.shape[0])),reservoir_levels_2018['POONDI'],'b',label = "year 2018",linewidth = 4)
plt.legend()

plt.subplot(437)
plt.plot(list(range(reservoir_levels_2019.shape[0])),reservoir_levels_2019['POONDI'],'b',label = "year 2019",linewidth = 4)
plt.legend()


# 1. All the above graphs are of levels of 'POONDI' reservoir but all the graphs do not follow the same behaviour.
# 2. This shows that 'POONDI' reservoir have different water levels throughout the different year.
# 3. Hence, we can say that the reservoir's replenishing life cycle is not year dependent.
# 4. Or, we can say that replenishing cycle of 'POONDI' is not same as we have fixed season time every year.
# 5. So, there are high chances that reservoir water level depends on some other factor which is unpredictive.
# ---

# ### Looking at the behaviour of rainfall for 'POONDI' through out all the years  

# In[ ]:


plt.figure(figsize=(27,24))

plt.subplot(431)
plt.plot(list(range(reservoir_rainfalls_2004.shape[0])),reservoir_rainfalls_2004['POONDI'],'bo-',label = "year 2004",linewidth = 4)
plt.legend()
plt.ylim((0,310))

plt.subplot(432)
plt.plot(list(range(reservoir_rainfalls_2005.shape[0])),reservoir_rainfalls_2005['POONDI'],'bo-',label = "year 2005",linewidth = 4)
plt.legend()
plt.ylim((0,310))

plt.subplot(433)
plt.plot(list(range(reservoir_rainfalls_2006.shape[0])),reservoir_rainfalls_2006['POONDI'],'bo-',label = "year 2006",linewidth = 4)
plt.legend()
plt.ylim((0,310))

plt.subplot(434)
plt.plot(list(range(reservoir_rainfalls_2007.shape[0])),reservoir_rainfalls_2007['POONDI'],'bo-',label = "year 2007",linewidth = 4)
plt.legend()
plt.ylim((0,310))

plt.subplot(435)
plt.plot(list(range(reservoir_rainfalls_2008.shape[0])),reservoir_rainfalls_2008['POONDI'],'bo-',label = "year 2008",linewidth = 4)
plt.legend()
plt.ylim((0,310))

plt.subplot(436)
plt.plot(list(range(reservoir_rainfalls_2009.shape[0])),reservoir_rainfalls_2009['POONDI'],'bo-',label = "year 2009",linewidth = 4)
plt.legend()
plt.ylim((0,310))

plt.subplot(437)
plt.plot(list(range(reservoir_rainfalls_2010.shape[0])),reservoir_rainfalls_2010['POONDI'],'bo-',label = "year 2010",linewidth = 4)
plt.legend()
plt.ylim((0,310))

plt.subplot(438)
plt.plot(list(range(reservoir_rainfalls_2011.shape[0])),reservoir_rainfalls_2011['POONDI'],'bo-',label = "year 2011",linewidth = 4)
plt.legend()
plt.ylim((0,310))

plt.subplot(439)
plt.plot(list(range(reservoir_rainfalls_2012.shape[0])),reservoir_rainfalls_2012['POONDI'],'bo-',label = "year 2012",linewidth = 4)
plt.legend()
plt.ylim((0,310))


# In[ ]:


plt.figure(figsize=(27,24))

plt.subplot(431)
plt.plot(list(range(reservoir_rainfalls_2013.shape[0])),reservoir_rainfalls_2013['POONDI'],'bo-',label = "year 2013",linewidth = 4)
plt.legend()
plt.ylim((0,310))

plt.subplot(432)
plt.plot(list(range(reservoir_rainfalls_2014.shape[0])),reservoir_rainfalls_2014['POONDI'],'bo-',label = "year 2014",linewidth = 4)
plt.legend()
plt.ylim((0,310))

plt.subplot(433)
plt.plot(list(range(reservoir_rainfalls_2015.shape[0])),reservoir_rainfalls_2015['POONDI'],'bo-',label = "year 2015",linewidth = 4)
plt.legend()
plt.ylim((0,310))

plt.subplot(434)
plt.plot(list(range(reservoir_rainfalls_2016.shape[0])),reservoir_rainfalls_2016['POONDI'],'bo-',label = "year 2016",linewidth = 4)
plt.legend()
plt.ylim((0,310))

plt.subplot(435)
plt.plot(list(range(reservoir_rainfalls_2017.shape[0])),reservoir_rainfalls_2017['POONDI'],'bo-',label = "year 2017",linewidth = 4)
plt.legend()
plt.ylim((0,310))

plt.subplot(436)
plt.plot(list(range(reservoir_rainfalls_2018.shape[0])),reservoir_rainfalls_2018['POONDI'],'bo-',label = "year 2018",linewidth = 4)
plt.legend()
plt.ylim((0,310))

plt.subplot(437)
plt.plot(list(range(reservoir_rainfalls_2019.shape[0])),reservoir_rainfalls_2019['POONDI'],'bo-',label = "year 2019",linewidth = 4)
plt.legend()
plt.ylim((0,310))


# ---
# ### Comparing the reservoir levels of all the reservoirs year wise

# ---
# #### For year 2004

# In[ ]:


# width,height
plt.figure(figsize=(27,24))

plt.subplot(321)
plt.plot(list(range(reservoir_levels_2004.shape[0])),reservoir_levels_2004['POONDI'],'b',label = "POONDI reservoir levels in year 2004",linewidth = 4)
plt.legend()

plt.subplot(322)
plt.plot(list(range(reservoir_levels_2004.shape[0])),reservoir_levels_2004['CHOLAVARAM'],'r',label = "CHOLAVARAM reservoir levels in year 2004",linewidth = 4)
plt.legend()

plt.subplot(323)
plt.plot(list(range(reservoir_levels_2004.shape[0])),reservoir_levels_2004['REDHILLS'],'k',label = "REDHILLS reservoir levels in year 2004",linewidth = 4)
plt.legend()

plt.subplot(324)
plt.plot(list(range(reservoir_levels_2004.shape[0])),reservoir_levels_2004['CHEMBARAMBAKKAM'],'g',label = "CHEMBARAMBAKKAM reservoir levels in year 2004",linewidth = 4)
plt.legend()


plt.title("Comparision of reservoir levels of all the Reservoirs in year 2004")


# ---
# #### For year 2005

# In[ ]:


# width,height
plt.figure(figsize=(27,24))

plt.subplot(321)
plt.plot(list(range(reservoir_levels_2005.shape[0])),reservoir_levels_2005['POONDI'],'b',label = "POONDI reservoir levels in year 2005",linewidth = 4)
plt.legend()

plt.subplot(322)
plt.plot(list(range(reservoir_levels_2005.shape[0])),reservoir_levels_2005['CHOLAVARAM'],'r',label = "CHOLAVARAM reservoir levels in year 2005",linewidth = 4)
plt.legend()

plt.subplot(323)
plt.plot(list(range(reservoir_levels_2005.shape[0])),reservoir_levels_2005['REDHILLS'],'k',label = "REDHILLS reservoir levels in year 2005",linewidth = 4)
plt.legend()

plt.subplot(324)
plt.plot(list(range(reservoir_levels_2005.shape[0])),reservoir_levels_2005['CHEMBARAMBAKKAM'],'g',label = "CHEMBARAMBAKKAM reservoir levels in year 2005",linewidth = 4)
plt.legend()


plt.title("Comparision of reservoir levels of all the Reservoirs in year 2005")


# ---
# #### For 2006

# In[ ]:


# width,height
plt.figure(figsize=(27,24))

plt.subplot(321)
plt.plot(list(range(reservoir_levels_2006.shape[0])),reservoir_levels_2006['POONDI'],'b',label = "POONDI reservoir levels in year 2006",linewidth = 4)
plt.legend()

plt.subplot(322)
plt.plot(list(range(reservoir_levels_2006.shape[0])),reservoir_levels_2006['CHOLAVARAM'],'r',label = "CHOLAVARAM reservoir levels in year 2006",linewidth = 4)
plt.legend()

plt.subplot(323)
plt.plot(list(range(reservoir_levels_2006.shape[0])),reservoir_levels_2006['REDHILLS'],'k',label = "REDHILLS reservoir levels in year 2006",linewidth = 4)
plt.legend()

plt.subplot(324)
plt.plot(list(range(reservoir_levels_2006.shape[0])),reservoir_levels_2006['CHEMBARAMBAKKAM'],'g',label = "CHEMBARAMBAKKAM reservoir levels in year 2006",linewidth = 4)
plt.legend()


plt.title("Comparision of reservoir levels of all the Reservoirs in year 2006")


# * By above plots, we can infer that all the reservoirs have similar kind of reservoir levels in different years.
# * But, capacity or reservoir levels of 'POONDI', 'REDHILLS' and 'CHEMBARAMBAKKAM' are of same type by volume.
# * 'CHOLAVARAM' have lowest reservoir levels among all the reservoirs, the reason could be low capacity or less water is coming into it via different rivers or other sourecs.
# 
# ---

# ---
# Now, We will try to see the total distribution of levels on the basis of months, year wise.
# So, we need to group the data of reservoirs on the basis of months and then plot the bar graph to see the distribution.

# In[ ]:


reservoir_levels_grouped_2004 = reservoir_levels_2004.groupby('month').sum()


# In[ ]:


reservoir_levels_grouped_2004


# ---
# Plotting the Bar Graphs for all four reservoirs levels for year 2004  

# In[ ]:


plt.figure(figsize=(18,18))

plt.subplot(321)
plt.bar(list(range(1,13)),reservoir_levels_grouped_2004['POONDI'], label = "POONDI-2004")
plt.xlabel("Months from 1 to 12")
plt.ylabel("Total reservoir level for each Month")
plt.legend()

plt.subplot(322)
plt.bar(list(range(1,13)),reservoir_levels_grouped_2004['REDHILLS'], label = "REDHILLS-2004")
plt.xlabel("Months from 1 to 12")
plt.ylabel("Total reservoir level for each Month")
plt.legend()


plt.subplot(323)
plt.bar(list(range(1,13)),reservoir_levels_grouped_2004['CHEMBARAMBAKKAM'], label = "CHEMBARAMBAKKAM-2004")
plt.xlabel("Months from 1 to 12")
plt.ylabel("Total reservoir level for each Month")
plt.legend()


plt.subplot(324)
plt.bar(list(range(1,13)),reservoir_levels_grouped_2004['CHOLAVARAM'], label = "CHOLAVARAM-2004")
plt.xlabel("Months from 1 to 12")
plt.ylabel("Total reservoir level for each Month")
plt.legend()


# ---
# * Instead of Getting the data in the form of sum, it's better to get the mean of the data month wise**

# In[ ]:


reservoir_levels_grouped_mean_2004 = reservoir_levels_2004.groupby('month').mean()


# In[ ]:


# Custom Colors for all 12 different months

my_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
             '#911eb4', '#46f0f0', '#f032e6', '#000000', '#fabebe', '#008080', '#000075']


# In[ ]:


reservoir_levels_grouped_mean_2004


# In[ ]:


# It's better to compare the mean, year wise

plt.figure(figsize=(18,18))

plt.subplot(321)
plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2004['POONDI'], label = "POONDI-2004", color = my_colors)
plt.xlabel("Months from 1 to 12")
plt.ylabel("Total reservoir level for each Month")
plt.legend()

plt.subplot(322)
plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2004['REDHILLS'], label = "REDHILLS-2004", color = my_colors)
plt.xlabel("Months from 1 to 12")
plt.ylabel("Total reservoir level for each Month")
plt.legend()


plt.subplot(323)
plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2004['CHEMBARAMBAKKAM'], label = "CHEMBARAMBAKKAM-2004", color = my_colors)
plt.xlabel("Months from 1 to 12")
plt.ylabel("Total reservoir level for each Month")
plt.legend()


plt.subplot(324)
plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2004['CHOLAVARAM'], label = "CHOLAVARAM-2004", color = my_colors)
plt.xlabel("Months from 1 to 12")
plt.ylabel("Total reservoir level for each Month")
plt.legend()


plt.title("Month Wise Mean Distribution of reservoirs levels in 2004")


# ---
# 1. Now, Compring the mean reservoir levels for the year 2005 for all the months 

# In[ ]:


reservoir_levels_grouped_mean_2005 = reservoir_levels_2005.groupby('month').mean()


# In[ ]:


reservoir_levels_grouped_mean_2005


# In[ ]:



plt.figure(figsize=(18,18))

plt.subplot(321)
plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2005['POONDI'], label = "POONDI-2005", color = my_colors)
plt.xlabel("Months from 1 to 12")
plt.ylabel("Total reservoir level for each Month")
plt.legend()

plt.subplot(322)
plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2005['REDHILLS'], label = "REDHILLS-2005", color = my_colors )
plt.xlabel("Months from 1 to 12")
plt.ylabel("Total reservoir level for each Month")
plt.legend()


plt.subplot(323)
plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2005['CHEMBARAMBAKKAM'], label = "CHEMBARAMBAKKAM-2005", color = my_colors)
plt.xlabel("Months from 1 to 12")
plt.ylabel("Total reservoir level for each Month")
plt.legend()


plt.subplot(324)
plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2005['CHOLAVARAM'], label = "CHOLAVARAM-2005", color = my_colors)
plt.xlabel("Months from 1 to 12")
plt.ylabel("Total reservoir level for each Month")
plt.legend()


plt.title("Month Wise Mean Distribution of reservoirs levels in 2005")


# ---
# * Water Reservoir Level For Year 2006

# In[ ]:


reservoir_levels_grouped_mean_2006 = reservoir_levels_2006.groupby('month').mean()


# In[ ]:


reservoir_levels_grouped_mean_2006


# In[ ]:


# Plotting the reservoir bar for year 2006


plt.figure(figsize=(18,18))

plt.subplot(321)
plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2006['POONDI'], label = "POONDI-2006", color = my_colors)
plt.xlabel("Months from 1 to 12")
plt.ylabel("Total reservoir level for each Month")
plt.legend()

plt.subplot(322)
plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2006['REDHILLS'], label = "REDHILLS-2006", color = my_colors )
plt.xlabel("Months from 1 to 12")
plt.ylabel("Total reservoir level for each Month")
plt.legend()


plt.subplot(323)
plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2006['CHEMBARAMBAKKAM'], label = "CHEMBARAMBAKKAM-2006", color = my_colors)
plt.xlabel("Months from 1 to 12")
plt.ylabel("Total reservoir level for each Month")
plt.legend()


plt.subplot(324)
plt.bar(list(range(1,13)),reservoir_levels_grouped_mean_2006['CHOLAVARAM'], label = "CHOLAVARAM-2006", color = my_colors)
plt.xlabel("Months from 1 to 12")
plt.ylabel("Total reservoir level for each Month")
plt.legend()


plt.title("Month Wise Mean Distribution of reservoirs levels in 2006")


# ---
# **Looking at the Population Growth of the Chennai Along the Years**

# In[ ]:


chennai_population


# In[ ]:


chennai_population.dtypes


# In[ ]:


plt.figure(figsize=(15,15))
plt.plot(chennai_population['Year'],chennai_population['Population'],'-ob')
plt.xlim(1950,2050)
plt.ylim(1491293,15376000)
plt.xlabel("Years")
plt.ylabel("Population")
plt.title("Growth of Population with Respect to Years")
plt.show()


# #### We can see that population growth of chennai since 2000 to 2019 has been like exponential.
# #### Though according to recent data, rate of growth have gone down but overall population has increased, hence population has increased quite fast in the years from 2000 to 2019.
# #### Speculation and prediction also shows that population of chennai will keep increasing according to the given data which is actually the case in whole india.
# #### Considering our Country INDIA as well though our rate of growth is going down but over all our population will increase
# #### So, as more number of consumers will come and are coming then there will be heavy pressure on these reservoirs also to fullfill the needs of population in the coming years.
# #### So, population growth will also add onto the water problems on different levels.
# ---

# ---
# * We can see the relation between all the data we have. We can see how the data of all the reservoirs are correlated. Once we will see the data relation then we may be able to analyse one reservoir and can predict others.* 

# In[ ]:


reservoir_levels_data_only = reservoir_levels[['POONDI','CHOLAVARAM','REDHILLS','CHEMBARAMBAKKAM']]


# In[ ]:


reservoir_levels_data_only.head()


# In[ ]:


# Creating the correlation matrix
reservoir_data_corr_matrix = reservoir_levels_data_only.corr()


# In[ ]:


reservoir_data_corr_matrix


# In[ ]:


# For better analysis, we will create the heatmap
sns.heatmap(reservoir_data_corr_matrix)


# * As most of the reservoirs have correlation factor more than 0.75, we can say that these reservoirs have good and positive relation with each other.
# * We can also say that if one reservoir is replenishing then other reservoir will also replenish and one reservoir is getting empty then others will also.
# * We can verify this by using another test as well called as ANOVA test. But, for doing so, we must normalise all the data as well because data of different reservoir have different capacity

# In[ ]:





# In[ ]:




