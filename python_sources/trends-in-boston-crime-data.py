#!/usr/bin/env python
# coding: utf-8

# Trends in Boston Crime Data. 
# 
# **Aims **
# 
# * Identify Geolocation trends over time.
# * Identify the best and worst place to stay safe from **most happening crimes** in Boston based on the data.

# In[ ]:


#If you are facing issue with workspace showing loading sign and not actually loading the data.
import os
os.listdir("../input") 
os.listdir("../input/crimes-in-boston") 


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Reading Data set
dataSet = pd.read_csv('../input/crimes-in-boston/crime.csv', encoding='latin-1')

offenseCode = pd.read_csv('../input/crimes-in-boston/offense_codes.csv',encoding='Windows-1252')

dataSet.head()

# Any results you write to the current directory are saved as output.


# In[ ]:


offenseCode.head()


# Data preprocessing to bring all the relevant data into one place.

# In[ ]:


#Merging the dataSet to add Offense_Code_Names. 
dataSet = pd.merge(dataSet, offenseCode, how= "left", left_on= "OFFENSE_CODE", right_on = "CODE")
#Add extra info of district 
districtData = pd.DataFrame({'DISTRICT':['A1','A15','A7', 'B2','B3', 'C11', 'C6', 'D14', 'D4', 'E13', 'E18', 'E5',''], 'NAMES':['Downtown','Charlestown','East Boston','Roxbury', 'Mattapan', 'Dorchester', 'South Boston', 'Brighton', 'South End', 
                           'Jamaica Plain', 'Hyde Park', 'West Roxbury', 'Unknown Location']})

#Merging the dataset based on the district key 
dataSet = dataSet.merge(districtData, on = "DISTRICT")
#Renaming the data columns 
dataSet=dataSet.rename(columns= {"NAMES" : "DISTRICT_NAMES","NAME":"CRIME_NAME"})

#Droping extra duplicate data due to merge
dataSet = dataSet.drop(['CODE'], axis = 1)
 
dataSet.head()


# Now we have introduced new fields such as "Crime_Name" and "Districit_Names".

# In[ ]:


#Further Data Preprocessing to remove Lat and Long values with -1
dataSet.Lat.replace(-1, None, inplace=True)
dataSet.Long.replace(-1, None, inplace=True)

#For visual Data Analysis
import seaborn as sns


# Trends in Crime Based on Geo Location data. 

# In[ ]:


#Subsetting the data for the year 2016
#Plotting the data based on the crime location and District it took place.
#Alpha with such small value helps us understand the contribution each District had in 2016 to total crimes
#This code can be used for any other year 
year=2016
g= sns.scatterplot(x="Lat",y="Long",data = dataSet[dataSet['YEAR']==year], 
				alpha=0.01, hue="DISTRICT_NAMES")
g.set_title("District Contribution 2016")
g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)


# **What has been trend of contribution to overall crime based on the location over the years ?
# **
# As we have data for 4 years. We can look into the trend by plotting scatter plot for each year side by side. 

# In[ ]:


#We will be using FaceGrid for plotting several scatter plots side by side for analysis
graph_grid_years = sns.FacetGrid(dataSet, col = "YEAR", hue="DISTRICT_NAMES", margin_titles=True)
graph_grid_years.map(plt.scatter,"Lat","Long",alpha=0.01,)
graph_grid_years.add_legend()


# In[ ]:


#Trend over the years for specific months 
year = 2017
grid_year_2017 = sns.FacetGrid(dataSet[dataSet["YEAR"]==year], 
								 col = "MONTH", col_wrap=4, hue="DISTRICT_NAMES")
grid_year_2017.map(plt.scatter,"Lat","Long",alpha=0.005,)
grid_year_2017.add_legend(framealpha=1)


# In[ ]:


#Trend for any district over the years and month 
#For this example we are looking into Downtown.
district = "Downtown"

graph_grid_years_area = sns.FacetGrid(dataSet[dataSet["DISTRICT_NAMES"]==district]
, row ="MONTH",col="YEAR" ,hue="UCR_PART")
#Setting the lims to ignore the outliers 
graph_grid_years_area.set(ylim = (-71.08,-71.04),xlim = (42.34,42.37))
graph_grid_years_area.map(plt.scatter,"Lat","Long",alpha=0.1,)
graph_grid_years_area.add_legend()


# Visualizing the data for downtown we can see that there is high concentration of UCR_Part2 Crimes. Furthermore areas under the Lat **(42.35-42.36)** and **Long (-71.070 to -71.055)** has high contribution to such crimes.     

# **What is the safest day to be out in the streets in Boston ? **
# 

# In[ ]:


plt.figure(figsize=(15,5))
cPlot=sns.countplot(y="DAY_OF_WEEK",data=dataSet, 
			  order=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])


# Based on the frequencey of the crimes, weekend appears to relatively safest day to be out in Boston. 

# In[ ]:



uniqueIncidentCodes = dataSet['OFFENSE_CODE_GROUP'].unique()
#Forming the crosstab to extract MOST happening crimes.
crime_count = pd.crosstab(dataSet['OFFENSE_CODE_GROUP'],dataSet["YEAR"],values=dataSet.YEAR
						  ,aggfunc='count').reset_index()
#Sorting the values for each year

crime_count = crime_count.sort_values(by=[2015,2016,2017,2018],ascending= [0,0,0,0])
#Most Happening Crimes top 6
crime_count_worst = crime_count.head(6) 
crime_count_worst


# **What is the list of top happening crimes in Boston ? **

# In[ ]:


name_worst_crime = crime_count_worst["OFFENSE_CODE_GROUP"]
name_worst_crime


# Distribution of the crime over different hours of the day

# In[ ]:


#Overall Crime hour
plt.figure(figsize=(15,5))
cPlot=sns.countplot(x="HOUR",data=dataSet)


# Relative Safest time to be out in Boston is around 1:00 AM-8:00 AM 

# In[ ]:


#Sub division to avoid worst crime 
cPlot = sns.catplot(x="HOUR",col="OFFENSE_CODE_GROUP",kind="count",col_wrap=2,
					data= 
					dataSet[dataSet["OFFENSE_CODE_GROUP"].isin(name_worst_crime)] )


# Extracting the list of districts with top contribution in the worst crime (by numbers)

# In[ ]:


#Distrcit Contribution to Worst Crime by number of happenings
plt.figure(figsize=(15,5))
wc_dataSet = dataSet[dataSet["OFFENSE_CODE_GROUP"].isin(name_worst_crime)]
cPlot=sns.countplot(y="DISTRICT_NAMES",data=wc_dataSet)


# Graph reflects contribution of each district to most happening crimes in Boston. Relatively **Charlestown** seems like a best place to be while in Boston.

# In[ ]:


#Top Distrcit Contributions to Worst Crime by number of happenings
d_contrib = pd.crosstab(wc_dataSet["DISTRICT_NAMES"],wc_dataSet["OFFENSE_CODE_GROUP"]).reset_index()
d_contrib = d_contrib.sort_values(by=name_worst_crime.head(1).iloc[0], ascending= [0])
#List of worst districts to be in Boston 
name_worst_district = d_contrib.head(6)["DISTRICT_NAMES"]
name_worst_district
#This list can be further used to do analysis 


# As it turns out. **South End** has the largest contribution in happening of **Larceny**. 

# In[ ]:


#Narrow down to worst District and worst Crime DataSet
wc_wd_dataSet = wc_dataSet[wc_dataSet["DISTRICT_NAMES"].isin(name_worst_district)]
#This data set contains the districts with the highest contribution to the top happening crimes in boston. 
wc_wd_dataSet

#Time Division to top worst crime in Top Worst Distrcit 
plt.figure(figsize=(15,5))
cPlot=sns.countplot(x="HOUR",data=wc_wd_dataSet[(wc_wd_dataSet["OFFENSE_CODE_GROUP"]
==name_worst_crime.head(1).iloc[0]) & (wc_wd_dataSet["DISTRICT_NAMES"]
==name_worst_district.head(1).iloc[0] )])
cPlot.set_title("Crime by Hour in worst district ( "+name_worst_district.head(1).iloc[0]+" )")



# This analysis is a step to look into Boston crime dataset based on the presepective of most happening crime to be the worst of all. Another prespective is to look into the crime data based on the categories defined by UCR.
# 
# Thank you.
