#!/usr/bin/env python
# coding: utf-8

# # South African Crime Statistics (2005 - 2016)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Read CSV File and Add Totals Column 
population_stats = pd.read_csv('../input/ProvincePopulation.csv').sort_values('Population',ascending=False)
crime_stats = pd.read_csv('../input/SouthAfricaCrimeStats_v2.csv')
crime_stats['Total 2005-2016'] = crime_stats.sum(axis=1)
crime_stats.head()


# In[ ]:


# Group Crime Counts by Province
crimes_by_province = crime_stats.groupby(['Province'])['2005-2006','2006-2007','2007-2008','2008-2009',
                                              '2009-2010','2010-2011','2011-2012','2012-2013',
                                              '2013-2014','2014-2015','Total 2005-2016']


# In[ ]:


# Group Crime Counts by Category
crimes_by_category = crime_stats.groupby(['Category'])['2005-2006','2006-2007','2007-2008','2008-2009',
                                              '2009-2010','2010-2011','2011-2012','2012-2013',
                                              '2013-2014','2014-2015','Total 2005-2016']


# In[ ]:


# Group Crime Counts by Station
crimes_by_station = crime_stats.groupby(['Station'])['2005-2006','2006-2007','2007-2008','2008-2009',
                                              '2009-2010','2010-2011','2011-2012','2012-2013',
                                              '2013-2014','2014-2015','Total 2005-2016']


# In[ ]:


#Add counts, Reset Index & Sort by Total Crimes Between 2005-2016
province_totals = crimes_by_province.sum().reset_index().sort_values('Total 2005-2016',ascending=False)
category_totals = crimes_by_category.sum().reset_index().sort_values('Total 2005-2016',ascending=False)
station_totals = crimes_by_station.sum().reset_index().sort_values('Total 2005-2016',ascending=False)


# In[ ]:


# Create Total Stations by Province Dataframe
total_province_stations = pd.DataFrame(crime_stats['Province'].value_counts()).reset_index()
total_province_stations['Total Stations'] = total_province_stations['Province']
total_province_stations.drop('Province',axis=1,inplace=True)
total_province_stations['Province'] = total_province_stations['index']
total_province_stations.drop('index',axis=1,inplace=True)


# In[ ]:


# Create All Province Totals Dataframe (Crime + Population Data)

# Set Index To Province (To add totals)
province_totals.set_index('Province',inplace=True)
total_province_stations.set_index('Province',inplace=True)
population_stats.set_index('Province',inplace=True)

# Add Totals to province_totals Dataframe
province_totals['Total Stations'] = total_province_stations['Total Stations']
province_totals['Population'] = population_stats['Population']
province_totals['Area'] = population_stats['Area']
province_totals['Density'] = population_stats['Density']

# Reset index back
province_totals = province_totals.reset_index()
total_province_stations = total_province_stations.reset_index()
population_stats = population_stats.reset_index()


# In[ ]:


plt.figure(figsize=(14,8)) # this creates a figure 14 inch wide, 8 inch high
ax = sns.barplot(data=total_province_stations,x='Province',y='Total Stations')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",fontsize=10,)
ax.set_title('Number of Police Stations by Province',fontsize=16)
ax.set_ylabel('Number of Police Stations',fontsize=14)
ax.set_xlabel('South African Provinces',fontsize=14)


# In[ ]:


plt.figure(figsize=(14,8)) # this creates a figure 14 inch wide, 8 inch high
ax = sns.barplot(data=province_totals,x='Province',y='Total 2005-2016')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",fontsize=10,)
ax.set_title('Total Crimes Reported by Province (2005-2016)',fontsize=16)
ax.set_ylabel('Number of Crimes Reported',fontsize=14)
ax.set_xlabel('South African Provinces',fontsize=14)


# In[ ]:


# Plot Figure
plt.figure(figsize=(14,10)) # this creates a figure 14 inch wide, 10 inch high
ax = sns.barplot(data=category_totals,y='Category',x='Total 2005-2016',palette='coolwarm')
ax.set_title('Total Crimes Reported by Category (2005-2016)',fontsize=16)
ax.set_ylabel('Type of Crime',fontsize=14)
ax.set_xlabel('Number of Crimes Reported',fontsize=14)


# In[ ]:


plt.figure(figsize=(14,8)) # this creates a figure 14 inch wide, 8 inch high
ax = sns.barplot(data=station_totals.head(30),x='Station',y='Total 2005-2016', palette='coolwarm')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",fontsize=10,)
ax.set_title('Total Crimes Reported by Top 30 Police Stations (2005-2016)',fontsize=16)
ax.set_ylabel('Number of Crimes Reported',fontsize=13)
ax.set_xlabel('South African Police Stations',fontsize=14)


# In[ ]:


f,ax = plt.subplots(figsize=(10, 8))
ax.set_title('Correlation Heatmap',fontsize=16)
corr = province_totals.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# #### From the above Correlation Heatmap, there is clearly a strong positive correlation between the total amount of crimes committed compared to the population and density of a Province. There is also a positive correlation between the total number of police stations in a province compared to the total amount of crimes. This does not mean that there is higher crime because there are more police stations but perhaps there are more stations to combat the higher amount of crime. It can also be observed that there is a negative correlation between the area size of a province compared to the number of crimes committed. All this can be further illustrated with a Correlation Gradient below:

# In[ ]:


corr.style.background_gradient().set_precision(2) # Set precision to 2 decimals

