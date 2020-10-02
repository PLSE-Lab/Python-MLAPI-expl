#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import folium
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Read the files**

# In[ ]:


crime = pd.read_csv('../input/crime.csv', encoding='ISO-8859-1')
offense = pd.read_csv('../input/offense_codes.csv', encoding='ISO-8859-1')


# It is interesting to see this few shootings for the period of 4 years. I expected more to be honest.

# In[ ]:


crime.info()


# In[ ]:


crime.head()


# Pay careful attention to the dates those crimes happend. For year 2015 and 2018, the records aren't complete. It is clear that
# year 2015 only has data for months after May, while year 2018 only has data for months before October.
# 
# The result of analysis performed using year 2015 and 2018 in combination with year 2016 and 2017's data will probablity not reflect
# much useful information since the data would be skewed. So I opted to not use any records from year 2015 and 2018.

# In[ ]:


print('Year 2015\'s record include months: ', set(crime[crime['YEAR'] == 2015]['MONTH']))
print('Year 2018\'s record include months: ', set(crime[crime['YEAR'] == 2018]['MONTH']))


# In[ ]:


crime['OFFENSE_DESCRIPTION'].unique()


# ### Text Cleansing
# 
# Before jumping into the analysis or visualization part, I want to do some text cleasing to re-arrange the 
# data and only discard the unwanted columns and rows.

# In[ ]:


# create a copy of the original dataframe, only perserving data from year 2016 and 2017 and wanted columns
crime_new = crime.query('YEAR == 2016 or YEAR == 2017')[['OFFENSE_DESCRIPTION', 'SHOOTING', 'OCCURRED_ON_DATE', 'YEAR', 'MONTH', 'DAY_OF_WEEK', 'HOUR', 'Lat', 'Long']]
# rename the columns 
crime_new.rename(columns={'OFFENSE_DESCRIPTION':'OFFENSE', 'OCCURRED_ON_DATE':'DATE'}, inplace=True)
# convert dates in string format to DatetimeIndex and set it as the new index for the dataframe
dates = pd.to_datetime(crime_new['DATE'])
crime_new.drop(axis=1, columns=['DATE'], inplace=True)
crime_new['DATE'] = dates
crime_new.set_index(keys=['DATE'], drop=True, inplace=True)
# sort the dataframe by index
crime_new.sort_index(axis=0, ascending=True, inplace=True)


# In[ ]:


print('There are {} types of crimes in this dataset'.format(len(set(crime['OFFENSE_DESCRIPTION']))))


# There are **244** types of crimes recorded in this dataset. However, I'm only interested in the following 6 major categories:
# * Auto Theft related crimes
# * Weapon related crimes
# * Child related crimes
# * Larceny
# * Assault
# * Robbery
# 
# Since I'm only interested in 6 categories of crimes, I'll use regex to rename any offenses that has the following keywords in its description in order to simplify future data extraction.
# 
# At last, I created a new dataframe named 'crime_df'. This dataframe contains all rows that are associated with the 6 crime categories happened in year 2016 and 2017.
# 
# All of my analysis and visualization will be focused on these categories of crime.

# In[ ]:


crime_new.replace({'OFFENSE':[r'LARCENY [\w\W]+',
                            r'ASSAULT [\w\W]+',
                            r'AUTO THEFT [\w\W]+',
                            r'CHILD [\w\W]+',
                            r'ROBBERY [\w\W]+',
                            r'WEAPON [\w\W]+']},
                   {'OFFENSE':['LARCENY',
                            'ASSAULT',
                            'AUTO THEFT',
                            'CHILD',
                            'ROBBERY',
                            'WEAPON']}, 
                            regex=True,
                            inplace=True)

crime_df = crime_new[(crime_new['OFFENSE'] == 'LARCENY')|(crime_new['OFFENSE'] == 'ASSAULT')|
                    (crime_new['OFFENSE'] == 'AUTO THEFT')|(crime_new['OFFENSE'] == 'CHILD')|
                    (crime_new['OFFENSE'] == 'ROBBERY')|(crime_new['OFFENSE'] == 'WEAPON')]


# In[ ]:


# also, replace all NaN values in the SHOOTING column to string 'N'
crime_df.fillna({'SHOOTING': 'N'}, inplace=True)


# ### Geographic Information
# 
# In this section I will map out the crimes using Folium's heatmap to provide a clearer understanding on the frequency,
# intensity, and categories of crime that happened in Boston in year 2016 and 2017.
# 
# There are 5 layers of heatmap, each corresponds to 1 of the 5 main crime category I mentioned earlier. It is clear that the 
# distribution of crimes vary depending on their category and general location in terms of district. 
# 
# I tried to use Choropleth to visualize this however, the GIS information that I extracted from website Analyze Boston - Zoning District contain more districts than what this dataset has. So there are large areas with no inputs. That's why I had to with the heatmap approach.

# In[ ]:


Larceny = crime_df.query('OFFENSE == \'LARCENY\'').dropna()
Assault = crime_df.query('OFFENSE == \'ASSAULT\'').dropna()
Auto = crime_df.query('OFFENSE == \'AUTO THEFT\'').dropna()
Child = crime_df.query('OFFENSE == \'CHILD\'').dropna()
Robbery = crime_df.query('OFFENSE == \'ROBBERY\'').dropna()
Weapon = crime_df.query('OFFENSE == \'WEAPON\'').dropna()


# In[ ]:


from folium.plugins import HeatMap
#from folium.plugins import FastMarkerCluster, MarkerCluster

# map object's starting location
map = folium.Map(location=[42.306821, -71.060300],
                 zoom_start=12,
                 tiels='OpenStreetMap')

HeatMap(Auto[['Lat', 'Long']],
        min_opacity=0.8,
        radius=8,
        name='Auto Theft').add_to(map)

HeatMap(Larceny[['Lat', 'Long']],
        min_opacity=0.8,
        radius=8,
        show=False,
        name='Larceny').add_to(map)

HeatMap(Assault[['Lat', 'Long']],
        min_opacity=0.8,
        radius=8,
        show=False,
        name='Assault').add_to(map)

HeatMap(Child[['Lat', 'Long']],
        min_opacity=0.8,
        radius=8,
        show=False,
        name='Child').add_to(map)

HeatMap(Robbery[['Lat', 'Long']],
        min_opacity=0.8,
        radius=8,
        show=False,
        name='Robery').add_to(map)

HeatMap(Weapon[['Lat', 'Long']],
        min_opacity=0.8,
        radius=8,
        show=False,
        name='Weapon').add_to(map)


"""cluster = FastMarkerCluster(Auto_coord,
                            name='Auto Theft').add_to(map)"""


folium.LayerControl().add_to(map);


# In[ ]:


display(map)


# ### Exploratory Analysis

# In this section I will uses graphical illustrations to visualize data. I'm particularly interested in questions like: which month had the most crime occur? which day of week had crimes occur most frequently, etc. 
# 
# Please note, these questions are answered with regard to the previously categorized 6 crime categories:
# **Larceny, Auto Theft, Assault, Robbery, Child,** and **Weapon**.
# 
# I'll try to answer the above questions along with other analysis. This is an ongoing personal project, so I'll come back and do more as time goes.

# In[ ]:


# set the plot style
import seaborn as sns


# In[ ]:


months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

grid1 = sns.catplot(x='MONTH', kind="count", palette="ch:.25", data=crime_df, aspect=2);
grid1.ax.set_title('Crimes by Month', fontsize=14)
grid1.ax.set_ylabel('NUMBER OF CRIMES')
grid1.ax.set_xticklabels(months);
grid1.ax.tick_params(axis='x', rotation=45)


# For the graph above, it looks like that the crime distribution is uniform across all months. Feburary had the least amount of crimes occur but that might be due to the fact that it has the least number of calander days to begin with.

# In[ ]:


palette = sns.cubehelix_palette(n_colors=7, start=2.5, rot=0, light=0.9)
order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

grid3 = sns.catplot(x='DAY_OF_WEEK', kind='count', data=crime_df, order=order, palette=palette, aspect=1.5)
grid3.ax.set_xlabel('DAY')
grid3.ax.set_ylabel('NUMBER OF CRIMES')
grid3.ax.set_title('Crimes by Day of a Week', fontsize=14);


# In[ ]:


palette = sns.cubehelix_palette(n_colors=24, start=3, rot=0, light=0.9)

grid2 = sns.catplot(x='HOUR', kind='count', data=crime_df, palette=palette, aspect=1.5)
grid2.ax.set_xlabel('HOUR')
grid2.ax.set_ylabel('NUMBER OF CRIMES')
grid2.ax.set_title('Crimes by Hour of a Day', fontsize=14);


# We can see that the crimes are concentrated at night time

# Similar to the previous graph, the amount of crimes occurred on the different days of a week also does not vary much. Friday seemed to have accumulated a higher amount of crimes. This could potentially indicate something. However, at this stage, I don't want to draw my conclusion too early. The granularity of the data being analyzed at this moment is still too coarse. 
# 
# In the following part, I'll look at the data at a finer level, by their crime category.

# In[ ]:


Larceny_num = [len(Larceny[Larceny['MONTH'] == month]) for month in range(1,13)]
Assault_num = [len(Assault[Assault['MONTH'] == month]) for month in range(1,13)]
Auto_num = [len(Auto[Auto['MONTH'] == month]) for month in range(1,13)]
Robbery_num = [len(Robbery[Robbery['MONTH'] == month]) for month in range(1,13)]
Child_num = [len(Child[Child['MONTH'] == month]) for month in range(1,13)]
Weapon_num = [len(Weapon[Weapon['MONTH'] == month]) for month in range(1,13)]

crime_category = [Larceny_num, Assault_num, Auto_num, Robbery_num, Child_num, Weapon_num]
legends = ['Larceny', 'Assault', 'Auto Theft', 'Robbery', 'Child', 'Weapon']
months_numeric = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
plt.figure(figsize=(12,8))
for item in crime_category:
    sns.lineplot(x=months_numeric, y=item)
#plt.legend(legends, loc='lower center')
plt.legend(legends, bbox_to_anchor=(1.02, 1), ncol=1, loc=2)
plt.xticks(ticks=months_numeric, labels=months);
plt.grid(b=True, axis='x', alpha=0.5)
plt.title('Crimes by Month by Category', fontsize=14)
plt.xlabel('MONTH')
plt.ylabel('NUMBER OF CRIMES');


# The above graph should provide at least some insight into the level of magnitutde in terms of the number of crimes occurred grouped by category. It is clear that Larceny and Assault are the two categories ranking the top two for happening most frequently while crimes associated with children occurs the least.
# 
# It is also worthwhile noting that the occurrances of certain crime categories is fairly constant for example: weapon or child related crimes.
# 
# Crimes like Larceny and Assault fluctuates a lot and I suspect there exists some degrees of correlation between the two, which I will discover in the next section.

# ### Correlation
# 
# To be updated

# ### Number of Crimes Throughout a Year
# 
# Another meaningful visualization is to present the number of crimes happened throughout a year, and observe if there exists any patterns. Especially if on a certain given day, the number of crimes increases or decreases dramatically.

# In[ ]:


# extract the month and day values from the datatime index
month_day = crime_df.index.strftime('%m-%d')
crime_df['MONTH_DAY'] = month_day


# In[ ]:


# create a pivot table counting the number of offenses occured on each day of month
pivot = pd.pivot_table(data=crime_df, index='MONTH_DAY', aggfunc=['count'])
pivot.columns = pivot.columns.droplevel(level=0)


# In[ ]:


plt.figure(figsize=(20,8))
sns.lineplot(x=pivot.index, y='OFFENSE', data=pivot)
#plt.autoscale(enable=True, axis='x', tight=True)
plt.xticks(ticks=[16,46,77,107,138,168,199,229,260,290,321,351],
           labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
plt.xlabel('MONTH')
plt.title('Number of Crimes by Days', fontsize=14)

# label the important days in a year
important_dates = [0, 76, 84, 128, 200, 248, 315, 358]
day_names = ['New Year\'s Day', 'St.Patrick\'s Day', 'Good Friday', 'Mother\'s Day', 'Father\'s Day', 'Labor Day',
             'Veterans Day', 'Christmas Eve']
offense_num = [pivot.iloc[date].OFFENSE for date in important_dates]

for date in important_dates:
    plt.axvline(x=date, ymin=0, ymax=1, dashes=[5,2,1,2], color='grey', alpha=0.75)

plt.text(x=important_dates[0], y=155, s='{}\n{}'.format(day_names[0], offense_num[0]))
plt.text(x=important_dates[1], y=150, s='{}\n{}'.format(day_names[1], offense_num[1]))
plt.text(x=important_dates[2], y=65, s='{}\n{}'.format(day_names[2], offense_num[2]))
plt.text(x=important_dates[3], y=155, s='{}\n{}'.format(day_names[3], offense_num[3]))
plt.text(x=important_dates[4], y=70, s='{}\n{}'.format(day_names[4], offense_num[4]))
plt.text(x=important_dates[5], y=65, s='{}\n{}'.format(day_names[5], offense_num[5]))
plt.text(x=important_dates[6], y=65, s='{}\n{}'.format(day_names[6], offense_num[6]))
plt.text(x=important_dates[7]-10, y=155, s='{}\n{}'.format(day_names[7], offense_num[7]));


# In[ ]:




