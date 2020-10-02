#!/usr/bin/env python
# coding: utf-8

# **EDA for Chicago Crime data**
# 
# This is an data exploration performed in the Chicago crime dataset, extracted from the Chicago Police Department's CLEAR (Citizen Law Enforcement Analysis and Reporting) system.
# 
# **Note:**
# Please review my kernels and express your reviews or suggest your recommendation. I respect all the reviews and recommendations and it will be useful for my future works :)
# 
# Let's jump into the explorations

# In[ ]:


# Import all the libraries which will be used in the exploration process
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for data visualisations
import matplotlib.pyplot as plt # for data plotting and visualisations
from matplotlib.pyplot import figure, pie
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
import math

import bq_helper
from bq_helper import BigQueryHelper

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = 40, 10


# Loading the reference of the Chicago crime dataset from the **BigQueryAPI**

# In[ ]:


chicago_crime = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="chicago_crime")
bq_assistant = BigQueryHelper("bigquery-public-data", "chicago_crime")


# Select the necessary columns from the dataset of the Chicago crime dataset.
# 
# **Note**
# The limit is set to 300000 since loading the full dataset making my kernel suck.

# In[ ]:


select_query = """SELECT date,district,primary_type,location_description,ward,arrest,domestic,community_area,year,latitude,longitude,location
            FROM `bigquery-public-data.chicago_crime.crime`
            LIMIT 300000"""
crime_data = chicago_crime.query_to_pandas_safe(select_query)


# **Heatmap** is used to observe the crime counts based on the month and the years. The month is extracted from the date column using the **datetime**

# In[ ]:


month_year_frame = pd.DataFrame(columns=[])

for i in range(0,crime_data.shape[0]):
    month = crime_data.iloc[i].date.strftime("%b")
    year = str(crime_data.iloc[i].year)
        
    try:
        get_count = month_year_frame.at[month, year]
        if np.isnan(get_count):
            month_year_frame.at[month, year] = 1
        else:
            month_year_frame.at[month, year] = get_count+1
    except (ValueError,KeyError):
        month_year_frame.at[month, year] = 1

month_year_frame.index = pd.CategoricalIndex(month_year_frame.index, 
                               categories=['Jan', 'Feb', 'Mar', 'Apr','May','Jun', 'Jul', 'Aug','Sep', 'Oct', 'Nov', 'Dec'])
month_year_frame = month_year_frame.sort_index()
month_year_frame = month_year_frame.reindex(sorted(month_year_frame.columns), axis=1)

sns.heatmap(month_year_frame, cmap='gist_ncar')


# > The year 2001 has the highest crime count for the range 2001 to present. The crime count goes around 8000 to 12000
# 
# > The crime rate has gone down from the May of 2002 and it's been decreased in the upcoming years, which indicates that the Police department has took necessary actions to control the crime rate in Chicago

# Let's extract the arrest made by the Police Department for the crime in the year 2001 to present and compare with the crime heatmap and the heatmap for the arrest made.

# In[ ]:


corresponding_arrest = pd.DataFrame(columns=[])

for i in range(0,crime_data.shape[0]):
    month = crime_data.iloc[i].date.strftime("%b")
    year = str(crime_data.iloc[i].year)

    if crime_data.iloc[i].arrest:        
        try:
            get_count = corresponding_arrest.at[month, year]

            if np.isnan(get_count):
                corresponding_arrest.at[month, year] = 1
            else:
                corresponding_arrest.at[month, year] = get_count+1
        except (ValueError,KeyError):
            corresponding_arrest.at[month, year] = 1
            
corresponding_arrest.index = pd.CategoricalIndex(corresponding_arrest.index, 
                               categories=['Jan', 'Feb', 'Mar', 'Apr','May','Jun', 'Jul', 'Aug','Sep', 'Oct', 'Nov', 'Dec'])
corresponding_arrest = corresponding_arrest.sort_index()
corresponding_arrest = corresponding_arrest.reindex(sorted(corresponding_arrest.columns), axis=1)


# In[ ]:


fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

sns.heatmap(month_year_frame, cmap="gist_ncar",ax=ax1)
sns.heatmap(corresponding_arrest, cmap="gist_ncar",ax=ax2)


# > From the comparison, we can say that the department has maintained the crime rate by making the arrest for the crime, however there is a lack of actions for the crime during the year 2001 to 2012 which can be seen clearly from the heat trends above.

# The district plays an important role in crime occurence. The plot will be made for the total crime occurs and the arrest made against the district.

# In[ ]:


district_list = []

for i in range(0,crime_data.shape[0]):
    district = crime_data.iloc[i].district
    arrest = crime_data.iloc[i].arrest
    get_index = -1
    
    for j in range(0, len(district_list)):
        if (district_list[j][0] == district):
            get_index = j
            if arrest:
                district_list[j][1]+=1
            else:
                district_list[j][2]+=1
    
    if get_index == -1:
        if arrest:
            district_list.append([district, 1, 0])
        else:
            district_list.append([district, 0, 1])


get_district = pd.DataFrame(columns=['district','arrest','not_arrest'], data=district_list) 
get_district['Total'] = get_district.apply(lambda x: x.arrest+x.not_arrest, axis=1)

sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(10, 15))

# Load the example car crash dataset

# Plot the total crashes
sns.set_color_codes("pastel")
sns.barplot(x="Total", y="district", data=get_district,
            label="Total", color="b", orient='h')

sns.set_color_codes("muted")
sns.barplot(x="arrest", y="district", data=get_district,
            label="Arrest", color="b", orient='h')

ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(ylabel="",
       xlabel="Total cases vs arrest")
sns.despine(left=True, bottom=True)


# > The district 4 has the place where the highest crime (37834) taken place among all the district
# 
# > The plot represents that the arrest has not made for most of the crime in all the district

# The Police department has categorised the crime into several types and Let's see which type of crime has been taken place the more.

# In[ ]:


get_type = []

for i in range(0,crime_data.shape[0]):
    primary = crime_data.iloc[i].primary_type
    get_index = -1
    
    for j in range(0, len(get_type)):
        if (get_type[j][0] == primary):
            get_index = j
            get_type[j][1]+=1
    
    if get_index == -1:
        get_type.append([primary, 1])

type_data = pd.DataFrame(columns=['Type', 'count'], data=get_type)
fig1, ax1 = plt.subplots()
fig1.set_size_inches(18.5, 10.5)
ax1.pie(type_data['count'], labels=type_data['Type'], autopct='%1.1f%%',startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# > The Theft occured over 20.3% around Chicago whereas Battery crime has covered around 18.4% and the third is the 13.1% of criminal damages

# Let's compare the crime rates based on the time duration. The data is transformed into four categories based on the years: 2001-2005, 2006-2010, 2011-2015, 2016-2020 and we will compare the crime count based on the year gaps.

# In[ ]:


crime_data['month'] = crime_data.apply(lambda x: x.date.strftime("%b"), axis=1)


# In[ ]:


year_range = {
    1: [2001,2005],
    2: [2006,2010],
    3: [2011,2015],
    4: [2016,2020]
}

def get_year_ref(year):
    for index, dic in enumerate(year_range):
        gap = year_range[dic]
        if year >= gap[0] and year <= gap[1]:
            return dic
        
get_month_year = []

for i in range(0,crime_data.shape[0]):
    row = crime_data.iloc[i]
    get_index = -1
    get_year_index = get_year_ref(row.year)
    
    for j in range(0, len(get_month_year)):
        if get_month_year[j][0] == row.month:
            get_index = j
            get_month_year[j][get_year_index]+=1
    
    if get_index == -1:
        create_arr = [0] * 5
        create_arr[0] = row.month
        create_arr[get_year_index] = 1
        get_month_year.append(create_arr)

month_wise_crime = pd.DataFrame(columns=['month','2001-2005','2006-2010','2011-2015','2016-2020'], data=get_month_year) 
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
month_wise_crime['month'] = pd.Categorical(month_wise_crime['month'], categories=months, ordered=True)
month_wise_crime.sort_values('month', inplace=True)


# In[ ]:


rcParams['figure.figsize'] = 20, 10
sns.barplot(data=month_wise_crime)


# The graph shows that the crime rates has been reducing over the years. From the graph, the crime count of 2001-2005 is around 16000 and it gets decreased to around 1500 in the year 2016-2020. This shows that the police department has been taken several actions to reduce and control the crime rates

# Let's plot the data which are categoried separtely based on the year with respect to the month. The month is extracted from the date column and stored in a separte column in the crime_data

# In[ ]:


fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

sns.barplot(x='month',y='2001-2005', data=month_wise_crime, ax=ax1)
sns.barplot(x='month',y='2006-2010', data=month_wise_crime, ax=ax2)
sns.barplot(x='month',y='2011-2015', data=month_wise_crime, ax=ax3)
sns.barplot(x='month',y='2016-2020', data=month_wise_crime, ax=ax4)


# It is seen that March to August is the month in all the year category data where the crime rate are high when compared to the rest of the months. One possibility would be the temperature of the month when the crime taken place. The temperature of Chicago on the month March to August varies around 12c to 27c which says that the frequency of people in the public place are very high whereas the rest of the month, the temperature is below 10c, so the frequency of people in the public place are very low

# Let's plot the crime rate based on the district of Chicago. To make this visualization, the shape file of the district is used which is available in the **data.cityofchicago**

# In[ ]:


fig = plt.figure(figsize=(22, 12))
ax = fig.add_subplot(111)
cm = plt.get_cmap('Reds')

district_val = pd.DataFrame(crime_data.district.value_counts().reset_index().values, columns=["district", "count"])

m = Basemap(projection='lcc', resolution='l', 
            lat_0=41.867779, lon_0=-87.638403,
            width=0.06E6, height=0.06E6)
m.drawmapboundary()

m.readshapefile('../input/geo_export_4ea0a4fd-5ba9-4f3f-bb35-ccd16bfc2ff9', 
                    name='world', 
                    drawbounds=True, 
                    color='gray')

for info,shape in zip(m.world_info, m.world):
    color = '#dddddd'
    for i in range(0,len(district_val)):
        if str(math.ceil(district_val.iloc[i].district)) == info['dist_num']:
            color =  cm(district_val.iloc[i]['count'] / district_val['count'].sum())
            break


    patches = [Polygon(np.array(shape), True)]
    pc = PatchCollection(patches)
    pc.set_facecolor(color)
    ax.add_collection(pc)


# > Based on the heat trend in the map, it is said that the south of Chicago which is district 4 has the high crime rate and the second would be district 24 followed by district 8.
# 
# > The lower crime rate in Chicage are the district 31, followed by district 20 and district 1

# **Note**
# There will be future improvements in this kernel based on the reviews and the suggestions given for this kernel.

# In[ ]:




