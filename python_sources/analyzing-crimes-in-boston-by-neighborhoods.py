#!/usr/bin/env python
# coding: utf-8

# <a id = 'top'></a>

# # Analyze crimes in Boston (Version 1.0)

# **Ying Zhou**

# [1. Data Wrangling](#data_wrangling)

# [1.1 Dropping non-crimes](#drop3)

# [2. Preliminary Analysis](#pre_anal)

# [2.1 Crime and day of the week](#day)

# [2.2 Is the crime problem in Boston getting better or worse?](#year)

# [2.3 Crime and month](#month)

# [2.4 What are the most common crimes in Boston?](#common)

# [2.5 Where are crimes located in Boston?](#location)

# [2.6 What time of day are crimes commited?](#time)

# [2.7 How many crimes involve shooting?](#shooting)

# [3. Crime rates of neighborhoods of Boston](#crime_rates)

# [3.1 Calculate the population of each neighborhood of Boston](#3.0)

# [3.2 Determining the neighborhood in which a crime took place](#3.1)

# [3.3 Calculate the crime rates](#3.2)

# [4. Rates of particular crimes in Boston](#4)

# [4.1 Where are the rape reports?](#4.1)

# [4.2 Violent Crimes](#4.2)

# [4.3 Property Crimes](#4.3)

# [4.4 Results](#4.4)

# [5. Conclusion](#5)

# Let's first import the usual packages.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# Since we need to draw graphs we need to write our multiliner function here which can help us leave more room for tick labels if the tick labels are really long.

# In[ ]:


def multiliner(string_list, n):
    length = len(string_list)
    for i in range(length):
        rem = i % n
        string_list[i] = '\n' * rem + string_list[i]
    return string_list


# Time to get the data!

# In[ ]:


df = pd.read_csv('../input/boston-crime-incident-reports-20152019/crime.csv')


# In[ ]:


df.shape


# In[ ]:


df.head(20)


# <a id = 'data_wrangling'></a>
# [Return to top](#top)
# # 1. Data Wrangling

# Since `incident_number` is unique let's make it the index. Then we should drop `location` due to redundancy. Moreover we need to fill the NaNs in `shooting` because NaNs in this case indicate the absence of shooting in the crime.

# In[ ]:


#df.set_index('incident_number', inplace = True)
df['shooting'].fillna(0, inplace = True)
df.drop(columns = ['location'], inplace = True)


# In[ ]:


df.head(5)


# Now let's check the dtypes.

# In[ ]:


df.dtypes


# `reporting_area` should be int64s instead of a float64s. But before that let's first fix the remaining NaNs.

# In[ ]:


df.isna().sum()


# It sees that there are a lot of crimes missing numerous crucial data. They are unlikely to be filled anyway so let's first focus on analyzing data that does not require them. I think we should allow `reporting_area` to be float64 for now.

# <a id = 'drop3'></a>
# [Return to top](#top)
# ## 1.1 Dropping non-crimes

# Now let's analyze the crime classes.

# In[ ]:


df_od = df.groupby('offense_description').size().reset_index(name = 'counts').set_index('offense_description').sort_values(by = 'counts', ascending = False)


# In[ ]:


df_od


# Some large classes here aren't crimes at all. We have to fix that. Let's focus on the UCR Part numbers.

# According to [Wikipedia](https://en.wikipedia.org/wiki/Uniform_Crime_Reports), UCR part I and part II offenses are actually offenses. As for part III let's check what they actually are.

# In[ ]:


df_ucr = df.groupby('ucr_part').size().reset_index(name = 'counts').set_index('ucr_part')


# In[ ]:


df_ucr


# In[ ]:


df_p3 = df.loc[df['ucr_part'] == 'Part Three']


# In[ ]:


df_3d = df_p3.groupby('offense_description').size().reset_index(name = 'counts').set_index('offense_description')


# In[ ]:


df_3d


# Most of them are non-criminal or unrelated to what we usually think of as "crimes". I think we should ignore the entire part 3 or at least most part 3 incidents. Our report is about crime in Boston, not towed motor vehicles in Boston or accidents in Boston.

# Now let's focus on "Other". What does that mean?

# In[ ]:


df_other = df.loc[df['ucr_part'] == 'Other']


# In[ ]:


df_otherd = df_other.groupby('offense_description').size().reset_index(name = 'counts').set_index('offense_description')


# In[ ]:


df_otherd


# OK so these are almost all crimes or at the very least these are crime-related. This will be included then.

# What about NaN?

# In[ ]:


df_ucrna = df.loc[df['ucr_part'].isnull()]


# In[ ]:


df_nad = df_ucrna.groupby('offense_description').size().reset_index(name = 'counts').set_index('offense_description')


# In[ ]:


df_nad


# OK so most of them are crimes. I will drop the 5 investigate person incidents though. Now we will drop all Part 3 and investigate person incidents.

# In[ ]:


df_clean = df.loc[(df['ucr_part'] != 'Part Three') & (df['offense_description'] != 'INVESTIGATE PERSON')]


# In[ ]:


df_clean.shape


# <a id = 'pre_anal'></a>
# [Return to top](#top)
# # 2. Preliminary Analysis

# <a id = 'day'></a>
# [Return to top](#top)
# ## 2.1 Crime and day of the week

# In[ ]:


df_day = df_clean.groupby('day_of_week').size().reset_index(name = 'counts').set_index('day_of_week')


# In[ ]:


df_day


# It seems that Fridays in Boston are unusually filled with crimes while the opposite is true for Sundays and to a less extant Saturdays.

# Before plotting the graph we have to fix the ordering.

# In[ ]:


df_day.reset_index(inplace = True)


# In[ ]:


df_day['day_of_week'] = pd.Categorical(df_day['day_of_week'], categories = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'], ordered = True)


# In[ ]:


df_day.set_index('day_of_week', inplace = True)


# In[ ]:


df_day.sort_values(by = 'day_of_week', inplace = True)


# Time to do some plotting!

# In[ ]:


fig = plt.figure(figsize = (10,7))
ax = plt.subplot(111)
ind = np.arange(7)
crimes_by_day = df_day['counts']
rects = ax.bar(ind, crimes_by_day, width = 0.8, color = ['green','olive','olive','olive','olive','red','green'])
ax.set_xticks(ind)
ax.set_xticklabels(['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
ax.set_title('Crimes in Boston by days of the week')
ax.set_ylabel('Amount of crimes')
for rect in rects:
    height = rect.get_height()
    ax.text(rect.get_x() + 0.15, 1.01 * height, height, fontsize = 12)


# <a id = 'year'></a>
# [Return to top](#top)
# ## 2.2 Is the crime problem in Boston getting better or worse?

# In[ ]:


df_year = df_clean.groupby('year').size().reset_index(name = 'counts').set_index('year')


# In[ ]:


df_year


# Disregarding 2015 and 2019 since there are not enough data about them in the data set it is easy to see that the crime situation in 2017 is slightly better than the situation in 2016. The crime situation in 2018 is slightly better than the situation in 2017.

# In[ ]:


df_year.drop(labels = [2015,2019], inplace = True)


# In[ ]:


fig2 = plt.figure(figsize = (5,7))
ind2 = np.arange(3)
ax2 = plt.subplot(111)
rects = ax2.bar(ind2, df_year['counts'], width = 0.4, color = ['red','green','blue'])
ax2.set_xticks(ind2)
ax2.set_xticklabels([2016,2017,2018])
ax2.set_xlabel('Year')
ax2.set_ylabel('Amount of crimes')
ax2.set_title('Crimes in Boston by year')
for rect in rects:
    height = rect.get_height()
    ax2.text(rect.get_x() - 0.01, 1.01 * height, height, fontsize = 14)


# <a id = 'month'></a>
# [Return to top](#top)
# ## 2.3 Crime and month

# Are there more crimes in summers or in winters?

# Since the data set is from Aug.2015 to Jan. 2019 let's analyze the trends from 2016 to 2018.

# In[ ]:


df_1618 = df_clean.loc[(df_clean['year'] > 2015) & (df_clean['year'] < 2019)]


# In[ ]:


df_1618.shape


# In[ ]:


df_month = df_1618.groupby('month').size().reset_index(name = 'counts').set_index('month')


# In[ ]:


df_month


# In[ ]:


fig3 = plt.figure(figsize = (10,7))
ind3 = np.arange(12)
ax3 = plt.subplot(111)
rects = ax3.bar(ind3, df_month['counts'], width = 0.8,color = ['yellow','lime','yellow','yellow','red','red','red','darkred','orange','orange','green','green'])
ax3.set_xticks(ind3)
ax3.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax3.set_xlabel('Month')
ax3.set_ylabel('Amount of crimes')
ax3.set_title('Crimes in Boston by month')
for rect in rects:
    height = rect.get_height()
    ax3.text(rect.get_x() - 0.13, 1.01 * height, height, fontsize = 14)


# It seems that there is more crime during summer months and less crime during winter ones. It's possible that crime is fairly weather-dependent.

# <a id = 'common'></a>
# [Return to top](#top)
# ## 2.4 What are the most common crimes in Boston?

# In[ ]:


df_ocg = df_clean.groupby('offense_code_group').size().reset_index(name = 'counts').set_index('offense_code_group').sort_values(by = 'counts', ascending = False)


# In[ ]:


df_ocg


# In[ ]:


fig41 = plt.figure(figsize = (20,8))
ind41 = np.arange(10)
ax41 = plt.subplot(111)
y_data = df_ocg['counts'].head(10)
df_riocg = df_ocg.reset_index()
rects = ax41.bar(ind41, y_data, width = 0.8,color = 'r')
ax41.set_xticks(ind41)
ax41.set_xticklabels(multiliner(df_ocg.index.tolist()[:10], 2))
ax41.set_xlabel('Offense Code Group')
ax41.set_ylabel('Amount of crimes')
ax41.set_title('Crimes in Boston by offense code group')
for rect in rects:
    height = rect.get_height()
    ax41.text(rect.get_x() + 0.2, 1.02 * height, height, fontsize = 14)


# In terms of offense code groups the five most common crimes in Boston are larceny, other, drug violation, simple assault and vandalism.

# In[ ]:


df_od = df_clean.groupby('offense_description').size().reset_index(name = 'counts').set_index('offense_description').sort_values(by = 'counts', ascending = False)


# In[ ]:


df_od


# In[ ]:


fig42 = plt.figure(figsize = (20,8))
ind42 = np.arange(10)
ax42 = plt.subplot(111)
y_data = df_od['counts'].head(10)
df_riod = df_od.reset_index()
rects = ax42.bar(ind42, y_data, width = 0.8,color = 'r')
ax42.set_xticks(ind42)
ax42.set_xticklabels(multiliner(df_od.index.tolist()[:10], 3))
ax42.set_xlabel('Offense Description')
ax42.set_ylabel('Amount of crimes')
ax42.set_title('Crimes in Boston by offense description')
for rect in rects:
    height = rect.get_height()
    ax42.text(rect.get_x() + 0.2, 1.02 * height, height, fontsize = 14)


# In terms of offense descriptions the five most common crimes in Boston are vandalism, assault simple - battery, threats to do bodily harm, larceny theft from building and larceny threat from motor vehicle - non-accessory.

# <a id = 'location'></a>
# [Return to top](#top)
# ## 2.5 Where are crimes located in Boston?

# In[ ]:


df_districts = df_clean.groupby('district').size().reset_index(name = 'counts').set_index('district').sort_values('counts', ascending = False)


# In[ ]:


df_districts


# In[ ]:


fig5 = plt.figure(figsize = (10,7))
ind5 = np.arange(12)
ax5 = plt.subplot(111)
rects = ax5.bar(ind5, df_districts['counts'], width = 0.8,color = 'r')
ax5.set_xticks(ind5)
ax5.set_xticklabels(df_districts.index)
ax5.set_xlabel('District')
ax5.set_ylabel('Amount of crimes')
ax5.set_title('Crimes in Boston by district')
for rect in rects:
    height = rect.get_height()
    if height > 9999:
        hor = rect.get_x() - 0.13
    else:
        hor = rect.get_x() - 0.03
    ax5.text(hor, 1.01 * height, height, fontsize = 14)


# It seems that B2, D4, C11 and A1 have the most crimes. On the other hand A15, A7, E18 and E13 have the least amount of crimes.

# Analysis of whether a certain police district of Boston necessarily has more crimes compared to some other police district per capita is impossible because different police district differ a lot in terms of population size.

# However analyzing the same information using neighborhood boundaries should be possible.

# <a id = 'time'></a>
# [Return to top](#top)
# ## 2.6 What time of day are crimes commited?

# In[ ]:


df_hour = df_clean.groupby('hour').size().reset_index(name = 'counts').set_index('hour')


# In[ ]:


df_hour


# The most crimes are commited between 4PM and 8PM. Between 1AM and 8AM few crimes are commited. There are two short bumps of crime between 12PM and 1PM and between 12AM and 1AM. On the other hand the period between 1AM and 8AM is fairly crime-free.

# In[ ]:


fig6 = plt.figure(figsize = (20,7))
ind6 = np.arange(24)
ax6 = plt.subplot(111)
color = []
for i in range(24):
    amount = df_hour.loc[i, 'counts']
    if amount > 12000:
        color.append('darkred')
    elif amount < 4000:
        color.append('lime')
    elif amount > 10000:
        color.append('r')
    elif amount < 6000:
        color.append('g')
    else:
        color.append('olive')
rects = ax6.bar(ind6, df_hour['counts'], width = 0.8,color = color)
ax6.set_xticks(ind6)
ax6.set_xticklabels(df_hour.index)
ax6.set_xlabel('Hour')
ax6.set_ylabel('Amount of crimes')
ax6.set_title('Crimes in Boston by hour')
for rect in rects:
    height = rect.get_height()
    if height > 9999:
        hor = rect.get_x() - 0.13
    else:
        hor = rect.get_x() - 0.03
    ax6.text(hor, 1.01 * height, height, fontsize = 14)


# <a id = 'shooting'></a>
# [Return to top](#top)
# ## 2.7 How many crimes involve shooting?

# In[ ]:


df_shooting = df_clean.groupby('shooting').size().reset_index(name = 'counts').set_index('shooting')


# In[ ]:


df_shooting


# OK so Y represents shooting.

# In[ ]:


shooting_rate = df_shooting.loc['Y', 'counts']/df_clean.shape[0]


# In[ ]:


shooting_rate


# About 0.66% of all crimes involve shooting.

# In[ ]:


fig7 = plt.figure(figsize = (10,5))
labels = ['Shooting', 'No shooting']
ax7 = plt.subplot(111)
size = [shooting_rate, 1 - shooting_rate]
ax7.pie(size, explode = [0.5,0], labels = labels, autopct = '%1.2f%%', shadow = True, colors = ['red','blue'])
ax7.axis('equal')
ax7.legend()
ax7.set_title('What percentage of crimes involve shooting?')


# <a id = 'crime_rates'></a>
# [Return to top](#top)
# # 3. Crime rates of neighborhoods of Boston

# We are going to continue our exploration in 2.5 . It is impossible to calculate the population of police districts. However the population of neighborhoods of Boston are still possible to obtain. We are going to use the definitions of neighborhoods on the [Analyze Boston](http://bostonopendata-boston.opendata.arcgis.com/datasets/3525b0ee6e6b427f9aab5d0a1d0a1a28_0) website.

# <a id = '3.0'></a>
# ## 3.1 Calculate the population of each neighborhood of Boston

# In[ ]:


import json
with open('../input/boston-neighborhoods-geojson/Boston_Neighborhoods.geojson', 'r') as f:
    boston_geojson = json.load(f)
features = boston_geojson['features']
nbh_list = []
for feature in features:
    nbh_list.append(feature['properties']['Name'])
print(nbh_list)


# In[ ]:


boston_geojson['features'][0]['geometry']


# In[ ]:


nbh_list.sort()


# In[ ]:


nbh_list


# Now it's time to determine the population of each neighborhood. The data is also obtained from [Analyze Boston](https://data.boston.gov/dataset/boston-neighborhood-demographics) because different websites have completely different definitions of many neighborhoods. Here according to the City of Boston Chinatown and Leather district have been merged into Downtown and Bay Village is merged into South End.

# In[ ]:


nbh_list.remove('Chinatown')
nbh_list.remove('Bay Village')
nbh_list.remove('Leather District')


# In[ ]:


nbh_list


# In[ ]:


nbh_pop = {
    'Allston':22312,
 'Back Bay':16622,
 'Beacon Hill':9023,
 'Brighton':52685,
 'Charlestown':16439,
 'Dorchester':114249,
 'Downtown':15992,
 'East Boston':40508,
 'Fenway':33895,
 'Harbor Islands':535,
 'Hyde Park':32317,
 'Jamaica Plain':35541,
 'Longwood':4861,
 'Mattapan':22500,
 'Mission Hill':16874,
 'North End':8608,
 'Roslindale':26368,
 'Roxbury':49111,
 'South Boston':31110,
 'South Boston Waterfront':2564,
 'South End':29612,
 'West End':5423,
 'West Roxbury':30445
}


# In[ ]:


df_nb = pd.DataFrame.from_dict(data = nbh_pop, orient = 'index', columns = ['population'])


# In[ ]:


df_nb


# We need to preserve the data.

# In[ ]:


df_nb.to_csv('population.csv')


# <a id = '3.1'></a>
# [Return to top](#top)
# ## 3.2 Determining the neighborhood in which a crime took place

# Since we already have the information about the population of the neighborhoods. Now we need to determine the number of crimes that take place within a neighborhood. All the crimes that did not have latitude and longitude need to be dropped so we may have some slight underestimation and distortion.

# In order to do so we need to make sure that we can attribute a crime to a neighborhood we need to use the Geojson data from [Analyze Boston](http://bostonopendata-boston.opendata.arcgis.com/datasets/3525b0ee6e6b427f9aab5d0a1d0a1a28_0).

# In[ ]:


from shapely.geometry import Point, shape


# In[ ]:


def point_to_neighborhood (lat, long, geojson):
    point = Point(long, lat)
    features = geojson['features']
    for feature in features:
        polygon = shape(feature['geometry'])
        neighborhood = feature['properties']['Name']
        if polygon.contains(point):
            if neighborhood == 'Chinatown' or neighborhood == 'Leather District':
                return 'Downtown'
            elif neighborhood == 'Bay Village':
                return 'South End'
            else:
                return neighborhood
    print(f'Point ({long},{lat}) is not in Boston.')
    return None


# In[ ]:


boston_geojson


# In[ ]:


point_to_neighborhood(42.372269, -71.039015, boston_geojson)


# In[ ]:


df_nafree = df_clean.dropna(subset = ['lat','long'])


# In[ ]:


df_nafree.shape


# In[ ]:


df_nafree.shape[0]/df_clean.shape[0]


# So we can retain 94%+ of the data. That's good.

# In[ ]:


for index, row in df_nafree.iterrows():
    lat = df_nafree.at[index, 'lat']
    long = df_nafree.at[index, 'long']
    #print(index)
    #print(lat)
    #print(long)
    neighborhood = point_to_neighborhood(lat, long, boston_geojson)
    #print(neighborhood)
    df_nafree.at[index, 'Neighborhood'] = neighborhood


# In[ ]:


df_nafree.tail(10)


# Now it is time to first store the data and then calculate the crime rates.

# In[ ]:


df_nafree.to_csv('crimes_with_neighborhoods.csv')


# In[ ]:


df_nbh = df_nafree.groupby('Neighborhood').size().reset_index(name = 'count').set_index('Neighborhood')


# In[ ]:


df_nbh


# In[ ]:


df_nafree['Neighborhood'].isna().sum()


# The 497 new NANs don't matter much. However we should still understand why they are NANs. Most of them are literally (-1,-1)s that should have been considered NaN. The rest are actually in Boston but since they are near the boundaries of Boston the Geojson left them out of Boston. It is impossible to determine the exact neighborhood of these cases which is why we have to drop them.

# In[ ]:





# <a id = '3.2'></a>
# [Return to top](#top)
# ## 3.3 Calculate the crime rates

# In[ ]:


df_nbh.index.size


# All neighborhoods are present except for the 3 that got merged into other neighborhoods and Harbor Islands which is basically almost unpopulated.

# In[ ]:


df_nbh_full = pd.concat([df_nb,df_nbh],axis = 1, sort = True)


# In[ ]:


df_nbh_full


# In[ ]:


df_nbh_full.at['Harbor Islands','count'] = 0


# In[ ]:


for index, row in df_nbh_full.iterrows():
    df_nbh_full.at[index, 'Crime rate'] = df_nbh_full.at[index, 'count'] / df_nbh_full.at[index, 'population']


# In[ ]:


df_nbh_full = df_nbh_full.sort_values('Crime rate', ascending = False)


# In[ ]:


df_nbh_full


# Downtown appears unusually dangerous. However it seems that it should be an outlier because most people who pass through downtown Boston including most criminals don't actually live there. At the same time Harbor islands isn't really necessarily completely crime-free. Instead what's more likely is that it is so isolated that incidents there are unlikely to be known to the Boston Police.

# Now we need to normalize the crime rate using time. To do so we first need to understand the time range.

# In[ ]:


df.tail(5)


# In[ ]:


df.sort_values(by = 'occurred_on_date').head(5)


# In[ ]:


df.sort_values(by = 'occurred_on_date').tail(5)


# So we have 1322 days.

# In[ ]:


for index, row in df_nbh_full.iterrows():
    df_nbh_full.at[index, 'Crime rate'] = df_nbh_full.at[index, 'Crime rate'] * 365 / 1322


# Since people usually calculate crime rate per 100,000 residents we will do so.

# In[ ]:


for index, row in df_nbh_full.iterrows():
    df_nbh_full.at[index, 'Crime rate'] = df_nbh_full.at[index, 'Crime rate'] * 100000


# In[ ]:


df_nbh_full


# In[ ]:


df_nbh_full.index


# In[ ]:


multiliner(['a','b','c'],3)


# In[ ]:


fig0 = plt.figure(figsize = (20,10))
ax0 = plt.subplot(111)
ind0 = np.arange(23)
crime_rate_by_neighborhood = df_nbh_full['Crime rate']
rects = ax0.bar(ind0, crime_rate_by_neighborhood, width = 0.8, color = 'r')
ax0.set_xticks(ind0)
ax0.set_xticklabels(multiliner(df_nbh_full.index.tolist(),3))
ax0.set_title('Crime rate (per 100,000 residents) of neighborhoods of Boston')
ax0.set_ylabel('Crime rate')
for rect in rects:
    height = rect.get_height()
    if height >= 9999.5:
        hor = rect.get_x() - 0.02
    else:
        hor = rect.get_x() + 0.05
    ax0.text(hor, 1.01 * height + 250, int(round(height)), fontsize = 12)


# <a id = '4'></a>
# [Return to top](#top)
# # 4. Rates of particular crimes

# Now let's focus on the actual categories UCR use. We will focus on violent crimes (defined as murder and nonnegligent manslaughter, rape, robbery, and aggravated assault), property crimes (defined as burglary, larceny-theft, and motor vehicle theft) and arson.

# Before doing so we first need to drop some NaNs.

# In[ ]:


df_particulars = df_nafree.dropna(subset = ['Neighborhood'])


# In[ ]:


df_particulars.shape


# In[ ]:


df_particulars['offense_description'].unique().tolist()


# In[ ]:


df_particulars['offense_code_group'].unique().tolist()


# UCR Part II offenses should be irrelevant. However we still need to make sure. NANs in ucr_part are already irrelevant.

# In[ ]:


df_p2 = df_particulars.loc[df['ucr_part'] == 'Part Two']


# In[ ]:


df_2p = df_p2.groupby('offense_code_group').size().reset_index(name = 'count').set_index('offense_code_group')


# In[ ]:


df_2p


# It is clear that they are irrelevant. Hence we will drop them.

# In[ ]:


df_p1 = df_particulars.loc[df['ucr_part'] == 'Part One']


# In[ ]:


df_1p = df_p1.groupby('offense_code_group').size().reset_index(name = 'count').set_index('offense_code_group')


# In[ ]:


df_1p


# <a id = '4.1'></a>
# [Return to top](#top)
# ## 4.1 Where are the rape reports?

# OK so we should tackle arson separately first. However we first need to find rapes.

# In[ ]:


df_1d = df_p1.groupby('offense_description').size().reset_index(name = 'count').set_index('offense_description')


# In[ ]:


df_1d


# Again there is no rape. Let's check again.

# In[ ]:


l_op = df.groupby('offense_code_group').size().reset_index(name = 'counts')['offense_code_group'].tolist()


# In[ ]:


l_op


# In[ ]:


l_od = df.groupby('offense_description').size().reset_index(name = 'counts')['offense_description'].tolist()


# In[ ]:


l_od


# OK. So there is indeed no rapes reported at all. This implies nothing other than the data being flawed.

# Now we can finally begin to calculate the rates of crimes other than rape.

# <a id = '4.2'></a>
# [Return to top](#top)
# ## 4.2 Violent crimes

# Let's first calculate the rates of violent crimes. Let's start from murder.

# In[ ]:


df_murder = df_particulars.loc[df_particulars['offense_description'] == 'MURDER, NON-NEGLIGIENT MANSLAUGHTER']


# In[ ]:


df_murdern = df_murder.groupby('Neighborhood').size().reset_index(name = 'murder_and_nonnegligent_manslaughter').set_index('Neighborhood')


# In[ ]:


df_murdern


# Now let's move on to aggravated assults.

# In[ ]:


df_aa = df_particulars.loc[df_particulars['offense_code_group'] == 'Aggravated Assault']


# In[ ]:


df_aa_d = df_aa.groupby('offense_description').size().reset_index(name = 'count').set_index('offense_description')


# In[ ]:


df_aa_d


# Now I think we should check the neighborhoods.

# In[ ]:


df_aan = df_aa.groupby('Neighborhood').size().reset_index(name = 'aggravated_assault').set_index('Neighborhood')


# In[ ]:


df_aan


# Now it's time to look at robbery.

# In[ ]:


df_robbery = df_particulars.loc[df_particulars['offense_code_group'] == 'Robbery']


# In[ ]:


df_r_d = df_robbery.groupby('offense_description').size().reset_index(name = 'count').set_index('offense_description')


# In[ ]:


df_r_d


# In[ ]:


df_robberyn = df_robbery.groupby('Neighborhood').size().reset_index(name = 'robbery').set_index('Neighborhood')


# In[ ]:


df_robberyn


# <a id = '4.3'></a>
# [Return to top](#top)
# ## 4.3 Property Crime

# Let's start from arson.

# In[ ]:


df_arson = df_particulars.loc[df_particulars['offense_code_group'] == 'Arson']


# In[ ]:


df_arson_d = df_arson.groupby('offense_description').size().reset_index(name = 'count').set_index('offense_description')


# In[ ]:


df_arson_d


# In[ ]:


df_arsonn = df_arson.groupby('Neighborhood').size().reset_index(name = 'arson').set_index('Neighborhood')


# In[ ]:


df_arsonn


# Now it's time to look up auto theft. Here auto theft / motor vehicle theft only includes theft *of* motor vehicles, not theft of property *from* them.

# In[ ]:


df_at = df_particulars.loc[df_particulars['offense_code_group'] == 'Auto Theft']


# In[ ]:


df_at_d = df_at.groupby('offense_description').size().reset_index(name = 'count').set_index('offense_description')


# In[ ]:


df_at_d


# In[ ]:


df_atn = df_at.groupby('Neighborhood').size().reset_index(name = 'auto_theft').set_index('Neighborhood')


# In[ ]:


df_atn


# Time to discuss larceny.

# In[ ]:


df_larceny = df_particulars.loc[(df_particulars['offense_code_group'] == 'Larceny') | (df_particulars['offense_code_group'] == 'Larceny From Motor Vehicle')]


# In[ ]:


df_larceny_d = df_larceny.groupby('offense_description').size().reset_index(name = 'count').set_index('offense_description')


# In[ ]:


df_larceny_d


# In[ ]:


df_larcenyn = df_larceny.groupby('Neighborhood').size().reset_index(name = 'larceny').set_index('Neighborhood')


# In[ ]:


df_larcenyn


# Finally let's discuss burglary.

# In[ ]:


df_burglary = df_particulars.loc[(df_particulars['offense_code_group'] == 'Other Burglary') | (df_particulars['offense_code_group'] == 'Commercial Burglary') | (df_particulars['offense_code_group'] == 'Residential Burglary') | (df_particulars['offense_code_group'] == 'Burglary - No Property Taken')]


# In[ ]:


df_burglary_d = df_burglary.groupby('offense_description').size().reset_index(name = 'count').set_index('offense_description')


# In[ ]:


df_burglary_d


# In[ ]:


df_burglaryn = df_burglary.groupby('Neighborhood').size().reset_index(name = 'burglary').set_index('Neighborhood')


# In[ ]:


df_burglaryn


# <a id = '4.4'></a>
# [Return to top](#top)
# ## 4.4 Results

# In[ ]:


df_nbh_crimes = pd.concat([df_nbh_full, df_murdern, df_aan, df_robberyn, df_arsonn, df_atn, df_larcenyn, df_burglaryn], axis = 1, sort = True)


# In[ ]:


df_nbh_crimes


# In[ ]:


df_nbh_crimes.fillna(0, inplace = True)


# In[ ]:


df_nbh_crimes


# In[ ]:


for index, row in df_nbh_crimes.iterrows():
    df_nbh_crimes.at[index, 'violent_crimes'] = df_nbh_crimes.at[index, 'murder_and_nonnegligent_manslaughter'] + df_nbh_crimes.at[index, 'aggravated_assault'] + df_nbh_crimes.at[index, 'robbery']
    df_nbh_crimes.at[index, 'property_crimes'] = df_nbh_crimes.at[index, 'arson'] + df_nbh_crimes.at[index, 'auto_theft'] + df_nbh_crimes.at[index, 'larceny'] + df_nbh_crimes.at[index, 'burglary']
    df_nbh_crimes.at[index, 'murder_rate'] = df_nbh_crimes.at[index, 'murder_and_nonnegligent_manslaughter'] * 365 / 1322 / df_nbh_crimes.at[index, 'population'] * 100000
    df_nbh_crimes.at[index, 'aggravated_assault_rate'] = df_nbh_crimes.at[index, 'aggravated_assault'] * 365 / 1322 / df_nbh_crimes.at[index, 'population'] * 100000
    df_nbh_crimes.at[index, 'robbery_rate'] = df_nbh_crimes.at[index, 'robbery'] * 365 / 1322 / df_nbh_crimes.at[index, 'population'] * 100000
    df_nbh_crimes.at[index, 'arson_rate'] = df_nbh_crimes.at[index, 'arson'] * 365 / 1322 / df_nbh_crimes.at[index, 'population'] * 100000
    df_nbh_crimes.at[index, 'auto_theft_rate'] = df_nbh_crimes.at[index, 'auto_theft'] * 365 / 1322 / df_nbh_crimes.at[index, 'population'] * 100000
    df_nbh_crimes.at[index, 'larceny_rate'] = df_nbh_crimes.at[index, 'larceny'] * 365 / 1322 / df_nbh_crimes.at[index, 'population'] * 100000
    df_nbh_crimes.at[index, 'burglary_rate'] = df_nbh_crimes.at[index, 'burglary'] * 365 / 1322 / df_nbh_crimes.at[index, 'population'] * 100000
    df_nbh_crimes.at[index, 'violent_crime_rate'] = df_nbh_crimes.at[index, 'violent_crimes'] * 365 / 1322 / df_nbh_crimes.at[index, 'population'] * 100000
    df_nbh_crimes.at[index, 'property_crime_rate'] = df_nbh_crimes.at[index, 'property_crimes'] * 365 / 1322 / df_nbh_crimes.at[index, 'population'] * 100000


# In[ ]:


df_nbh_crimes


# In[ ]:


df_nbh_crimes.to_csv('crime_rates_by_neighborhood.csv')


# In[ ]:


fig01 = plt.figure(figsize = (20,10))
ax01 = plt.subplot(111)
ind01 = np.arange(23)
murder_rate_by_neighborhood = df_nbh_crimes['murder_rate']
rects = ax01.bar(ind01, murder_rate_by_neighborhood, width = 0.8, color = 'r')
ax01.set_xticks(ind01)
ax01.set_xticklabels(multiliner(df_nbh_crimes.index.tolist(),2))
ax01.set_title('Murder rate (per 100,000 residents) of neighborhoods of Boston')
ax01.set_ylabel('Murder rate')
for rect in rects:
    height = rect.get_height()
    if height >= 9.95:
        hor = rect.get_x() + 0.08
    else:
        hor = rect.get_x() + 0.12
    ax01.text(hor, 1.01 * height + 0.1, "{:.1f}".format(height), fontsize = 12)


# Allston, Back Bay, Beacon Hill, Harbor Islands, Longwood, North End and South Boston Waterfront are completely murder-free according to our data.

# On the other hand Mattapan, Dorchester and Roxbury have high murder rates.

# Now let's plot the graph of aggravated assault.

# In[ ]:


fig02 = plt.figure(figsize = (20,10))
ax02 = plt.subplot(111)
ind02 = np.arange(23)
aa_rate_by_neighborhood = df_nbh_crimes['aggravated_assault_rate']
rects = ax02.bar(ind02, aa_rate_by_neighborhood, width = 0.8, color = 'r')
ax02.set_xticks(ind02)
ax02.set_xticklabels(multiliner(df_nbh_crimes.index.tolist(),2))
ax02.set_title('Aggravated assault rate (per 100,000 residents) of neighborhoods of Boston')
ax02.set_ylabel('Aggravated assault rate')
for rect in rects:
    height = rect.get_height()
    if height >= 999.95:
        hor = rect.get_x() - 0.04
    elif height >= 99.95:
        hor = rect.get_x()
    elif height >= 9.95:
        hor = rect.get_x() + 0.08
    else:
        hor = rect.get_x() + 0.12
    ax02.text(hor, 1.01 * height + 10, "{:.1f}".format(height), fontsize = 12)


# Harbor Islands, Beacon Hill, Brighton and West Roxbury have low aggravated assault rates. On the other hand Downtown, Roxbury, Mattapan, Dorchester and West End have high aggravated assault rates.

# Now let's plot the graph of robbery.

# In[ ]:


fig03 = plt.figure(figsize = (20,10))
ax03 = plt.subplot(111)
ind03 = np.arange(23)
r_rate_by_neighborhood = df_nbh_crimes['robbery_rate']
rects = ax03.bar(ind03, r_rate_by_neighborhood, width = 0.8, color = 'r')
ax03.set_xticks(ind03)
ax03.set_xticklabels(multiliner(df_nbh_crimes.index.tolist(),2))
ax03.set_title('Robbery rate (per 100,000 residents) of neighborhoods of Boston')
ax03.set_ylabel('Robbery rate')
for rect in rects:
    height = rect.get_height()
    if height >= 999.95:
        hor = rect.get_x() - 0.04
    elif height >= 99.95:
        hor = rect.get_x()
    elif height >= 9.95:
        hor = rect.get_x() + 0.08
    else:
        hor = rect.get_x() + 0.12
    ax03.text(hor, 1.01 * height + 7, "{:.1f}".format(height), fontsize = 12)


# Harbor Islands, South Boston Waterfront, Brighton and West Roxbury have low robbery rates. On the other hand Downtown, Roxbury, Mattapan, Dorchester and Back Bay have high robbery rates.

# Now let's plot the graph of arson.

# In[ ]:


fig04 = plt.figure(figsize = (20,10))
ax04 = plt.subplot(111)
ind04 = np.arange(23)
ar_rate_by_neighborhood = df_nbh_crimes['arson_rate']
rects = ax04.bar(ind04, ar_rate_by_neighborhood, width = 0.8, color = 'r')
ax04.set_xticks(ind04)
ax04.set_xticklabels(multiliner(df_nbh_crimes.index.tolist(),2))
ax04.set_title('Arson rate (per 100,000 residents) of neighborhoods of Boston')
ax04.set_ylabel('Arson rate')
for rect in rects:
    height = rect.get_height()
    if height >= 999.95:
        hor = rect.get_x() - 0.04
    elif height >= 99.95:
        hor = rect.get_x()
    elif height >= 9.95:
        hor = rect.get_x() + 0.08
    else:
        hor = rect.get_x() + 0.16
    ax04.text(hor, 1.01 * height + 0.02, "{:.1f}".format(height), fontsize = 12)


# Harbor Islands, South Boston Waterfront and Longwood have no arson at all. On the other hand Roslindale, Roxbury, and Downtown have high arson rates.

# Now let's plot the graph of auto theft.

# In[ ]:


fig05 = plt.figure(figsize = (20,10))
ax05 = plt.subplot(111)
ind05 = np.arange(23)
at_rate_by_neighborhood = df_nbh_crimes['auto_theft_rate']
rects = ax05.bar(ind05, at_rate_by_neighborhood, width = 0.8, color = 'r')
ax05.set_xticks(ind05)
ax05.set_xticklabels(multiliner(df_nbh_crimes.index.tolist(),2))
ax05.set_title('Auto theft rate (per 100,000 residents) of neighborhoods of Boston')
ax05.set_ylabel('Auto theft rate')
for rect in rects:
    height = rect.get_height()
    if height >= 999.95:
        hor = rect.get_x() - 0.04
    elif height >= 99.95:
        hor = rect.get_x()
    elif height >= 9.95:
        hor = rect.get_x() + 0.08
    else:
        hor = rect.get_x() + 0.16
    ax05.text(hor, 1.01 * height + 3, "{:.1f}".format(height), fontsize = 12)


# Harbor Islands, West Roxbury and Brighton have low auto theft rates. On the other hand Roslindale, Roxbury, Downtown, Dorchester, South Boston Waterfront and Back Bay have high auto theft rates.

# Now let's plot the graph of larceny.

# In[ ]:


fig06 = plt.figure(figsize = (20,10))
ax06 = plt.subplot(111)
ind06 = np.arange(23)
la_rate_by_neighborhood = df_nbh_crimes['larceny_rate']
rects = ax06.bar(ind06, la_rate_by_neighborhood, width = 0.8, color = 'r')
ax06.set_xticks(ind06)
ax06.set_xticklabels(multiliner(df_nbh_crimes.index.tolist(),2))
ax06.set_title('Larceny rate (per 100,000 residents) of neighborhoods of Boston')
ax06.set_ylabel('Larceny rate')
for rect in rects:
    height = rect.get_height()
    if height >= 999.95:
        hor = rect.get_x() - 0.06
    elif height >= 99.95:
        hor = rect.get_x() + 0.02
    elif height >= 9.95:
        hor = rect.get_x() + 0.08
    else:
        hor = rect.get_x() + 0.16
    ax06.text(hor, 1.01 * height + 30, "{:.1f}".format(height), fontsize = 12)


# Harbor Islands, West Roxbury, Brighton and Roslindale have low larceny rates. On the other hand Downtown and Back Bay have high larceny rates.

# Finally let's plot the graph of burglary.

# In[ ]:


fig07 = plt.figure(figsize = (20,10))
ax07 = plt.subplot(111)
ind07 = np.arange(23)
b_rate_by_neighborhood = df_nbh_crimes['burglary_rate']
rects = ax07.bar(ind07, b_rate_by_neighborhood, width = 0.8, color = 'r')
ax07.set_xticks(ind07)
ax07.set_xticklabels(multiliner(df_nbh_crimes.index.tolist(),2))
ax07.set_title('Burglary rate (per 100,000 residents) of neighborhoods of Boston')
ax07.set_ylabel('Burglary rate')
for rect in rects:
    height = rect.get_height()
    if height >= 999.95:
        hor = rect.get_x() - 0.06
    elif height >= 99.95:
        hor = rect.get_x() + 0.02
    elif height >= 9.95:
        hor = rect.get_x() + 0.08
    else:
        hor = rect.get_x() + 0.16
    ax07.text(hor, 1.01 * height + 3, "{:.1f}".format(height), fontsize = 12)


# Harbor Islands, Longwood and West Roxbury have low burglary rates. On the other hand Allston, Back Bay and Downtown have high burglary rates.

# <a id = '5'></a>
# [Return to top](#top)
# # 5. Conclusion

# * Fridays in Boston are unusually filled with crimes while the opposite is true for Sundays and to a less extant Saturdays.
# * From 2016 to 2018 crime slightly declined.
# * There is more crime during summer months and less crime during winter ones. It's possible that crime is fairly weather-dependent.
# * In terms of offense code groups the five most common crimes in Boston are larceny, other, drug violation, simple assault and vandalism.
# * In terms of offense descriptions the five most common crimes in Boston are vandalism, assault simple - battery, threats to do bodily harm, larceny theft from building and larceny threat from motor vehicle - non-accessory.
# * Police districts B2, D4, C11 and A1 have the most crimes. On the other hand A15, A7, E18 and E13 have the least amount of crimes.
# * The most crimes are commited between 4PM and 8PM. Between 1AM and 8AM few crimes are commited. There are two short bumps of crime between 12PM and 1PM and between 12AM and 1AM. On the other hand the period between 1AM and 8AM is fairly crime-free.
# * About 0.66% of all crimes involve shooting.
# * Allston, Back Bay, Beacon Hill, Harbor Islands, Longwood, North End and South Boston Waterfront are completely murder-free. On the other hand Mattapan, Dorchester and Roxbury have high murder rates.
# * Harbor Islands, Beacon Hill, Brighton and West Roxbury have low aggravated assault rates. On the other hand Downtown, Roxbury, Mattapan, Dorchester and West End have high aggravated assault rates.
# * Harbor Islands, South Boston Waterfront, Brighton and West Roxbury have low robbery rates. On the other hand Downtown, Roxbury, Mattapan, Dorchester and Back Bay have high robbery rates.
# * Harbor Islands, South Boston Waterfront and Longwood have no arson at all. On the other hand Roslindale, Roxbury, and Downtown have high arson rates.
# * Harbor Islands, West Roxbury and Brighton have low auto theft rates. On the other hand Roslindale, Roxbury, Downtown, Dorchester, South Boston Waterfront and Back Bay have high auto theft rates.
# * Harbor Islands, West Roxbury, Brighton and Roslindale have low larceny rates. On the other hand Downtown and Back Bay have high larceny rates.
# * Harbor Islands, Longwood and West Roxbury have low burglary rates. On the other hand Allston, Back Bay and Downtown have high burglary rates.
