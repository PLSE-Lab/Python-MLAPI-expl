#!/usr/bin/env python
# coding: utf-8

# # 1. The Lists of Data Table
# ### 1) Case Data
# - **Case**: Data of COVID-19 infection cases in South Korea
# 
# ### 2) Patient Data
# - **PatientInfo**: Epidemiological data of COVID-19 patients in South Korea
# - **PatientRoute**: Route data of COVID-19 patients in South Korea
# 
# ### 3) Time Series Data
# - **Time**: Time series data of COVID-19 status in South Korea
# - **TimeAge**: Time series data of COVID-19 status in terms of the age in South Korea
# - **TimeGender**: Time series data of COVID-19 status in terms of gender in South Korea
# - **TimeProvince**: Time series data of COVID-19 status in terms of the Province in South Korea
# 
# ### 4) Additional Data
# - **Region**: Location and statistical data of the regions in South Korea
# - **Weather**: Data of the weather in the regions of South Korea
# - **SearchTrend**: Trend data of the keywords searched in NAVER which is one of the largest portals in South Korea
# - **SeoulFloating**: Data of floating population in Seoul, South Korea (from SK Telecom Big Data Hub)

# # 2. The Structure of our Dataset
# - What color means is that they have similar properties.
# - If a line is connected between columns, it means that the values of the columns are partially shared.
# - The dotted lines mean weak relevance.
# ![db](https://user-images.githubusercontent.com/50820635/78222744-b0824a80-7500-11ea-84d8-49775e562108.PNG)

# # 3. The Detailed Description of each Data Table

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


path = '/kaggle/input/coronavirusdataset/'

case = p_info = pd.read_csv(path+'Case.csv')
p_info = pd.read_csv(path+'PatientInfo.csv')
p_route = pd.read_csv(path+'PatientRoute.csv')
time = pd.read_csv(path+'Time.csv')
t_age = pd.read_csv(path+'TimeAge.csv')
t_gender = pd.read_csv(path+'TimeGender.csv')
t_provin = pd.read_csv(path+'TimeProvince.csv')
region = pd.read_csv(path+'Region.csv')
weather = pd.read_csv(path+'Weather.csv')
search = pd.read_csv(path+'SearchTrend.csv')
floating = pd.read_csv(path+'SeoulFloating.csv')


# ##### Before the Start..
# - We make a structured dataset based on the report materials of KCDC and local governments.
# - In Korea, we use the terms named '-do', '-si', '-gun' and '-gu',
# - The meaning of them are explained below.
# 
# ***
# 
# 
# ### Levels of administrative divisions in South Korea
# #### Upper Level (Provincial-level divisions)
# - **Special City**:
# *Seoul*
# - **Metropolitan City**:
# *Busan / Daegu / Daejeon / Gwangju / Incheon / Ulsan*
# - **Province(-do)**:
# *Gyeonggi-do / Gangwon-do / Chungcheongbuk-do / Chungcheongnam-do / Jeollabuk-do / Jeollanam-do / Gyeongsangbuk-do / Gyeongsangnam-do*
# 
# #### Lower Level (Municipal-level divisions)
# - **City(-si)**
# [List of cities in South Korea](https://en.wikipedia.org/wiki/List_of_cities_in_South_Korea)
# - **Country(-gun)**
# [List of counties of South Korea](https://en.wikipedia.org/wiki/List_of_counties_of_South_Korea)
# - **District(-gu)**
# [List of districts in South Korea](https://en.wikipedia.org/wiki/List_of_districts_in_South_Korea)
# 
# ***
# 
# <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2815958%2F1c50702025f44b0c1ce92460bd2ea3f9%2Fus_hi_30-1.jpg?generation=1582819435038273&amp;alt=media" width=700>
# 
# ***
# 
# Sources
# - http://nationalatlas.ngii.go.kr/pages/page_1266.php
# - https://en.wikipedia.org/wiki/Administrative_divisions_of_South_Korea

# ### 1) Case
# #### Data of COVID-19 infection cases in South Korea
# 1. case_id: the ID of the infection case
#   > - case_id(7) = region_code(5) + case_number(2)  
#   > - You can check the region_code in 'Region.csv'
# - province: Special City / Metropolitan City / Province(-do)
# - city: City(-si) / Country (-gun) / District (-gu)
#   > - The value 'from other city' means that where the group infection started is other city.
# - group: TRUE: group infection / FALSE: not group
#   > - If the value is 'TRUE' in this column, the value of 'infection_cases' means the name of group.  
#   > - The values named 'contact with patient', 'overseas inflow' and 'etc' are not group infection. 
# - infection_case: the infection case (the name of group or other cases)
#   > - The value 'overseas inflow' means that the infection is from other country.  
#   > - Tha value 'etc' includes individual cases, cases where relevance classification is ongoing after investigation, and cases under investigation.
# - confirmed: the accumulated number of the confirmed
# - latitude: the latitude of the group (WGS84)
# - longitude: the longitude of the group (WGS84)
# 

# In[ ]:


case.head()


# In[ ]:


print('There are {} unique values which are: \n'.format(len(case['province'].unique())))
print(case['province'].unique())
print('\n Out of which 9 are are provinces and 8 are Special cities/Metropolitan cities')


# In[ ]:


local_infection = case.pivot_table(index=['group'], aggfunc='size')
local_infection


# So there are 62 instance of local spread while 50 cases had roots from other cities.

# In[ ]:


group_cases = case.pivot_table(index = ['group'], aggfunc = 'sum').reset_index()
group_cases = group_cases.replace({True: 'Locally transmitted', False : 'Other cities'})
group_cases


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import branca.colormap
from collections import defaultdict
print('Matplotlib, Seaborn and Folium are imported to visualize')


# In[ ]:


sns.barplot(x = 'group', y = 'confirmed', data = group_cases, ci = None).set(ylabel='confirmed cases', title='Number of cases wrt transmission')


# In[ ]:


print('There are {} unique values of the groups.'.format(len(case['infection_case'].unique())))


# # Local Spread

# ###### First understanding the local spread in the country.
# On the basis of **group** feature we can see the the local spread closely
# 
# **group** : TRUE: group infection / FALSE: not group
# If the value is 'TRUE' in this column, the value of 'infection_cases' means the name of group.
# 
# The values named 'contact with patient', 'overseas inflow' and 'etc' are not group infection.

# In[ ]:


#extracting data with groups where only local spread took place
local_case_grp = case.query('group == True')
local_case_grp.head()


# In[ ]:


print('So there are {} different groups where local spread took place'.format(len(local_case_grp['infection_case'].unique())))


# Now that we have the cases started due to the group, we will look at the cases in the same city.
# Removing the cases from groups of other cities as, we don't have their coordinates.

# In[ ]:


local_case_grp = local_case_grp.loc[local_case_grp['city'] != 'from other city']
print('Now the shape of the data is {}'.format(local_case_grp.shape))
local_case_grp.head()


# In[ ]:


latitude,longitude = 35.9078, 127.7669      #Coordinates of South Korea
#Plotting map of South korea
S_korea = folium.Map(location = [latitude,longitude],zoom_start = 7)


S_korea.add_child(plugins.HeatMap(zip(local_case_grp['latitude'],local_case_grp['longitude'], local_case_grp['confirmed']), radius = 10))


# In[ ]:


print('There are {} different province where local spread through {} groups took place. \n         '.format(len(local_case_grp['province'].unique()),len(local_case_grp['infection_case'].unique())))


# As there are 39 different infection case in the data of 40 rows, each name of the group in the is different.

# In[ ]:


infection_case_local = local_case_grp.pivot_table(index = 'infection_case',aggfunc = 'sum')['confirmed'].reset_index()
infection_case_local['confirmed'] = np.log(infection_case_local['confirmed'])
fig_dims = (10,10)
fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x = 'confirmed',y = 'infection_case', data = infection_case_local, ax = ax)     .set(ylabel = 'Group of infection', xlabel = 'Confirmed cases', title = 'Cases due to group')


# In[ ]:


local_case_grp.loc[local_case_grp['infection_case']=='Shincheonji Church']


# 1. It can be concluded that Shincheonji Church is the hotspot of the local spread.
# 2. Shincheonji Church is present in the city of Nam-gu in the Daegu province.
# 3. Which will lead to the increase in the number of cases in Daegu.

# In[ ]:


confirmed = local_case_grp.pivot_table(index=['province'], aggfunc='sum')
local_province_case_grp = local_case_grp.pivot_table(index=['province'], aggfunc=pd.Series.nunique)
local_province_case_grp = local_province_case_grp.merge(confirmed, on='province',how = 'inner')
local_province_case_grp = local_province_case_grp.drop(columns = ['case_id_x','confirmed_x','group_x','latitude','longitude','case_id_y','group_y'])                                 .rename(columns = {'confirmed_y':'confirmed_cases'}).reset_index()
local_province_case_grp


# In[ ]:


g = sns.PairGrid(local_province_case_grp, x_vars=["city","confirmed_cases"], y_vars=["province"], height=6)
g.map(sns.barplot)
g.fig.suptitle('Cities infected and confirmed cases wrt province', y = 1.05)


# 1. It is clearly visible that the province Daegu has the most number of locally spread cases of 4968 cases in its 4 cities.
# 2. City Seoul has the most number of affected cities of 6 but has comparitively very low number of cases. Hence it is the best performance city in controlling the local spread.

# In[ ]:


sns.barplot(x= 'confirmed',y = 'city',data= local_case_grp.loc[local_case_grp['province'] == 'Daegu'],ci = None)         .set(xlabel='Confirmed cases', ylabel='Cities in Daegu', title='Cases in Daegu')


# In[ ]:


sns.barplot(x= 'confirmed',y = 'city',data= local_case_grp.loc[local_case_grp['province'] == 'Seoul'],ci = None)         .set(xlabel='Confirmed cases', ylabel='Cities in Seoul', title='Cases in Seoul')


# In[ ]:


sns.barplot(x= 'confirmed',y = 'city',data= local_case_grp.loc[local_case_grp['province'] == 'Gyeongsangbuk-do'],ci = None)         .set(xlabel='Confirmed cases', ylabel='Cities in Gyeongsangbuk-do', title='Cases in Gyeongsangbuk-do')


# 1. It is clear that Daegu is the most affected Province with Nam-gu being the worst hit city.
# 2. Seoul has well distributed number of cases and comparitively less cases with most cities affected.
# 3. Gyeongsangbuk-do also has significant amount of case with Cheondo-gun being the worst hit city.

# In[ ]:


local_case_city_grp = local_case_grp.pivot_table(index='city',aggfunc = 'sum').reset_index()
plt.figure(figsize=(15,10))
local_case_city_grp['confirmed'] = np.log(local_case_city_grp['confirmed'])
sns.barplot(x= 'confirmed',y = 'city', color = 'purple',data= local_case_city_grp).set(xlabel='confirmed cases', title='Confirmed cases in different cities due to Local Transmission')


# ### The conclusion of the local spread
# 1. The  highest local spread too place in Daegu with Nam-gu with most cases.
# 2. Shincheonji Church being the hotspot in the Nam-gu city in Daegu province with most cases of local spread.
# 3. Seoul have distributed number of cases but still controlled.

# # Spread from other cities

# In[ ]:


other_cities_grp = case.query('group == False')
other_cities_grp.head()
#print(other_cities_grp.shape)


# Here we have all the cases which did not spread through local transmission but from other countries or through patient.
# That's why we don't have the coordinates of the city.
# 
# Let's drop those columns.

# In[ ]:


other_cities_grp = other_cities_grp.drop(columns=['city','group','latitude','longitude'],axis=1)
other_cities_grp.head()


# In[ ]:


other_cities_province_grp = other_cities_grp.pivot_table(index = ['province'],aggfunc = 'sum')
#other_cities_province_grp['confirmed'] = np.log(other_cities_province_grp.['confirmed'])
plt.figure(figsize=(15,10))
sns.barplot(y = 'province',x = 'confirmed', data = other_cities_grp,ci = None ).set(xlabel='confirmed cases', title='Cases due to non-local transmission')


# In[ ]:


other_cities_grp['infection_case'].value_counts().plot(kind = 'barh',rot = 0,figsize = (10,5))
plt.xlabel("Count of Spread", labelpad=14)
plt.ylabel("Causes", labelpad=14)
plt.title("Cases due to", y=1.02)


# The value 'overseas inflow' means that the infection is from other country.
# Tha value 'etc' includes individual cases, cases where relevance classification is ongoing after investigation, and cases under investigation.
# 
# 1. All the causes caused the same amount of spread  

# In[ ]:


other_city_case_grp = other_cities_grp.pivot_table(index='infection_case',aggfunc = 'sum').reset_index()

other_city_case_grp['confirmed'] = other_city_case_grp['confirmed']
sns.barplot(x= 'confirmed',y = 'infection_case',data= other_city_case_grp).set(xlabel='Cause', ylabel='confirmed cases', title='Reasons of cases')


# 1. Cases confirmed due to coming in contact with the patient is 1240 which is the most cases due to the spread from outside.
# 2. Cases from overseas inflow is the lowest which 745.
# 3. Other causes also contributed 1024 cases.

# In[ ]:


g = sns.catplot(x="province", y="confirmed", hue="infection_case", data=other_cities_grp,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels('confirmed cases')
g.fig.suptitle('Confirmed cases in cities due to different causes')
g.fig.set_figwidth(25)
g.fig.set_figheight(25)
g.fig.autofmt_xdate()


# ## Conclusion
# 1. It can be concluded that Daegu is the most affected city among all and is suffer by both local and overseas transmission main reason being the contact with the patient.
# 2. Seoul being the the province with most number of affected cities controlled the spread really well. Having the most number of overseas inflow transmission, it can be concluded that it cacn be a tourist place or may be many NRIs.
# 3. Number of cases from local transmission is 6785 while cases from overseas is 3009.
# 4. 'Contact with the patient' caused the most number of cases.
# 5. Reasons for many cases are still unknown or under investigation.
# 

# ### 2) PatientInfo
# #### Epidemiological data of COVID-19 patients in South Korea
# 1. patient_id: the ID of the patient
#   > - patient_id(10) = region_code(5) + patient_number(5)
#   > - You can check the region_code in 'Region.csv'
#   > - There are two types of the patient_number  
#       1) local_num: The number given by the local government.  
#       2) global_num: The number given by the KCDC  
# - global_num: the number given by KCDC
#   > - There are some patients having no global_num.
#   > - The paitents in Busan doesn't have the global_num.
# - sex: the sex of the patient
# - birth_year: the birth year of the patient
# - age: the age of the patient
#   > - 0s: 0 ~ 9  
#   > - 10s: 10 ~ 19  
#   ...  
#   > - 90s: 90 ~ 99  
#   > - 100s: 100 ~ 109
# - country: the country of the patient
# - province: the province of the patient
# - city: the city of the patient
# - disease: TRUE: underlying disease / FALSE: no disease
# - infection_case: the case of infection
# - infection_order: the order of infection
# - infected_by: the ID of who infected the patient
#   > - This column refers to the  'patient_id' column. 
# - contact_number: the number of contacts with people
# - symptom_onset_date: the date of symptom onset
# - confirmed_date: the date of being confirmed
# - released_date: the date of being released
# - deceased_date: the date of being deceased
# - state: isolated / released / deceased
#   > - isolated: being isolated in the hospital
#   > - released: being released from the hospital
#   > - deceased: being deceased

# In[ ]:


p_info.head()


# ### 3) PatientRoute
# #### Route data of COVID-19 patients in South Korea
# - patient_id: the ID of the patient
# - global_num: the number given by KCDC
# - date: YYYY-MM-DD
# - province: Special City / Metropolitan City / Province(-do)
# - city: City(-si) / Country (-gun) / District (-gu)
# - latitude: the latitude of the visit (WGS84)
# - longitude: the longitude of the visit (WGS84)

# In[ ]:


p_route.head()


# ### 4) Time
# #### Time series data of COVID-19 status in South Korea
# - date: YYYY-MM-DD
# - time: Time (0 = AM 12:00 / 16 = PM 04:00)
#   > - The time for KCDC to open the information has been changed from PM 04:00 to AM 12:00 since March 2nd.
# - test: the accumulated number of tests
#   > - A test is a diagnosis of an infection.
# - negative: the accumulated number of negative results
# - confirmed: the accumulated number of positive results
# - released: the accumulated number of releases
# - deceased: the accumulated number of deceases

# In[ ]:


time.head()


# ### 5) TimeAge
# #### Time series data of COVID-19 status in terms of the age in South Korea
# - date: YYYY-MM-DD
#   > - The status in terms of the age has been presented since March 2nd.
# - time: Time
# - age: the age of patients
# - confirmed: the accumulated number of the confirmed
# - deceased: the accumulated number of the deceased

# In[ ]:


t_age.head()


# ### 6) TimeGender
# #### Time series data of COVID-19 status in terms of the gender in South Korea
# - date: YYYY-MM-DD
#   > - The status in terms of the gender has been presented since March 2nd.
# - time: Time
# - sex: the gender of patients
# - confirmed: the accumulated number of the confirmed
# - deceased: the accumulated number of the deceased

# In[ ]:


t_gender.head()


# ### 7) TimeProvince
# #### Time series data of COVID-19 status in terms of the Province in South Korea
# - date: YYYY-MM-DD
# - time: Time
# - province: the province of South Korea
# - confirmed: the accumulated number of the confirmed in the province
#   > - The confirmed status in terms of the provinces has been presented since Feburary 21th.
#   > - The value before Feburary 21th can be different.
# - released: the accumulated number of the released in the province
#   > - The confirmed status in terms of the provinces has been presented since March 5th.
#   > - The value before March 5th can be different.
# - deceased: the accumulated number of the deceased in the province
#   > - The confirmed status in terms of the provinces has been presented since March 5th.
#   > - The value before March 5th can be different.

# In[ ]:


t_provin.head()


# ### 8) Region
# #### Location and statistical data of the regions in South Korea
# - code: the code of the region
# - province: Special City / Metropolitan City / Province(-do)
# - city: City(-si) / Country (-gun) / District (-gu)
# - latitude: the latitude of the visit (WGS84)
# - longitude: the longitude of the visit (WGS84)
# - elementary_school_count: the number of elementary schools
# - kindergarten_count: the number of kindergartens
# - university_count: the number of universities
# - academy_ratio: the ratio of academies
# - elderly_population_ratio: the ratio of the elderly population
# - elderly_alone_ratio: the ratio of elderly households living alone
# - nursing_home_count: the number of nursing homes
# 
# Source of the statistic: [KOSTAT (Statistics Korea)](http://kosis.kr/)

# In[ ]:


region.head()


# ### 9) Weather
# #### Data of the weather in the regions of South Korea
# - code: the code of the region
# - province: Special City / Metropolitan City / Province(-do)
# - date: YYYY-MM-DD
# - avg_temp: the average temperature
# - min_temp: the lowest temperature
# - max_temp: the highest temperature
# - precipitation: the daily precipitation
# - max_wind_speed: the maximum wind speed
# - most_wind_direction: the most frequent wind direction
# - avg_relative_humidity: the average relative humidity
# 
# Source of the weather data: [KMA (Korea Meteorological Administration)](http://data.kma.go.kr)

# In[ ]:


weather.head()


# ### 10) SearchTrend
# #### Trend data of the keywords searched in NAVER which is one of the largest portal in South Korea
# - date: YYYY-MM-DD
# - cold: the search volume of 'cold' in Korean language
#   > - The unit means relative value by setting the highest search volume in the period to 100.
# - flu: the search volume of 'flu' in Korean language
#   > - Same as above.
# - pneumonia: the search volume of 'pneumonia' in Korean language
#   > - Same as above.
# - coronavirus: the search volume of 'coronavirus' in Korean language
#   > - Same as above.
# 
# 
# Source of the data: [NAVER DataLab](https://datalab.naver.com/)

# In[ ]:


search.head()


# ### 11) SeoulFloating
# #### Data of floating population in Seoul, South Korea (from SK Telecom Big Data Hub)
# 
# - date: YYYY-MM-DD
# - hour: Hour
# - birth_year: the birth year of the floating population
# - sext: he sex of the floating population
# - province: Special City / Metropolitan City / Province(-do)
# - city: City(-si) / Country (-gun) / District (-gu)
# - fp_num: the number of floating population
# 
# Source of the data: [SKT Big Data Hub](https://www.bigdatahub.co.kr)

# In[ ]:


floating.head()


# In[ ]:




