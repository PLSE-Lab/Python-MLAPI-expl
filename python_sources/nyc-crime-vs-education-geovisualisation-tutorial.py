#!/usr/bin/env python
# coding: utf-8

# # Intro
# In this notebook report we will exploere the Crime dataset of 2019 in NYC and the after school activities dataset in NYC (both found in the NYC datasets listed below).
# 
# Presnting the different aspects of the data and try to find some connections between education activities and the crime rate. The findings in that manner are'nt that conclusive as we are missing some crucial historical data about historical education activities.
# 
# Finally, I will build a simple Multi Target Regression (MTR) model, with 2 targets the lat and long coordinates. The model can predict  future location of crimes base on the future criminal profile and weekday. <br> 

# ## The Datasets
# Using the newer version of the crime data set.
# 
# After school activities NYC - https://data.cityofnewyork.us/Education/DYCD-after-school-programs/mbd7-jfnc
# 
# Arrest dataset of 2019 - https://data.cityofnewyork.us/Public-Safety/NYPD-Arrest-Data-Year-to-Date-/uip8-fykc

# ### The imports:

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import datetime
from datetime import date
import math
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects
from sklearn.metrics import mean_squared_error
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn import preprocessing
from random import sample 
import plotly.express as px

sns.set(rc={'figure.figsize':(12,10)})
sns.set(style="white", context="talk")

get_ipython().run_line_magic('matplotlib', 'inline')


# # Exploring NYPD Arrests dataset

# In[ ]:


arrest = pd.read_csv("../input/nypd-crime/NYPD_Arrest_Data__Year_to_Date_ (1).csv")
arrest[:5]


# Sorting some of the data out (code to meanings, droping columns, etc.)

# In[ ]:


def date_to_weekday(date):
    weekday_dict = {0:'Monday', 1:'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    date_time_obj = datetime.datetime.strptime(date, '%m/%d/%Y')
    return weekday_dict[date_time_obj.weekday()]
def code_to_loc(code):
    code_dict = {'B': 'Bronx', 'S': 'Staten Island', 'K': 'Brooklyn', 'M': 'Manhattan' , 'Q': 'Queens'}
    return code_dict[code]
def code_to_fel(code):
    code_dict = {'F': 'Felony','M': 'Misdemeanor', 'V': 'Violation', 'I': 'Other'}
    if code in code_dict:
        return code_dict[code]
    else:
        return 'Other'

date = arrest['ARREST_DATE'].str.split("/", n = 3, expand = True)
arrest['year'] = date[2].astype('int32')
arrest['day'] = date[1].astype('int32')
arrest['month'] = date[0].astype('int32')

arrest['ARREST_BORO'] = arrest['ARREST_BORO'].apply(code_to_loc)
arrest['WEEKDAY'] = arrest['ARREST_DATE'].apply(date_to_weekday)
arrest['LAW_CAT_CD'] = arrest['LAW_CAT_CD'].apply(code_to_fel)

arrest = arrest.drop(['ARREST_KEY', 'PD_CD', 'PD_DESC', 'KY_CD', 'LAW_CODE', 'JURISDICTION_CODE', 'X_COORD_CD', 'Y_COORD_CD'], axis=1)
arrest[:5]


# ### Lets further explorer the data
# 
# Is there any specific day within the month that has more crimes?

# In[ ]:


f, ax = plt.subplots(figsize=(25, 15))
sns.countplot(y="day", data=arrest, palette="pastel");


# It seems that the 31th is the lowest crime rate day, that make sense as there isn't a 31th day in evey month...
# 
# is there a more crime month then?

# In[ ]:


f, ax = plt.subplots(figsize=(10, 7))
sns.countplot(x="month", data=arrest)


# Not really, Feb is a bit less crimey (Cause it is probably more cold) but still not significant
# 
# Let's check the most arrested race:

# In[ ]:


sns.catplot(x="PERP_RACE", data=arrest,kind="count", palette="pastel", height=15, aspect=1.5);


# It seems that the protest about it is true there are much more Black arrests than others, running second is the Hispanic whites
# 
# What are the ages of the criminals?

# In[ ]:


f, ax = plt.subplots(figsize=(10, 8))
sns.countplot(y="AGE_GROUP", data=arrest, palette="pastel");


# The 20's - 40's seems like the most crime age
# 
# In which Borough in the city has the most crime?

# In[ ]:


## Borough of arrest. B(Bronx), S(Staten Island), K(Brooklyn), M(Manhattan), Q(Queens)
f, ax = plt.subplots(figsize=(10, 8))
sns.countplot(x="ARREST_BORO", data=arrest, palette="pastel");


# So Brooklyn has the most crime rate in the city at 2019, Manhatten is close second
# 
# Is there any differences within the sexes within the different races?

# In[ ]:


ax = sns.catplot(x="PERP_RACE", hue="PERP_SEX", kind="count",palette="cubehelix", data=arrest, height=15, aspect=2)


# We can observe that males conduct more crimes (or arrested crimes at least)
# 
# can we see if there is any changes within races and sexes in the different locations?

# In[ ]:


sns.catplot(x="ARREST_BORO", kind="count",hue="PERP_SEX",palette="Set2", data=arrest,height=12, aspect = 2);


# Nothing that significant
# 
# Maybe within the Age groups?

# In[ ]:


ax = sns.catplot(x="AGE_GROUP", hue="PERP_SEX", kind="count",palette="cubehelix", data=arrest, height=10, aspect = 2)


# Ok still not that significant
# 
# Let's move to more intresting and complex combinations:
# 
# What is the crime distributtion within the location of the different races

# In[ ]:


ax = sns.catplot(x="ARREST_BORO", hue="PERP_RACE", kind="count",palette="cubehelix", data=arrest, height=10, aspect = 2)


# Blacks lead everywhere except staten island, white hispanic are always second. Another thing is that white crime in the Bronx is signifcantly lower than other places...
# 
# Lets see if there is a certain day within the week that changes the crime rate (several checks like districst, sex, age)

# In[ ]:


ax = sns.catplot(x="WEEKDAY", hue="ARREST_BORO", kind="count",palette="bright", data=arrest, height=10, aspect = 2)
ax = sns.catplot(x="WEEKDAY", hue="AGE_GROUP", kind="count",palette="rainbow", data=arrest, height=10, aspect = 2)


# Wednesday has the most crimes, and sunday is the lowest, so the safest day is sunday in NYC!

# # Map Visualisation 
# Lets show it over the map with interactive plotting - 2 forms of interactive ploting heatmap and plotly scatter plot (with time animation)

# In[ ]:


# positions = [] 
# for index, row in arrest.iterrows():
positions = list(zip(arrest['Latitude'], arrest['Longitude']))
tiles = 'Stamen Terrain'
fol = folium.Map(location=[40.75,-73.98], zoom_start=10, tiles = tiles)
pos_samp = sample(positions, 22000)#22K is the max now as we join both DS togather 
HeatMap(pos_samp, radius = 8).add_to(fol) 
fol


# In[ ]:


px.set_mapbox_access_token(open("../input/map-key/key").read()) ## key needed for the API of the maps in plotly
arrest["size"] = 1
fig = px.scatter_mapbox(arrest, lat="Latitude", lon="Longitude", color="AGE_GROUP", size="size", animation_frame="month",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)
fig.show()
arrest = arrest.drop('size', axis = 1)


# The heatmap is far more informative so we will stick with it further along this analysis. <br>
# Eventhough that the interactive plot is very cool, if you zoom in and press play it looks like NYC disco!

# # After School activities dataset
# Let's proceed with the after school activities (we will come back to the bigger historical crime dataset later)
# 
# First some exploration:

# In[ ]:


edu = pd.read_csv("../input/nypd-crime/DYCD_after-school_programs (1).csv")
edu[:5]


# We will focus on the major boroughs in NYC:

# In[ ]:


boro_list = ['Bronx', 'Staten Island', 'Brooklyn', 'Manhattan' , 'Queens']
edu_boro = edu[edu['BOROUGH / COMMUNITY'].isin(boro_list)]


# In[ ]:


names = edu_boro.groupby('BOROUGH / COMMUNITY').count().index
my_circle=plt.Circle( (0,0), 0.7, color='white')
f, ax = plt.subplots(figsize=(15, 12))
cmap = plt.get_cmap('Spectral')
colors = [cmap(i) for i in np.linspace(0, 1, 6)]
plt.pie(edu_boro.groupby('BOROUGH / COMMUNITY').count()['Postcode'], labels=names, colors=colors, shadow=True)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()


# We can see that Brooklyn has the most activities (it make sense as it has more crime). Another intresting fact is that Manhattan which is the second after brooklyn in the crime rate has the 4th amount of activities (and very close to Staten island).
# 
# Let's draw the heatmap for the activities this time

# In[ ]:


positions = [] 
for index, row in edu.iterrows():
    if not math.isnan(row['Latitude']) :
        positions.append((row['Latitude'], row['Longitude']))
fol = folium.Map(location=[40.75,-73.98],tiles='Stamen Toner', zoom_start=11)
HeatMap(positions[:38000], radius = 8).add_to(fol)
fol


# As we saw Manhattan does have a shortage in after school activities!
# 
# Now let's combine both the crime heatmap and the educational activities on the same map (Activities locations marked in red and the heatmap signals the crime)

# In[ ]:


positions_edu = [] 

for index, row in edu.iterrows():
    if not math.isnan(row['Latitude']) :
        positions_edu.append((row['Latitude'], row['Longitude']))
positions_arr = list(zip(arrest['Latitude'], arrest['Longitude']))
fol = folium.Map(location=[40.75,-73.98], zoom_start=11, control_scale=True)

pos_samp = sample(positions_arr, 22000)#22K is the max now as we join both DS togather 
HeatMap(pos_samp, radius = 7).add_to(fol) 

for pos in positions_edu:
    folium.CircleMarker(location=[pos[0],pos[1]], radius=1, color='red', fill=False,).add_to(fol)
fol


# # Mid Summary & Conclusions so far
# 
# We can definitely see that in Manhattan where there are much less after school activities there is a bigger crime rate (In the Mid-Town and Hell's Kitchen districts)
# 
# Altough we can see that other districts have got more dense after school activities and they still have a sginificant crime rate (for example the Bronx or lower Manhattan), eventhough we have checked for all ages crime commiters.
# 
# In addition altough brooklyn is leading the table with the amount of crime as it is relativly big area the thus crime activity is less dense and the effectivness of the educational activities might be higher. of course there are alot of other influances like amount of police activity (it might also raise the amount of arrests) and others. 
# 
# Furthermore, we lack the historical data of the after school activities so we cannot show the changes within crime rate of activities over time

# # Creating Crime predictions with MTR model:
# Lets try to create MTR model to predict the loaction of a crime base on some basic features and then using the data lets try to forcast the next month. <br>
# I will check 2 different models Random forest and gradient boost regressors as the base estimators for the MTR. <br>
# First I will create the train and test data, later check the MSE of the models and later will present them on the map

# In[ ]:


def cat_to_num(df , col_name):
    le = preprocessing.LabelEncoder()
    new_col = le.fit_transform(df[col_name])
    return le , new_col

est1 = RandomForestRegressor()
model1 = MultiOutputRegressor(est1, 4)

est2 = GradientBoostingRegressor()
model2 = MultiOutputRegressor(est2, 4)

x_reg_train = arrest[arrest['month']<6].drop(['LAW_CAT_CD','ARREST_DATE','OFNS_DESC','ARREST_PRECINCT', 'day','year'],axis = 1)
x_reg_test = arrest[arrest['month']==6].drop(['LAW_CAT_CD','ARREST_DATE','OFNS_DESC','ARREST_PRECINCT', 'day', 'year'],axis = 1)

y_train = x_reg_train[['Latitude', 'Longitude']]
y_test = x_reg_test[['Latitude', 'Longitude']]

x_reg_train = x_reg_train.drop(['Latitude', 'Longitude'],axis = 1)
x_reg_test = x_reg_test.drop(['Latitude', 'Longitude'],axis = 1)

d = {}
for col in x_reg_train.columns:
    if x_reg_train.dtypes[col] == 'int32':
        continue
    le, new_col = cat_to_num(x_reg_train, col)
    d[col] = le
    x_reg_train[col] = new_col
    x_reg_test[col] = le.transform(x_reg_test[col])
model1.fit(x_reg_train, y_train)
model2.fit(x_reg_train, y_train)


# In[ ]:


pred1 = model1.predict(x_reg_test)
pred2 = model2.predict(x_reg_test)
y_test_mean = y_test.mean(axis=0).tolist()
print("The MSE of RF = {}\nThe MSE of GBR = {} ".format(mean_squared_error(y_test, pred1),mean_squared_error(y_test, pred2)))


# It seems like our model is predicting the upcoming location of crimes (based on thier sex, race, weekday, district, etc) pretty well! <br>
# The GBR performs slightly better when comapring the MSE <br>
# 
# Can we utilize it in order to recommend where should we perform educational activities / more law enforcment? <br>
# Let's try to create some profiles and predict thier future crimes:

# In[ ]:


future_crime = pd.DataFrame({
    'ARREST_BORO':['Brooklyn','Brooklyn', 'Manhattan','Queens', 'Bronx'], 
    'PERP_SEX':['M', 'F' , 'F', 'M' , 'F'], 
    'PERP_RACE':['BLACK', 'WHITE', 'WHITE HISPANIC', "WHITE" , 'ASIAN / PACIFIC ISLANDER'], 
    'WEEKDAY':['Sunday', 'Monday', 'Tuesday', 'Sunday', 'Friday'], 
    'AGE_GROUP':['<18', '25-44', '18-24', '45-64', '25-44'],
    'month':[1,3,4,5,2]
    })
for col in future_crime.columns:
    if future_crime.dtypes[col] == 'int64':
        continue
    future_crime[col] = d[col].transform(future_crime[col])
cords1 = model1.predict(future_crime)
cords2 = model2.predict(future_crime)


# I will not show the examples predicted crime locations on the map. <br>
# Blue is the GBR predicted location <br>
# Red is the RF predicted location <br>
# The heatmap presents the crime rate heatmap we saw earlier

# In[ ]:


positions_arr= list(zip(arrest['Latitude'], arrest['Longitude']))

fol = folium.Map(location=[40.75,-73.98], zoom_start=11, control_scale=True)

pos_samp = sample(positions_arr, 22000)#22K is the max now as we join both DS togather 
HeatMap(pos_samp, radius = 9).add_to(fol) 

for pos in cords1:
    folium.CircleMarker(location=[pos[0],pos[1]], radius=3, color='red', fill=True).add_to(fol)
for pos in cords2:
    folium.CircleMarker(location=[pos[0],pos[1]], radius=3, color='blue', fill=True).add_to(fol)
fol


# We can see that probably through some sampeling and providing several perfiles we can acheive some recommendations about where should we place enforcemnt \ educational activities in order to try and prevent future crimes.
# Optional uses could be creating those sorts of maps and location everyday for the next upcoming days in order to be able to prepare the law force manpower. Furthermore we can try to forcast future crimes in the manner of months and to be able to plan strategic educatinal plans to prevent future crimes trends.

# # Conclusion
# From this report we think that we can conclude several conclusions and insights:
# 1. Crime rate has alot of factors and the data we acheived wasn't sufficant to determine wheter the crime rate and education activities are related. But we can see through the visualities provided that there might be some effect.
# 2. We can recommend to the NYC city council to provide additional education activities in midtown Manhattan as it is signifacntly has less activities than other districes in the city and has a relativly high crime rate.
# 3. Additional recommendation about enforcment, first of all using the heat-map created police may be able to map POI in the city and provide more manpower to those areas. 
# The police could use the suggester model to recieve hints about future locations of crimes and the profile of the future criminal teenager. 
# 4. Furthermore the model created could be utilized for tactiacl crime prevention and also to aid stetegic plans to fight crime around the city in the future.
