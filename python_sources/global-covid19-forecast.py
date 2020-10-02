#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#warnings
import warnings
warnings.filterwarnings('ignore')

#folium
import folium
#plotly
import plotly.express as px
import plotly.figure_factory as ff





# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # **LOAD DATA**

# In[ ]:


#WEEK2 DATA
tw2=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")
test_week2=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")

#Population 2020
population=pd.read_csv("/kaggle/input/population-data/Population_2020.csv")

#WEATHER
weather=pd.read_csv("/kaggle/input/weather-data/weather.csv")



# # **DATA CLEANING**

# In[ ]:


#TW2.............................
tw2=tw2.rename(columns={"Province_State":"Province","Country_Region":"Country","ConfirmedCases":"Confirmed"})
tw2["Province"]=tw2["Province"].fillna('')

#WEATHER DATA...........................
weather=weather.drop(columns=["Unnamed: 0","Confirmed","Fatalities","capital","Province","Id"])
weather["humidity"]=weather["humidity"].fillna(0)
weather["sunHour"]=weather["sunHour"].fillna(0)
weather["tempC"]=weather["tempC"].fillna(0)
weather["windspeedKmph"]=weather["windspeedKmph"].fillna(0)



#POPULATION
population=population.drop('Unnamed: 0',axis=1)
population["fertility"]=population["fertility"].fillna(0)
population["age"]=population["age"].fillna(0)
population["urban_percentage"]=population["urban_percentage"].fillna(0)



# # **CREATING DATA**

# # **Merge tw2 and weather**

# In[ ]:


data=pd.merge(tw2,weather,on=["Country","Date"],how="inner")


# In[ ]:


data.info()


# # **Merge population and data**

# In[ ]:


data=pd.merge(data,population,on=["Country"],how="inner")


# In[ ]:


data.info()


# In[ ]:


#CHANGING FLOAT TO INT
data["Confirmed"]=data["Confirmed"].astype(int)
data["Fatalities"]=data["Fatalities"].astype(int)

#CONVERTING DATE OBJECT TO DATETIME FORMAT

data["Date"]=pd.to_datetime(data["Date"])


# In[ ]:


data.info()


# In[ ]:


data.shape


# In[ ]:


data["Date"].min(),data["Date"].max()


# In[ ]:


data.head()


# In[ ]:


data_latest=data[data["Date"]==max(data["Date"])].reset_index()
data_latest=data_latest.drop(['index','Id'],axis=1)

data_grouped=data.groupby(["Province","Country","Date"])["Confirmed",
                                                         "Fatalities",
                                                        "Lat","Long","humidity","sunHour","tempC",
                                                        "windspeedKmph","Population",
                                                        "density","fertility","age",
                                                        "urban_percentage"].agg({"Confirmed":"sum",
                                                                                "Fatalities":"sum",
                                                                                "Lat":"mean",
                                                                                "Long":"mean",
                                                                                "humidity":"mean",
                                                                                "sunHour":"mean",
                                                                                "tempC":"mean",
                                                                                "windspeedKmph":"mean",
                                                                                "Population":"mean",
                                                                                "density":"mean",
                                                                                "fertility":"mean",
                                                                                "age":"mean",
                                                                                "urban_percentage":"mean"}).reset_index()

data_grouped["Date"]=data_grouped["Date"].dt.strftime("%m/%d/%Y")

data_latest_grouped=data_latest.groupby(["Country"])["Confirmed","Fatalities","Population","Lat","Long"].agg({"Confirmed":"sum",
                                                                                                 "Fatalities":"sum",
                                                                                                 "Population":"mean",
                                                                                                 "Lat":"mean","Long":"mean"}).reset_index()


# In[ ]:


gdf=data.groupby(["Country"])["Confirmed",
                             "Fatalities",
                            "Lat","Long","humidity","sunHour","tempC",
                            "windspeedKmph","Population",
                            "density","fertility","age",
                            "urban_percentage"].agg({"Confirmed":"sum",
                                                    "Fatalities":"sum",
                                                    "Lat":"mean",
                                                    "Long":"mean",
                                                    "humidity":"mean",
                                                    "sunHour":"mean",
                                                    "tempC":"mean",
                                                    "windspeedKmph":"mean",
                                                    "Population":"mean",
                                                    "density":"mean",
                                                    "fertility":"mean",
                                                    "age":"mean",
                                                    "urban_percentage":"mean"}).reset_index()


# In[ ]:


data_latest.head()


# In[ ]:


data_grouped.head()


# # **SPREAD OF CORONAVIRUS OVER TIME ON MAP**

# In[ ]:


#SPREAD OVER TIME...........................................

fig=px.scatter_geo(data_grouped,locations="Country",
                  locationmode="country names",
                   color=np.log(data_grouped["Confirmed"]),
                  animation_frame="Date",
                   size=data_grouped["Confirmed"].pow(0.3),
                  projection="natural earth",
                  hover_name="Country",
                  title="Spread of Coronavirus Over time")

fig.show()


# # **WORLDWIDE CASES CONFIRMED**

# In[ ]:


#TOTAL CONFIRMED CASES AROUND THE WORLD

fig=px.choropleth(gdf,locations="Country",
                 locationmode="country names",
                 color=np.log(gdf["Confirmed"]),
                  hover_name="Country",
                  hover_data=["Confirmed","Population","tempC","windspeedKmph","humidity","sunHour"],
                  title="Total Confirmed cases around the world")

fig.show()


# As expected china will be having most no of confirmed cases
# The other countries like USA , France , Italy , Iran are also badly infected

# In[ ]:


most_con=data_latest_grouped.sort_values(by="Confirmed",ascending=False)[0:10].reset_index(drop=True).style.background_gradient(cmap="Reds")
most_con


# In[ ]:


fig=px.bar(data_latest_grouped.sort_values(by="Confirmed")[-10:],x="Confirmed",y="Country",
          color_discrete_sequence=['dark cyan'],orientation='h',
           text='Confirmed',height=800,title="Confirmed Cases")
fig.show()


#  As we can see from the table and the figure the most affected countries are china,us,france,italy and so on with china the most affected.
#  We can expect USA to have more confirmed cases by start of april.

# In[ ]:


temp_con=data.groupby(["Country","Date"])["Confirmed","Fatalities"].sum().reset_index()
temp_con["Date"]=temp_con["Date"].dt.strftime("%m/%d/%Y")


# In[ ]:


temp_con.head()


# In[ ]:


#LINE CHART OF CONFIRMED CASES OF EACH COUNTRY

fig=px.line(temp_con,x="Date",y="Confirmed",color="Country",title="Confirmed cases in each country over time")

fig.show()


# By mid march ,spread of virus in china shows a flat line,that tells us that china is recovering 
# On the other hand , if wee see other countries like US,Italy,France ,we can expect more confirmed cases by april

# # **WORLD FATALITIES**

# In[ ]:


#TOTAL FATALITY CASES AROUND THE WORLD

fig=px.choropleth(gdf,locations="Country",
                 locationmode="country names",
                 color=np.log(gdf["Fatalities"]),
                  hover_name="Country",
                  hover_data=["Fatalities","Population","tempC","windspeedKmph","humidity","sunHour"],
                  title="Total Fatality cases around the world")

fig.show()


# The above map shows that china has most death cases ,and then USA ,Italy ,France, Iran have most death cases

# # **TOP 10 COUNTRIES WITH MOST FATALITY CASES**

# In[ ]:


most_fat=data_latest_grouped.sort_values(by="Fatalities",ascending=False)[0:10].reset_index(drop=True).style.background_gradient(cmap="Reds")
most_fat


# China and USA have most death cases ,But Italy has less confirmed cases than france but still more fatality cases than france

# In[ ]:


#For a better visualization

fig=px.bar(data_latest_grouped.sort_values(by="Fatalities")[-10:],x="Fatalities",y="Country",
          color_discrete_sequence=['dark cyan'],orientation='h',
           text='Fatalities',height=800,title="Fatality Cases")
fig.show()


# In[ ]:


#LINE CHART OF CONFIRMED CASES OF EACH COUNTRY

fig=px.line(temp_con,x="Date",y="Fatalities",color="Country",title="Fatality cases in each country over time")

fig.show()


# From the above plot, we can expect usa,italy ,iran ,france to have more cases of fatality in coming weeks

# If we have look at the latest data , we can have a better understanding of which country is recovering and which one may be more infected in coming weeks

# In[ ]:


#LATEST CONFIRMED CASES AROUND THE WORLD

fig=px.choropleth(data_latest_grouped,locations="Country",
                 locationmode="country names",
                 color=np.log(data_latest_grouped["Confirmed"]),
                  hover_name="Country",
                  hover_data=["Confirmed","Population"],
                  title="Total Confirmed cases around the world")

fig.show()


# In[ ]:


#LATEST FATALITY CASES AROUND THE WORLD

fig=px.choropleth(data_latest_grouped,locations="Country",
                 locationmode="country names",
                 color=np.log(data_latest_grouped["Fatalities"]),
                  hover_name="Country",
                  hover_data=["Fatalities","Population"],
                  title="Latest Fatality cases around the world")

fig.show()


# By looking at the latest data,confirmed and fatality cases in usa , italy , iran have increased drastically in few weeks and can expect more in coming weeks.

# # **ANALYZING FEW FACTORS**

# **HUMIDITY**

# As we know from above maps, the most infected countries have humidity around 30 and 80

# In[ ]:


fig=px.bar(gdf.sort_values(by='humidity')[-10:],
           x="Confirmed",y="Country",
          color_discrete_sequence=['dark cyan'],orientation='h',
           text="Confirmed",hover_data=["Confirmed","Fatalities","Population","humidity"],
                                         height=500,title="Highest humidity countries")
fig.show()


# Kazakhstan has highest humidity but only 243 confirmed cases, also Canada has humidity of 88 and also around 67,000 confirmed cases So we can't say much looking at the humidity

# **POPULATION**

# In[ ]:


fig=px.bar(gdf.sort_values(by='Population')[-10:],
           x="Confirmed",y="Country",
          color_discrete_sequence=['dark cyan'],orientation='h',
           text="Confirmed",hover_data=["Confirmed","Fatalities","Population"],
                                         height=500,title="Most populated countries")
fig.show()


# India with second highest population has around 977 cases and same with US We can say that countries with more population can have more cases .i.e true as if there are more people chances of virus spreading is more

# In[ ]:


fig=px.bar(gdf.sort_values(by='Population')[0:10][::-1],
           x="Confirmed",y="Country",
          color_discrete_sequence=['dark cyan'],orientation='h',
           text="Confirmed",hover_data=["Confirmed","Fatalities","Population"],
                                         height=500,title="Most populated countries")
fig.show()


# But this plot shows that less populated countries like iceland also have more confirmed cases,
# So virus does spread in populated countries but also has a chance of spreading in countries wiht less population

# **WINDSPEED**

# What is expect is that countries that have high windspeed should have more cases , As virus can easily spread in such countries

# In[ ]:


fig=px.bar(gdf.sort_values(by='windspeedKmph')[-10:],
           x="Confirmed",y="Country",
          color_discrete_sequence=['dark cyan'],orientation='h',
           text="Confirmed",hover_data=["Confirmed","Fatalities","windspeedKmph"],
                                         height=500,title="Countries with high wind speed")
fig.show()


# In[ ]:


fig=px.bar(gdf.sort_values(by='windspeedKmph')[0:10][::-1],
           x="Confirmed",y="Country",
          color_discrete_sequence=['dark cyan'],orientation='h',
           text="Confirmed",hover_data=["Confirmed","Fatalities","windspeedKmph"],
                                         height=500,title="Countries with less wind speed")
fig.show()


# As we see in both the graphs,countries with high wind speed have confirmed cases but the one's with less wind speed also have more number of confirmed cases

# **TEMPERATURE**

# As we saw in above maps,colder countries have higher number of cases. The same we can expect from the following graphs

# In[ ]:


fig=px.bar(gdf.sort_values(by='tempC')[-10:],
           x="Confirmed",y="Country",
          color_discrete_sequence=['dark cyan'],orientation='h',
           text="Confirmed",hover_data=["Confirmed","Fatalities","tempC","Population","windspeedKmph"],
                                         height=500,title="Countries with high temperature")
fig.show()


# Among all the hotter countries ,thailand is having around 3000 cases which can be because of higher population

# Now we'll have a look at the colder countries

# In[ ]:


fig=px.bar(gdf.sort_values(by='tempC')[0:20][::-1],
           x="Confirmed",y="Country",
          color_discrete_sequence=['dark cyan'],orientation='h',
           text="Confirmed",hover_data=["Confirmed","Fatalities","tempC","Population","windspeedKmph"],
                                         height=800,title="Countries with low temperature")
fig.show()


# As we can see in a list of colder countries ,some have higher number of cases, but not all as expected Hence we can't say that countries with low temperature will have higher chance of more confirmed cases

# **HOURS OF SUNLIGHT**

# We expect that countries that have higher hours of sunlight,people interactions which can lead to increase in spreading of virus

# In[ ]:


fig=px.bar(gdf.sort_values(by='sunHour')[-20:],
           x="Confirmed",y="Country",
          color_discrete_sequence=['dark cyan'],orientation='h',
           text="Confirmed",hover_data=["Confirmed","Fatalities","tempC","Population","windspeedKmph","sunHour"],
                                         height=800,title="Countries with more hours of sunlight")
fig.show()


# So, countries with more hours of sunlight have higher chances of increased spreading of virus

# Lets look at the countries with less hours of sunlight

# In[ ]:


fig=px.bar(gdf.sort_values(by='sunHour')[0:20][::-1],
           x="Confirmed",y="Country",
          color_discrete_sequence=['dark cyan'],orientation='h',
           text="Confirmed",hover_data=["Confirmed","Fatalities","tempC","Population","windspeedKmph","sunHour"],
                                         height=800,title="Countries with more hours of sunlight")
fig.show()


# Only germany, france ,netherlands and few other countries have higher number of confirmed cases which can also be due to other factors

# # **MODEL**

# **CREATING TRAINING DATA**

# In[ ]:


#We have training data as (data) dataset
#We'll split this data into training and evaluating datasets

#For that we need to drop some columns and create a train_data dataset

train_data=data.drop(columns=["Id","Province","Country","Date","Lat","Long"],axis=1)


# In[ ]:


train_data.head()


# # **CREATING X AND y FROM TRAIN DATA**

# In[ ]:


#We'll create X and y 
#X will have all dependent features
#y will have target variables

y=train_data[["Confirmed","Fatalities"]]
X=train_data.drop(columns=["Confirmed","Fatalities"],axis=1)


# In[ ]:


X.head()


# In[ ]:


y.head()


# # **SPLITTING X AND y**

# In[ ]:


from sklearn.model_selection import train_test_split as tts

X_train,X_val,y_train,y_val=tts(X,y,test_size=0.2,random_state=42)


# In[ ]:


#training and testing data are ready
#We'll be using Random Forest Classifier

from sklearn.ensemble import RandomForestRegressor

#Model for predicting Confirmed cases
rf_confirmed=RandomForestRegressor(n_estimators=1000, random_state = 42)
#Model for predicting Fatality cases
rf_fatality=RandomForestRegressor(n_estimators=1000,random_state=42)


# # **FITTING ON CONFIRMED CASES**

# In[ ]:


#FITTING CONFIRMED MODEL TO TRAINING DATA
rf_confirmed.fit(X_train,y_train["Confirmed"])


# In[ ]:


#PREDICTING ON EVALUATING DATA
result_confirmed=rf_confirmed.predict(X_val)


# In[ ]:


#Error
from sklearn.metrics import mean_squared_log_error


# In[ ]:


error_confirmed=np.sqrt(mean_squared_log_error(y_val["Confirmed"],result_confirmed))
print(error_confirmed)


# # **FITTING ON FATALITY CASES**

# In[ ]:


rf_fatality.fit(X_train,y_train["Fatalities"])


# In[ ]:


result_fatality=rf_fatality.predict(X_val)


# In[ ]:


#Error
error_fatality=np.sqrt(mean_squared_log_error(y_val["Fatalities"],result_fatality))
print(error_fatality)


# # **FINAL VALIDATION SCORE**

# In[ ]:


print("Final Validatio score: {}".format(np.mean([error_confirmed,error_fatality])))


# # **FINAL MODEL FITTING**

# # **MODEL_CONFIRMED AND MODEL_FATALITIES**

# In[ ]:


model_confirmed=rf_confirmed.fit(X,y["Confirmed"])
model_fatalities=rf_fatality.fit(X,y["Fatalities"])


# # EXTRACTING FEATURE IMPORTANCES

# # IMP FEATURES FOR CONFIRMED

# In[ ]:


# Extract feature importances for confirmed
fi_con = pd.DataFrame({'feature': list(X.columns),
                   'importance': model_confirmed.feature_importances_})


# In[ ]:


fi_con.sort_values(by="importance",ascending=False).reset_index(drop=True)


# In[ ]:


# Get list of important variables for predicting confirmed cases
importances_confirmed = list(model_confirmed.feature_importances_)


# In[ ]:


features_list=list(X.columns)


# In[ ]:


#With data visualization for important variables for confirmed cases

# Set the style
plt.style.use('ggplot')
# list of x locations for plotting
x_values = list(range(len(importances_confirmed)))
# Make a bar chart
plt.bar(x_values, importances_confirmed, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, features_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances for Confirmed cases')


# We can clearly see population,temperature,humidity,windspeed,hours of sunlight are among the important factors that affect number of confirmed cases

# More the population ,higher chance of spreading of virus ,same with more hours of sunlight more chances of people interaction. Weather conditions play an important role in spreading of virus

# # IMP FEATURES FOR FATALITIES

# In[ ]:


# Extract feature importances for fatalities
fi_fatalities = pd.DataFrame({'feature': list(X.columns),
                   'importance': model_fatalities.feature_importances_})


# In[ ]:


fi_fatalities.sort_values(by="importance",ascending=False).reset_index(drop=True)


# In[ ]:


# Get a list of important variables for predicting fatality cases
importances_fatalities = list(model_fatalities.feature_importances_)


# In[ ]:


features_list=list(X.columns)


# In[ ]:


#With data visualization for fatality cases

# Set the style
plt.style.use('ggplot')
# list of x locations for plotting
x_values = list(range(len(importances_fatalities)))
# Make a bar chart
plt.bar(x_values, importances_fatalities, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, features_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances for fatality cases')


# More population more chances of spreading of virus ,also we can see there is slight importance that age plays in fatality cases.As older people have lesser immunity and higher chance of contracting the disease.Here also the other weather conditions play an important role

# # **CREATING TESTING DATA**

# In[ ]:


test_week2.head()


# In[ ]:


test_week2=test_week2.rename(columns={"ForecastId":"Id","Province_State":"Province","Country_Region":"Country"})
test_week2["Province"]=test_week2["Province"].fillna('')


# In[ ]:


weather.head()


# In[ ]:


test_week2.head()


# # **MERGING TEST AND WEATHER DATA**

# In[ ]:


test_df=test_week2.merge(weather,on=["Country","Date"],how='left')
test_df.head()


# # **MERGING TEST_DF AND POPULATION DATA**

# In[ ]:


test_df=test_df.merge(population,on=["Country"],how="left")
test_df.head()


# In[ ]:


test_df.info()


# # **CREATING X_TEST FROM TEST_DF**

# In[ ]:


X_test = test_df.set_index("Id").drop(["Lat", "Long", "Date", "Province", "Country"], axis=1).fillna(0)
X_test.head()


# In[ ]:


X_test.info()


# **EVALUATING ON TEST DATA**

# In[ ]:


y_pred_confirmed = model_confirmed.predict(X_test)
y_pred_fatalities = model_fatalities.predict(X_test)


# In[ ]:


len(y_pred_confirmed)


# # **CREATING SUBMISSION FILE**

# In[ ]:


submission = pd.DataFrame()
submission["ForecastId"]= pd.to_numeric(test_df["Id"], errors= 'coerce')
submission["ConfirmedCases"] = y_pred_confirmed
submission["Fatalities"] = y_pred_fatalities
submission["ConfirmedCases"]=submission["ConfirmedCases"].astype(int)
submission["Fatalities"]=submission["Fatalities"].astype(int)
submission = submission.drop_duplicates(subset= ['ForecastId'])
submission = submission.set_index(['ForecastId'])
submission.head()


# In[ ]:


submission.to_csv("submission.csv")


# In[ ]:




