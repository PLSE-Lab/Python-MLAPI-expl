#!/usr/bin/env python
# coding: utf-8

# COVID-19 has infected more than 5,000 people in South Korea. 
# 
# South Korea currently has the second highest infection counts in the world.
# 
# Last point of data collected is approx. 5th March 2020.
# 
# Thanks to the open source commitee and help to be able to explore and learn tools and explore this dataset. Special thanks to @vanshjatana for the help and layout.
# 
# ![SouthKorea](http://richiewong.co.uk/wp-content/uploads/2020/03/cait-ellis-Erld-XTqXv0-unsplash-scaled-e1583538082567.jpg)

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
from datetime import date, timedelta
from statsmodels.tsa.arima_model import ARIMA
from sklearn.cluster import KMeans


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_patient=pd.read_csv('/kaggle/input/coronavirusdataset/patient.csv')
df_time=pd.read_csv('/kaggle/input/coronavirusdataset/time.csv')
df_route=pd.read_csv('/kaggle/input/coronavirusdataset/route.csv')
df_trend=pd.read_csv('/kaggle/input/coronavirusdataset/trend.csv')


# # Cases in group
# 
# Let's look at the time dataset - which is a aggregation of the number of test undertaken

# In[ ]:


df_time['Days'] = range(1, 1+len(df_time))


# In[ ]:


df_time = df_time.rename(columns={'176': 'Date'})


# * Date: Year-Month-Day
# * acc_test: the accumulated number of tests
# * acc_negative: the accumulated number of negative results
# * acc_confirmed: the accumulated number of positive results
# * acc_released: the accumulated number of releases
# * acc_deceased: the accumulated number of deceases
# * new_test: the number of new tests
# * new_negative: the number of new negative results
# * new_confirmed: the number of new positive results
# * new_released: the number of new releases
# * new_deceased: the number of new deceases

# In[ ]:


df_time.dtypes


# In[ ]:


df_time.date = pd.to_datetime(df_time.date)


# In[ ]:


df_time['day_of_week'] = df_time['date'].dt.day_name()

df_time.head()


# In[ ]:


df_time.dtypes


# Let's see how the number of test changed

# In[ ]:


plt.figure(figsize=(15,6))
plt.bar(df_time.Days, df_time.new_test, color='blue')
plt.title('Number of tests undertaken', fontsize=15)
plt.xlabel('Days')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.show()


# A lot more test have been taken in day 30+ since 20th January

# In[ ]:


plt.figure(figsize=(15,6))
plt.bar(df_time.Days, df_time.acc_test, color='blue')
plt.title('Number of tests undertaken - Cumulative', fontsize=15)
plt.xlabel('Days')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.show()


# In[ ]:


plt.figure(figsize=(15,6))
plt.bar(df_time.Days, df_time.new_confirmed, color='red')
plt.title('Number of positive case', fontsize=15)
plt.xlabel('Days')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.show()


# In[ ]:


plt.figure(figsize=(15,6))
plt.bar(df_time.day_of_week, df_time.new_confirmed, color='red')
plt.title('Number of positive case for whole dataset by weekday', fontsize=15)
plt.xlabel('Days')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.show()


# Let's only look at it for the whole complete week for completeness

# In[ ]:


# Using a Boolean Mask
# Greater than the start date and smaller than the end date
mask = (df_time['date'] >= "2020-01-20") & (df_time['date'] <= "2020-03-01")

df_time_wholeweek = df_time.loc[mask]


# In[ ]:


plt.figure(figsize=(15,6))
plt.bar(df_time_wholeweek.day_of_week, df_time_wholeweek.new_confirmed, color='red')
plt.title('Number of positive case for complete week - 2020-01-20 to 2020-03-01', fontsize=15)
plt.xlabel('Days')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.show()


# This is super interesting, more positive cases seems to be in later during the week.
# 
# Even though new positive cases are arising everyday

# # Let's look at the patient data

# In[ ]:


df_patient.head(10)


# In[ ]:


print('There are a total of patients')
print(len(df_patient))
print('\n')
print('These are the rows that have been filled in')
print(df_patient.count())


# The most rich filled data points in the Patients Database
# * Country
# * Confirmed Date
# * State

# In[ ]:


df_patient.head()


# In[ ]:


df_patient.confirmed_date = pd.to_datetime(df_patient.confirmed_date)
df_patient.released_date = pd.to_datetime(df_patient.released_date)
df_patient.deceased_date = pd.to_datetime(df_patient.deceased_date)

daily_count = df_patient.groupby(df_patient.confirmed_date).id.count()
accumulated_count = daily_count.cumsum()


# Feature engineering to add a field for age and grouping them

# In[ ]:


df_patient['age'] = 2020 - df_patient['birth_year'] 


# In[ ]:


import math
def group_age(age):
    if age >= 0: # not NaN
        if age % 10 != 0:
            lower = int(math.floor(age / 10.0)) * 10
            upper = int(math.ceil(age / 10.0)) * 10 - 1
            return f"{lower}-{upper}"
        else:
            lower = int(age)
            upper = int(age + 9) 
            return f"{lower}-{upper}"
    return "Unknown"


df_patient["age_range"] = df_patient["age"].apply(group_age)


# In[ ]:


df_patient.dtypes


# In[ ]:


df_patient.head()


# In[ ]:


patient = df_patient


# Preprocessing

# In[ ]:


date_cols = ["confirmed_date", "released_date", "deceased_date"]
for col in date_cols:
    patient[col] = pd.to_datetime(patient[col])


# In[ ]:


patient["time_to_release_since_confirmed"] = patient["released_date"] - patient["confirmed_date"]

patient["time_to_death_since_confirmed"] = patient["deceased_date"] - patient["confirmed_date"]
patient["duration_since_confirmed"] = patient[["time_to_release_since_confirmed", "time_to_death_since_confirmed"]].min(axis=1)
patient["duration_days"] = patient["duration_since_confirmed"].dt.days
age_ranges = sorted(set([ar for ar in patient["age_range"] if ar != "Unknown"]))
patient["state_by_gender"] = patient["state"] + "_" + patient["sex"]


# In[ ]:


accumulated_count.plot()
plt.title('Accumulated Confirmed Count');


# Current State of Patient

# In[ ]:


infected_patients = patient.shape[0] #Total Patients
rp = patient.loc[patient["state"] == "released"].shape[0]
dp = patient.loc[patient["state"] == "deceased"].shape[0]
ip = patient.loc[patient["state"]== "isolated"].shape[0]
rp=rp/patient.shape[0]
dp=dp/patient.shape[0]
ip=ip/patient.shape[0]
print("The percentage of recovery is "+ str(round(rp*100,2)),"%")
print("The percentage of deceased is "+ str(round(dp*100,2)),"%")
print("The percentage of isolated is "+ str(round(ip*100,2)),"%")


# In[ ]:


states = pd.DataFrame(patient["state"].value_counts())
states["status"]=states.index
states.rename(columns={"state":"count"}, inplace = True)

fig = px.pie(states,
            values="count",
            names="status",
            title="Current state of patients",
            template="seaborn")
#Flexibility of labelling
fig.update_traces(rotation=90, pull=0.05, textinfo="value+percent+label")
fig.show()


# People who released

# In[ ]:


released = df_patient[df_patient.state == 'released']
released.head()


# People who are in isolated state

# In[ ]:


isolated_state = df_patient[df_patient.state == 'isolated']
isolated_state.head()


# Patient who died

# In[ ]:


dead = df_patient[df_patient.state == 'deceased']
dead.head()


# Age distribution of the released

# In[ ]:


plt.figure(figsize=(10,6))
sns.set_style("darkgrid")
plt.title("Age distribution of the released")
sns.kdeplot(data=released['age'], shade=True)


# In[ ]:


plt.figure(figsize=(10,6))
sns.set_style("darkgrid")
plt.title("Age distribution of the isolated")
sns.kdeplot(data=isolated_state['age'], shade=True)


# Age distribution of death

# In[ ]:


plt.figure(figsize=(10,6))
sns.set_style("darkgrid")
plt.title("Age distribution of the deceased")
sns.kdeplot(data=dead['age'], shade=True)


# Age distribution of death by gender

# In[ ]:


male_dead = dead[dead.sex=='male']
female_dead = dead[dead.sex=='female']

plt.figure(figsize=(10,6))
sns.set_style("darkgrid")
plt.title("Age distribution of the deceased by gender")
sns.kdeplot(data=female_dead['age'], label="Women", shade=True)
sns.kdeplot(data=male_dead['age'],label="Male" ,shade=True)


# In[ ]:


plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
sns.distplot(a=male_dead['age'], label="Men", kde=False)
sns.distplot(a=female_dead['age'], label="Women", kde=False)
plt.title("Age distribution of the deceased by sex")
plt.legend()


# Comparison of released and deceased by age

# In[ ]:


sns.kdeplot(data=dead['age'],label='deceased', shade=True)
sns.kdeplot(data=released['age'],label='released', shade=True)
sns.kdeplot(data=isolated_state['age'],label='released', shade=True)


# Death by gender

# In[ ]:


plt.figure(figsize=(15, 5))
plt.title('Sex')
dead.sex.value_counts().plot.bar();


# Reason for the infection

# In[ ]:


plt.figure(figsize=(15,5))
plt.title('Infection reason')
df_patient.infection_reason.value_counts().plot.bar();


# Majority are contacted with patients and vists to Daegu

# In[ ]:


plt.figure(figsize=(15,5))
plt.title('Groups')
df_patient.group.value_counts().plot.bar();


# State of Patient

# In[ ]:


sns.set(rc={'figure.figsize':(5,5)})
sns.countplot(x=df_patient['state'].loc[
    (df_patient['infection_reason']=='contact with patient')
])


# In[ ]:


age_gender_hue_order =["deceased_female",
                       "deceased_male"]
custom_palette = sns.color_palette("Reds")[3:6] + sns.color_palette("Blues")[2:5]

plt.figure(figsize=(12, 8))
sns.countplot(x = "age_range",
              hue="state_by_gender",
              order=age_ranges,
              hue_order=age_gender_hue_order,
              palette=custom_palette,
              data=patient)
plt.title("State by gender and age", fontsize=16)
plt.xlabel("Age range", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc="upper right")
plt.show()


# We can see most death cases are the elderly

# # Looking route data

# In[ ]:


df_route.head()


# In[ ]:


df_route.isna().sum()


# In[ ]:


clus=df_route.loc[:,['id','latitude','longitude']]
clus.head(10)


# In[ ]:


Y_axis = df_route[['latitude']]
X_axis = df_route[['longitude']]

# Within-Cluster-Sum-of-Squares

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X_axis, Y_axis)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# We choose the number of clusters based on the elbow method.

# In[ ]:


kmeans = KMeans(n_clusters = 4, init ='k-means++')
kmeans.fit(clus[clus.columns[1:3]])
clus['cluster_label'] = kmeans.fit_predict(clus[clus.columns[1:3]])
centers = kmeans.cluster_centers_
labels = kmeans.predict(clus[clus.columns[1:3]])


# Graphical representation of clusters

# In[ ]:


clus.plot.scatter(x = 'latitude', y = 'longitude', c=labels, s=30, cmap='coolwarm')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)


# We will verify our clusters by putting values in world map by making use of folium library

# Affected place in world map

# In[ ]:


import folium
southkorea_map = folium.Map(location=[36.55,126.983333 ], zoom_start=7,tiles='Stamen Toner')

for lat, lon,city in zip(df_route['latitude'], df_route['longitude'],df_route['city']):
    folium.CircleMarker([lat, lon],
                        radius=5,
                        color='red',
                      popup =('City: ' + str(city) + '<br>'),
                        fill_color='red',
                        fill_opacity=0.7 ).add_to(southkorea_map)
southkorea_map


# Patient in city
# 

# In[ ]:


plt.figure(figsize=(15,5))
plt.title('Number patients in city')
df_route.province.value_counts().plot.bar();


# Patients in Provience/State

# In[ ]:


plt.figure(figsize=(15,5))
plt.title('Number patients in province')
df_route.province.value_counts().plot.bar();


# Time from confirmation to release or death

# In[ ]:


plt.figure(figsize=(12,8))
sns.boxplot(x="state",
           y="duration_days",
           order=["released", "deceased"],
           data=patient)
plt.title("Time from confirmation to release or death", fontsize=16)
plt.xlabel("State", fontsize=16)
plt.ylabel("Days", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# Making data ready for prediction
# 
# 

# In[ ]:


data = daily_count.resample('D').first().fillna(0).cumsum() # cumulative

data = data[20:]
x = np.arange(len(data)).reshape(-1, 1)
y = data.values


# Regression Model

# In[ ]:


from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=[32, 32, 10], max_iter=50000, alpha=0.0005, random_state=26)
_=model.fit(x, y)


# In[ ]:


test = np.arange(len(data)+7).reshape(-1, 1)
pred = model.predict(test)
prediction = pred.round().astype(int)
week = [data.index[0] + timedelta(days=i) for i in range(len(prediction))]
dt_idx = pd.DatetimeIndex(week)
predicted_count = pd.Series(prediction, dt_idx)


# Graphical representatoin of current confirmed and predicted confirmed

# In[ ]:


accumulated_count.plot()
predicted_count.plot()
plt.title('Prediction of Accumulated Confirmed Count')
plt.legend(['current confirmd count', 'predicted confirmed count'])
plt.show()


# Autoregressive integrated moving average(Arima)

# In[ ]:


confirm_cs = pd.DataFrame(data).cumsum()
arima_data = confirm_cs.reset_index()
arima_data.columns = ['confirmed_date','count']
arima_data.head()


# In[ ]:


model = ARIMA(arima_data['count'].values, order=(1, 2, 1))
fit_model = model.fit(trend='c', full_output=True, disp=True)
fit_model.summary()


# In[ ]:


fit_model.plot_predict()
plt.title('Forecast vs Actual')
pd.DataFrame(fit_model.resid).plot()


# # Look at the trend dataset!

# In[ ]:


df_trend


# **Live updates counts:**
# https://www.worldometers.info/coronavirus/
# 
# **Visualisation from World Health Organisation:**
# https://experience.arcgis.com/experience/685d0ace521648f8a5beeeee1b9125cd
# 
# **For refrence**
# https://www.kaggle.com/vanshjatana/analysis-on-coronavirus
