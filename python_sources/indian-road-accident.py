#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import re


# In[ ]:


data_1970_2017 = pd.read_csv("/kaggle/input/indian-road-accident-statistics/RoadAccidentsPersonsKilledandInjuredfrom1970to2017.csv")


# In[ ]:


data_1970_2017.head()


# In[ ]:


fig = px.line(data_1970_2017, x="Years", y="Total Number of Persons Killed (in numbers)")
fig.show()


# In[ ]:


fig = px.area(data_1970_2017, x="Years", y="Total Number of Persons Killed (in numbers)", color="Years",line_group="Years")
fig.show()


# In[ ]:


fig = px.line(data_1970_2017, x="Years", y="Population of India (in thousands)")
fig.show()


# In[ ]:


# fig = px.line(data_1970_2017, x="Years", y="Road Length (in kms)")
# fig.show()
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=data_1970_2017["Years"], y=data_1970_2017["Population of India (in thousands)"]))
fig.show()


# In[ ]:


data_1970_2017.corr().nlargest(1,'Total Number of Persons Killed (in numbers)')


# In[ ]:


data_1970_2017.columns


# In[ ]:


data_correlation = data_1970_2017.iloc[:,[2,4,5,6]]
# data_1970_2017.corr()
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(data_correlation.corr(), annot = True, linewidths=.5,linecolor='red', fmt='.1f', ax=ax)
plt.show()


# In[ ]:


data_state_2017 = pd.read_csv("/kaggle/input/indian-road-accident-statistics/StateUT-wise Type of Road Accidents in 2017.csv")


# In[ ]:


data_1970_2017.shape


# In[ ]:


Top_fatal_accident = data_state_2017.iloc[[0,11,12,13,14,23,27,33]]


# In[ ]:


fig = go.Figure(data=[
    go.Bar(name='Total Accidents', x=Top_fatal_accident['States/UTs'], y=Top_fatal_accident['Total Accidents'], text=Top_fatal_accident['Total Accidents']),
    go.Bar(name='Fatal Accidents', x=Top_fatal_accident['States/UTs'], y=Top_fatal_accident['Fatal Accidents'], text=Top_fatal_accident['Fatal Accidents'])
    
])
# Change the bar mode
fig.update_layout(barmode='stack')
fig.show()


# In[ ]:


Top_fatal_states = data_state_2017.nlargest(10,'Fatal Accidents').iloc[1:]
Top_accidents = data_state_2017.nlargest(10,'Total Accidents').iloc[1:]


# In[ ]:


fig = px.bar(Top_fatal_states, x='States/UTs', y='Fatal Accidents', text='Fatal Accidents', color='States/UTs',  width=800, height=400)
# fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
# fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()


# In[ ]:


fig = px.bar(Top_accidents, x='States/UTs', y='Total Accidents', text='Total Accidents', color='States/UTs',  width=800, height=400)
fig.show()


# In[ ]:


data_occurance_actual = pd.read_csv("/kaggle/input/indian-road-accident-statistics/State UT-wise Road Accidents as per the Time of occurrence during 2014 and 2016.csv")


# In[ ]:


data_occurance = data_occurance_actual.iloc[[0,11,12,13,14,23,27,33],[1,11,12,13,14,15,16,17,18]]


# In[ ]:


data_occurance


# In[ ]:


data_occurance = data_occurance.rename(columns=lambda x: x.replace(' - 2016','')) 


# In[ ]:


data_occurance = data_occurance.T


# In[ ]:


data_occurance = data_occurance.rename(columns={0:'Andhra Pradesh', 11:'Karnataka', 12:'Kerala', 13:'Madhya Pradesh', 14:'Maharashtra', 23:'Tamil Nadu', 27:'Uttar Pradesh', 33:'Delhi'})


# In[ ]:


data_occurance = data_occurance.drop(['States/Uts'], axis=0)


# In[ ]:


data_occurance


# In[ ]:


data_occurance['Time'] = data_occurance.index


# In[ ]:


data_occurance.reset_index(inplace = True) 
data_occurance =data_occurance.drop(['index'], axis=1)


# In[ ]:


data_occurance


# In[ ]:


data_occurance = pd.melt(data_occurance,id_vars=['Time'],var_name='States', value_name='Number of person killed')


# In[ ]:


data_occurance


# In[ ]:


fig = px.line(data_occurance, x='Time', y="Number of person killed", color = 'States')
fig.show()


# In[ ]:


data_helmet_actual = pd.read_csv("/kaggle/input/indian-road-accident-statistics/StateUT-wise Accidents Victims Classified according to Non-Use of Safety Device (Non Wearing of Hel.csv")


# In[ ]:


data_helmet = data_helmet_actual.iloc[[0,11,12,13,14,23,27,33]]


# In[ ]:


data_helmet


# In[ ]:


fig = px.bar(data_helmet, x='States/UTs', y='Persons Killed', text='Persons Killed', color='States/UTs', title= "Person killed without using helmet",  width=800, height=400)
fig.show()


# In[ ]:


fig = px.pie(data_helmet, names='States/UTs', values='Persons Killed', title= "Person killed without using helmet")
fig.show()


# In[ ]:


data_driver_responsibility_actual = pd.read_csv("/kaggle/input/indian-road-accident-statistics/StateUT-wise Accidents Classified according to Responsibilities of Driver during 2014 and 2016.csv")


# In[ ]:


data_driver_responsibility = data_driver_responsibility_actual.iloc[[0,11,12,13,14,23,27,33],[1,69,72,75,78,81,84,87,90]]


# In[ ]:


data_driver_responsibility = data_driver_responsibility.rename(columns=lambda x: re.sub(r' \-.*', "",x))


# In[ ]:


data_driver_responsibility = data_driver_responsibility.T


# In[ ]:


data_driver_responsibility = data_driver_responsibility.rename(columns={0:'Andhra Pradesh', 11:'Karnataka', 12:'Kerala', 13:'Madhya Pradesh', 14:'Maharashtra', 23:'Tamil Nadu', 27:'Uttar Pradesh', 33:'Delhi'})


# In[ ]:


data_driver_responsibility


# In[ ]:


data_driver_responsibility = data_driver_responsibility.drop(['State/ UT'], axis=0)
data_driver_responsibility['Irresponsibility'] = data_driver_responsibility.index
data_driver_responsibility.reset_index(inplace = True)
data_driver_responsibility =data_driver_responsibility.drop(['index'], axis=1)
data_driver_responsibility = pd.melt(data_driver_responsibility,id_vars=['Irresponsibility'],var_name='States', value_name='Person killed')


# In[ ]:


data_driver_responsibility


# In[ ]:


# fig = px.line(data_driver_responsibility, x='Irresponsibility', y="Person killed", color = 'States',  width=1600, height=800)
fig = px.line(data_driver_responsibility, x='Irresponsibility', y="Person killed", color = 'States')
fig.show()


# In[ ]:


data_weather_actual = pd.read_csv("/kaggle/input/indian-road-accident-statistics/StateUT-wise Accidents Classified according to Type of Weather Condition during 2014 and 2016.csv")


# In[ ]:


data_weather = data_weather_actual.iloc[[0,11,12,13,14,23,27],[1,6,9,15,18,21,24,27,30,33,36]]
data_weather = data_weather.rename(columns=lambda x: re.sub(r'\-.*','',x))
data_weather = data_weather.T
data_weather = data_weather.rename(columns={0:'Andhra Pradesh', 11:'Karnataka', 12:'Kerala', 13:'Madhya Pradesh', 14:'Maharashtra', 23:'Tamil Nadu', 27:'Uttar Pradesh'})


# In[ ]:


data_weather = data_weather.drop(['State/ UT'], axis=0)
data_weather['Weather condition'] = data_weather.index
data_weather.reset_index(inplace = True)
data_weather =data_weather.drop(['index'], axis=1)


# In[ ]:


data_weather


# In[ ]:


data_weather = pd.melt(data_weather,id_vars=['Weather condition'],var_name='States', value_name='Person killed')
data_weather


# In[ ]:


fig = px.line(data_weather, x='Weather condition', y="Person killed", color = 'States')
fig.show()


# In[ ]:


data_vehicle_defect_actual = pd.read_csv("/kaggle/input/vehicle-defect/StateUT-wise Accidents classified according to Vehicular Defect during 2012 and 2016.csv")


# In[ ]:


data_vehicle_defect_actual.head()


# In[ ]:


data_vehicle_defect = data_vehicle_defect_actual.iloc[[0,11,12,13,14,23,27,33],[1,18,21,24,27]]
data_vehicle_defect = data_vehicle_defect.rename(columns=lambda x: re.sub(r' \-.*', "",x))
data_vehicle_defect


# In[ ]:


data_vehicle_defect = data_vehicle_defect.T
data_vehicle_defect = data_vehicle_defect.rename(columns={0:'Andhra Pradesh', 11:'Karnataka', 12:'Kerala', 13:'Madhya Pradesh', 14:'Maharashtra', 23:'Tamil Nadu', 27:'Uttar Pradesh', 33:'Delhi'})
data_vehicle_defect = data_vehicle_defect.drop(['State/ UT'], axis=0)
data_vehicle_defect['Vehicle defect type'] = data_vehicle_defect.index
data_vehicle_defect.reset_index(inplace = True)
data_vehicle_defect =data_vehicle_defect.drop(['index'], axis=1)


# In[ ]:


data_vehicle_defect = pd.melt(data_vehicle_defect,id_vars=['Vehicle defect type'],var_name='States', value_name='Person killed')


# In[ ]:


data_vehicle_defect


# In[ ]:


fig = px.line(data_vehicle_defect, x='Vehicle defect type', y="Person killed", color = 'States')
fig.show()


# In[ ]:


data_location_actual = pd.read_csv("/kaggle/input/indian-road-accident-statistics/State UT-wise Accidents classified according to Type of Location during 2014 and 2016.csv")


# In[ ]:


data_location = data_location_actual.iloc[[0,11,12,13,14,23,27,33],[1,48,51,54,60,63,72,75,78,81]]


# In[ ]:


data_location = data_location.rename(columns=lambda x: re.sub(r' \-.*', "",x))
data_location = data_location.T
data_location = data_location.rename(columns={0:'Andhra Pradesh', 11:'Karnataka', 12:'Kerala', 13:'Madhya Pradesh', 14:'Maharashtra', 23:'Tamil Nadu', 27:'Uttar Pradesh', 33:'Delhi'})
data_location = data_location.drop(['State/ UT'], axis=0)
data_location['Location'] = data_location.index
data_location.reset_index(inplace = True)
data_location =data_location.drop(['index'], axis=1)
data_location = pd.melt(data_location,id_vars=['Location'],var_name='States', value_name='Person killed')


# In[ ]:


fig = px.line(data_location, x='Location', y="Person killed", color = 'States')
fig.show()


# In[ ]:


data_per_thousand_actual = pd.read_csv("/kaggle/input/indianroadaccidentstatistics/StateUT-wise Severity of Road Accidents in India from 2014 to 2017.csv")


# In[ ]:


data_per_thousand = data_per_thousand_actual.iloc[[0,11,12,13,14,23,27,33],:]
data_per_thousand = data_per_thousand.rename(columns=lambda x: re.sub(r'.*\- ', "",x))
data_per_thousand = data_per_thousand.T


# In[ ]:


data_per_thousand = data_per_thousand.rename(columns={0:'Andhra Pradesh', 11:'Karnataka', 12:'Kerala', 13:'Madhya Pradesh', 14:'Maharashtra', 23:'Tamil Nadu', 27:'Uttar Pradesh', 33:'Delhi'})
data_per_thousand = data_per_thousand.drop(['States/UTs'], axis=0)
data_per_thousand['Year'] = data_per_thousand.index
data_per_thousand.reset_index(inplace = True)
data_per_thousand =data_per_thousand.drop(['index'], axis=1)
data_per_thousand = pd.melt(data_per_thousand,id_vars=['Year'],var_name='States', value_name='Person killed')


# In[ ]:


fig = px.line(data_per_thousand, x='Year', y="Person killed", color = 'States')
fig.show()


# Merging features into a single dataset

# In[ ]:


Accident_dataset = data_state_2017.iloc[:,[0,1]]
Accident_dataset = pd.merge(Accident_dataset, data_helmet_actual.iloc[:,[0,1]])
Accident_dataset = Accident_dataset.rename(columns={'Persons Killed':'Without using helmet'})
dummy = data_driver_responsibility_actual.iloc[:,[1,69,72,75,78,81,84,87,90]]


# In[ ]:


dummy = dummy.rename(columns={'State/ UT':'States/UTs'})
Accident_dataset = pd.merge(Accident_dataset, dummy)
dummy = data_location_actual.iloc[:,[1,48,51,54,60,63,72,75,78,81]]
dummy = dummy.rename(columns={'State/ UT':'States/UTs'})


# In[ ]:


Accident_dataset = pd.merge(Accident_dataset, dummy)
dummy = data_occurance_actual.iloc[:,[1,11,12,13,14,15,16,17,18]]
dummy = dummy.rename(columns={'States/Uts':'States/UTs'})
Accident_dataset = pd.merge(Accident_dataset, dummy)


# In[ ]:


dummy = data_vehicle_defect_actual.iloc[:,[1,18,21,24,27]]
dummy = dummy.rename(columns={'State/ UT':'States/UTs'})
Accident_dataset = pd.merge(Accident_dataset, dummy)


# In[ ]:


dummy = data_weather_actual.iloc[:,[1,6,9,15,18,21,24,27,30,33,36]]
dummy = dummy.rename(columns={'State/ UT':'States/UTs'})
Accident_dataset = pd.merge(Accident_dataset, dummy)


# In[ ]:


Accident_dataset = Accident_dataset.rename(columns=lambda x: re.sub(r' \-.*', "",x))
Accident_dataset.columns


# **Exploratory data analysis**

# In[ ]:


Accident_dataset_value = Accident_dataset
Accident_total_count = Accident_dataset.iloc[36,1:]


# In[ ]:


Accident_dataset_value[Accident_dataset_value.isnull().any(axis=1)]


# In[ ]:


Accident_dataset_value = Accident_dataset_value.drop([36])
Accident_dataset_value = Accident_dataset_value.fillna(Accident_dataset_value.mean())
Accident_dataset_value = Accident_dataset_value.set_index('States/UTs')
Accident_dataset_percentage = Accident_dataset_value/Accident_dataset_value[Accident_dataset_value.columns].sum()*100


# In[ ]:


Accident_total_count = pd.DataFrame(Accident_total_count).T
Accident_total_count.rename(index={36:'Total person killed'},inplace=True)
Accident_dataset_percentage = pd.concat([Accident_dataset_percentage, Accident_total_count])
Accident_dataset_percentage = Accident_dataset_percentage.append(pd.Series(name='Threshold percentage'))
# Accident_dataset_percentage.append(pd.Series(), ignore_index=True)
# Accident_dataset_percentage.iloc[37,:] = Accident_dataset_percentage.iloc[36,:]/36
# Accident_dataset_percentage


# In[ ]:


Accident_dataset_percentage = Accident_dataset_percentage.iloc[:,:].astype('float')
Accident_dataset_percentage


# In[ ]:


column_dict = {0: "Fatal Accidents", 1: "Without using helmet", 2: "Intake of Alcohal", 3: "Exceeding lawful speed", 4: "Jumping Red Light", 5: "Driving on wrong side", 6: "Jumping/Changing lanes", 7: "Overtaking", 8: "Using Mobile Phones while driving", 9: "Asleep or Fatigued or Sick", 10: "Near School/College/any other educational Institutes", 11: "Pedestrian crossing", 12: "Market Place", 13: "Near a Religious Place", 14: "Near Hospital", 15: "Narrow bridge or culverts", 16: "Near Petrol Pump", 17: "Near Bus Stand", 18: "Near or on Road under Construction", 19: "06-900hrs", 20: "09-1200hrs", 21: "12-1500hrs", 22: "15-1800hrs", 23: "18-2100hrs", 24: "21-2400hrs", 25: "00-300hrs", 26: "03-600hrs", 27: "Defective brakes", 28: "Defective Steering/Axil", 29: "Punctured/burst Typres", 30: "Worn out Tyres", 31: "Mist/fog", 32: "Cloudy", 33: "Heavy rain", 34: "Flooding of slipways/rivulers", 35: "Hail/sleet", 36: "snow", 37: "Strong wind", 38: "Dust storm", 39: "Very hot", 40: "Very cold"}


# In[ ]:


for data in column_dict.keys():
    a = np.sum(Accident_dataset_percentage.iloc[0:36,[data]].nlargest(5,column_dict[data]))/5
    b = np.sum(Accident_dataset_percentage.iloc[0:36,[data]].nlargest(8,column_dict[data]))/8
    c = np.sum(Accident_dataset_percentage.iloc[0:36,[data]].nlargest(10,column_dict[data]))/10
    print(a[0],b[0],c[0])


# In[ ]:


for data in column_dict.keys():
    Accident_dataset_percentage.iloc[37,[data]] = np.sum(Accident_dataset_percentage.iloc[0:36,[data]].nlargest(8,column_dict[data]))/8


# In[ ]:


Accident_dataset_percentage.index.names=[None]
Accident_dataset_percentage = Accident_dataset_percentage.reset_index()
Accident_dataset_percentage = Accident_dataset_percentage.rename(columns={'index': 'States/UTs'})
Accident_dataset_percentage


# In[ ]:


Accident_dataset_percentage.nlargest(15,'Fatal Accidents')[2:]


# In[ ]:


Accident_dataset_percentage.columns


# In[ ]:


# Accident_dataset_percentage.to_csv("Road_accident_dataset.csv")


# In[ ]:


Accident_dataset_percentage.to_csv("Road_accident_no_index_dataset.csv", encoding='utf-8', index=False)


# In[ ]:




