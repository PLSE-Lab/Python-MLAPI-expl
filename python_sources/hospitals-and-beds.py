#!/usr/bin/env python
# coding: utf-8

# **Libraries**

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import datetime
from datetime import date, timedelta
from sklearn.cluster import KMeans
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
import plotly.offline as py
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# In[ ]:


hospital  = pd.read_csv("../input/hospitalbedloc/HospitalBedsIndiaLocations.csv")


#  **Hospitals in India**

# In[ ]:


hospital.head()


# **This file contain Health Management Information System data and National Health Profile Data and we deal diffrently with them**

# In[ ]:


hmis = hospital.loc[:,['Sno','State/UT','Latitude','Longitude','NumPrimaryHealthCenters_HMIS','NumCommunityHealthCenters_HMIS','NumSubDistrictHospitals_HMIS','NumDistrictHospitals_HMIS','TotalPublicHealthFacilities_HMIS','NumPublicBeds_HMIS']]


# In[ ]:


nhp = hospital.loc[:,['Sno','State/UT','Latitude','Longitude','NumRuralHospitals_NHP18','NumRuralBeds_NHP18','NumUrbanHospitals_NHP18','NumUrbanBeds_NHP18']]


# In[ ]:


hmis = hmis.rename(columns={"State/UT":"State","NumPrimaryHealthCenters_HMIS":"Primary","NumCommunityHealthCenters_HMIS":"Community","NumSubDistrictHospitals_HMIS":"SubDistrict","NumDistrictHospitals_HMIS":"District","TotalPublicHealthFacilities_HMIS":"Total","NumPublicBeds_HMIS":"Public beds"})


# In[ ]:


nhp = nhp.rename(columns={'State/UT':"State",'NumRuralHospitals_NHP18':"Rural_hospitals",'NumRuralBeds_NHP18':"Rural_beds",'NumUrbanHospitals_NHP18':"Urban_hospitals",'NumUrbanBeds_NHP18':"Urban_beds"})


# In[ ]:


hmis = hmis.fillna(0)


# In[ ]:


hmis.head()


# **Tree plots by Total Hospitals**

# In[ ]:


fig = px.treemap(hmis, path=['State'], values='Total',
                  color='Total', hover_data=['State','Primary','Community','SubDistrict','District',"Public beds"],
                  color_continuous_scale='burgyl')
fig.show()


# **Tree plot by Beds**

# In[ ]:


fig = px.treemap(hmis, path=['State'], values='Public beds',
                  color='Public beds', hover_data=['State','Primary','Community','SubDistrict','District',"Total"],
                  color_continuous_scale='burgyl')
fig.show()


# **HMIS Hospitals in World Map**

# In[ ]:


import folium
india = folium.Map(location=[20.5937,78.9629 ], zoom_start=5,tiles='cartodbpositron')

for lat, lon,State,Primary ,Community, SubDistrict, District, Total in zip(hmis['Latitude'], hmis['Longitude'],hmis['State'],hmis['Total'],hmis['Primary'],hmis['Community'],hmis['SubDistrict'],hmis['District']):
    folium.CircleMarker([lat, lon],
                        radius=5,
                        color='red',
                      popup =('State:' + str(State) + '<br>'
                             'Total Hospitals:' + str(Total) + '<br>'
                              'Primary :' + str(Primary) + '<br>'
                             'Community:' + str(Community) + '<br>'
                             'Sub District:'+ str(SubDistrict) + '<br>'
                             'District:'+ str(District) + '<br>'

                             ),
                        
                        fill_color='red',
                        fill_opacity=0.7 ).add_to(india)
india


# In[ ]:


hmis = hmis.sort_values(by='Total', ascending=False)


# In[ ]:


fig = px.bar(hmis[['State', 'Total']].sort_values('Total', ascending=False), 
             y="Total", x="State", color='State', 
             log_y=True, template='ggplot2', title='Total Hospitals')
fig.show()


# In[ ]:


fig = px.bar(hmis[['State', 'Primary']].sort_values('Primary', ascending=False), 
             y="Primary", x="State", color='State', 
             log_y=True, template='ggplot2', title='Primary Hospitals')
fig.show()


# In[ ]:


fig = px.bar(hmis[['State', 'Community']].sort_values('Community', ascending=False), 
             y="Community", x="State", color='State', 
             log_y=True, template='ggplot2', title='Community Hospitals')
fig.show()


# In[ ]:


fig = px.bar(hmis[['State', 'District']].sort_values('District', ascending=False), 
             y="District", x="State", color='State', 
             log_y=True, template='ggplot2', title='District Hospitals')
fig.show()


# In[ ]:


fig = px.bar(hmis[['State', 'SubDistrict']].sort_values('SubDistrict', ascending=False), 
             y="SubDistrict", x="State", color='State', 
             log_y=True, template='ggplot2', title='SubDistrict Hospitals')
fig.show()


# In[ ]:


nhp = nhp.fillna(0)
nhp['Total_hospitals'] = nhp['Rural_hospitals'] + nhp['Urban_hospitals']
nhp['Total_beds'] = nhp['Rural_beds'] + nhp['Urban_beds']


# In[ ]:


nhp.head()


# # Total Beds

# In[ ]:


ms = nhp.sort_values(by=['Total_beds'],ascending=False)
ms = ms.head(30)
fig = px.funnel(ms, x='Total_beds', y='State')
fig.show()


# **Tree plot by Total Beds**

# In[ ]:


fig = px.treemap(nhp, path=['State'], values='Total_beds',
                  color='Total_beds', hover_data=['State','Rural_hospitals','Rural_beds','Urban_hospitals','Urban_beds','Total_hospitals'],
                  color_continuous_scale='burgyl')
fig.show()


# **Tree plot by Total Hospitals**

# In[ ]:


fig = px.treemap(nhp, path=['State'], values='Total_hospitals',
                  color='Total_hospitals', hover_data=['State','Rural_hospitals','Rural_beds','Urban_hospitals','Urban_beds','Total_beds'],
                  color_continuous_scale='burgyl')
fig.show()


# In[ ]:


fig = px.bar(nhp[['State', 'Total_hospitals']].sort_values('Total_hospitals', ascending=False), 
             y="Total_hospitals", x="State", color='State', 
             log_y=True, template='ggplot2', title='Total Hospitals')
fig.show()


# In[ ]:


fig = px.bar(nhp[['State', 'Rural_hospitals']].sort_values('Rural_hospitals', ascending=False), 
             y="Rural_hospitals", x="State", color='State', 
             log_y=True, template='ggplot2', title='Rural Hospitals')
fig.show()


# In[ ]:


fig = px.bar(nhp[['State', 'Rural_beds']].sort_values('Rural_beds', ascending=False), 
             y="Rural_beds", x="State", color='State', 
             log_y=True, template='ggplot2', title='Rural Beds')
fig.show()


# In[ ]:


fig = px.bar(nhp[['State', 'Urban_hospitals']].sort_values('Urban_hospitals', ascending=False), 
             y="Urban_hospitals", x="State", color='State', 
             log_y=True, template='ggplot2', title='Urban Hospitals')
fig.show()


# In[ ]:


fig = px.bar(nhp[['State', 'Urban_beds']].sort_values('Urban_beds', ascending=False), 
             y="Urban_beds", x="State", color='State', 
             log_y=True, template='ggplot2', title='Urban beds')
fig.show()


# In[ ]:


fig = px.bar(nhp[['State', 'Total_hospitals']].sort_values('Total_hospitals', ascending=False), 
             y="Total_hospitals", x="State", color='State', 
             log_y=True, template='ggplot2', title='Total Hospitals')
fig.show()


# In[ ]:


fig = px.bar(nhp[['State', 'Total_beds']].sort_values('Total_beds', ascending=False), 
             y="Total_beds", x="State", color='State', 
             log_y=True, template='ggplot2', title='Total Beds')
fig.show()


# In[ ]:


usa = pd.read_csv("../input/usa-hospitals/Hospitals.csv")


# In[ ]:


usa.head()


# In[ ]:


usa.info()


# **Total Hospitals in USA**

# In[ ]:


ushospital= usa.loc[:,["ID","NAME","ADDRESS","CITY","STATE","TELEPHONE","LATITUDE","LONGITUDE","TYPE"]]


# In[ ]:


ushospital.head()


# In[ ]:


plt.figure(figsize=(15,5))
plt.title('Type of Hospitals in US')
ushospital.TYPE.value_counts().plot.bar();


# In[ ]:


fig = px.pie( values=ushospital.groupby(['TYPE']).size().values,names=ushospital.groupby(['TYPE']).size().index)
fig.update_layout(
    font=dict(
        size=15,
        color="#242323"
    )
    )   
    
py.iplot(fig)


# In[ ]:


plt.figure(figsize=(15,5))
plt.title('Number of  Hospitals in each State')
ushospital.STATE.value_counts().plot.bar();


# In[ ]:


fig = px.pie( values=ushospital.groupby(['STATE']).size().values,names=ushospital.groupby(['STATE']).size().index)
fig.update_layout(
    font=dict(
        size=15,
        color="#242323"
    )
    )   
    
py.iplot(fig)


# **Specific Type of Hospitals**

# In[ ]:


spec = usa.groupby(['TYPE'])
spec


# In[ ]:


#type_of_hospital = spec.get_group("type of hospital")
type_of_hospital = spec.get_group("MILITARY")


# In[ ]:


type_of_hospital = type_of_hospital.loc[:,["ID","NAME","ADDRESS","CITY","STATE","TELEPHONE","LATITUDE","LONGITUDE","BEDS"]]


# In[ ]:


type_of_hospital.head()


# In[ ]:


plt.figure(figsize=(15,5))
plt.title('Number of  Hospitals in each State')
type_of_hospital.STATE.value_counts().plot.bar();


# In[ ]:


fig = px.pie( values=type_of_hospital.groupby(['STATE']).size().values,names=type_of_hospital.groupby(['STATE']).size().index)
fig.update_layout(
    font=dict(
        size=15,
        color="#242323"
    )
    )   
    
py.iplot(fig)


# In[ ]:


import folium
usa_map = folium.Map(location=[39.0902,-97.922211 ], zoom_start=4,tiles='cartodbpositron')

for LATITUDE, LONGITUDE,ADDRESS,NAME , TELEPHONE in zip(type_of_hospital['LATITUDE'], type_of_hospital['LONGITUDE'],type_of_hospital['NAME'],type_of_hospital['ADDRESS'],type_of_hospital['TELEPHONE']):
    folium.CircleMarker([LATITUDE, LONGITUDE],
                        radius=3,
                        color='red',
                        
                        popup =('Name:' + str(NAME) + '<br>'
                             'Address:' + str(ADDRESS) + '<br>'
                             'Telephone:' + str(TELEPHONE) + '<br>'
                             ),
                      
                        fill_color='red',
                        fill_opacity=0.7 ).add_to(usa_map)
usa_map


# In[ ]:


fig = px.choropleth(type_of_hospital, locations=type_of_hospital["STATE"],       

 color=type_of_hospital["BEDS"],
                    locationmode="USA-states",
                    scope="usa",
                    color_continuous_scale='Reds',

                   )

fig.show()

