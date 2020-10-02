#!/usr/bin/env python
# coding: utf-8

# # Import

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime
import seaborn as sns
import networkx as nx


# # Datasets

# In[ ]:


path = '/kaggle/input/coronavirusdataset/'
patient_data_path = path + 'PatientInfo.csv'
route_data_path = path + 'PatientRoute.csv'
time_data_path = path + 'Time.csv'

df_patient = pd.read_csv(patient_data_path)
df_route = pd.read_csv(route_data_path)
df_time = pd.read_csv(time_data_path)


# ## Patient
# 
# **Columns**
# 
# 1. **patient_id** the ID of the patient
# 2. **global_num** the number given by KCDC
# 3. **sex** the sex of the patient
# 4. **birth_year** the birth year of the patient
# 5. **age** the age of the patient
# 6. **country** the country of the patient
# 7. **province** the province of the patient
# 8. **city** the city of the patient
# 9. **disease** TRUE: underlying disease / FALSE: no disease
# 10. **infection_case** the case of infection
# 11. **infection_order** the order of infection
# 12. **infected_by** the ID of who infected the patient
# 13. **contact_number** the number of contacts with people
# 14. **symptom_onset_date** the date of symptom onset
# 15. **confirmed_date** the date of being confirmed
# 16. **released_date** the date of being released
# 17. **deceased_date** the date of being deceased
# 18. **state** isolated / released / deceased

# In[ ]:


df_patient.head()


# In[ ]:


df_patient.info()


# In[ ]:


df_patient.isna().sum()


# In[ ]:


df_patient.confirmed_date = pd.to_datetime(df_patient.confirmed_date)
df_patient.released_date = pd.to_datetime(df_patient.released_date)
df_patient.deceased_date = pd.to_datetime(df_patient.deceased_date)


# In[ ]:


df_patient['time_from_confirmed_to_death'] = df_patient.deceased_date - df_patient.confirmed_date
df_patient['time_from_released_to_death'] = df_patient.released_date - df_patient.confirmed_date
df_patient['age'] = datetime.now().year - df_patient.birth_year 


# In[ ]:


patient_deceased = df_patient[df_patient.state == 'deceased']
patient_isolated = df_patient[df_patient.state == 'isolated']
patient_released = df_patient[df_patient.state == 'released']


# ### Sex

# In[ ]:


f, ax = plt.subplots(figsize=(15, 5))
sns.countplot(y="sex", data=df_patient, color="c");


# ### Age distribution of the deceased by gender

# In[ ]:


male_dead = patient_deceased[patient_deceased.sex=='male']
female_dead = patient_deceased[patient_deceased.sex=='female']
plt.figure(figsize=(15,5))
plt.title("Age distribution of the deceased by gender")
sns.kdeplot(data=female_dead['age'], shade=True);
sns.kdeplot(data=male_dead['age'], shade=True);


# ### Birth year

# In[ ]:


f, ax = plt.subplots(figsize=(15, 5))
sns.distplot(df_patient.birth_year, color='c');


# ### Age

# In[ ]:


f, ax = plt.subplots(figsize=(15, 5))
sns.distplot(df_patient.age, color='c');


# ### Country

# In[ ]:


df_patient.country.value_counts()


# ### Infection reason

# In[ ]:


f, ax = plt.subplots(figsize=(15, 10))
sns.countplot(y="infection_case", data=df_patient, color="c");


# ### Region

# In[ ]:


f, ax = plt.subplots(figsize=(15, 10))
sns.countplot(y="province", data=df_patient, color="c");


# ### State

# In[ ]:


df_patient.state.value_counts()


# ### State / Age

# In[ ]:


f, ax = plt.subplots(figsize=(15, 5))
sns.distplot(patient_deceased.age, color='c');


# In[ ]:


f, ax = plt.subplots(figsize=(15, 5))
sns.distplot(patient_isolated.age, color='c');


# In[ ]:


f, ax = plt.subplots(figsize=(15, 5))
sns.distplot(patient_released.age, color='c');


# ### Daily confirmations

# In[ ]:


plt.figure(figsize=(15,5))
plt.title('Daily confirmations')
df_patient.groupby('confirmed_date').patient_id.count().plot();


# ### Confirmed count

# In[ ]:


plt.figure(figsize=(15,5))
plt.title('Confirmed count')
df_patient.groupby('confirmed_date').patient_id.count().cumsum().plot();


# ### Infection networks

# In[ ]:


data_infected_by = df_patient[df_patient.infected_by.notnull()]

def get_sex_for_patient_id(id):
    result = df_patient[df_patient.patient_id == id].sex.values
    return result[0] if len(result) > 0 else 'none'

def get_country_for_patient_id(id):
    result = df_patient[df_patient.patient_id == id].country.values;
    return result[0] if len(result) > 0 else 'none'


# #### Infection network for all samples

# In[ ]:


values = data_infected_by[['patient_id', 'infected_by']].values.astype(int)

plt.figure(figsize=(20,15))
plt.title("Infection network for all samples\n blue - Korea, red - China, green - rest")
G1=nx.Graph()
G1.add_edges_from(values)
c_map =  ['c' if get_country_for_patient_id(node) == 'Korea' 
          else 'r' if get_country_for_patient_id(node) == 'China' 
          else 'g'
          for node in G1 ]
# without labels - too long
nx.draw(G1,with_labels=False,node_color=c_map, width=3.0, node_size=300)


# #### Infection network in Korea

# In[ ]:


infected_network_korea = data_infected_by[data_infected_by.country == 'Korea']
values = infected_network_korea[['patient_id', 'infected_by']].values.astype(int)

plt.figure(figsize=(20,15))
plt.title("Infection network in Korea\n blue - male, red - female, green - no data")
G1=nx.Graph()
G1.add_edges_from(values)
c_map =  ['c' if get_sex_for_patient_id(node) == 'male' 
          else 'r' if get_sex_for_patient_id(node) == 'female' 
          else 'g'
          for node in G1 ]
# without labels - too long
nx.draw(G1,with_labels=False,node_color=c_map)


# # Route

# **Columns**
# 
# 
# 1. **patient_id** the ID of the patient
# 2. **global_num** the number given by KCDC
# 3. **date** YYYY-MM-DD
# 4. **province** Special City / Metropolitan City / Province(-do)
# 5. **city** City(-si) / Country (-gun) / District (-gu)
# 6. **latitude** the latitude of the visit (WGS84)
# 7. **longitude** the longitude of the visit (WGS84)

# In[ ]:


df_route.head()


# In[ ]:


df_route.info()


# ### City

# In[ ]:


f, ax = plt.subplots(figsize=(15, 10))
sns.countplot(y="city", data=df_route, color="c");


# ### Province

# In[ ]:


f, ax = plt.subplots(figsize=(15, 10))
sns.countplot(y="province", data=df_route, color="c");


# ### Visit

# In[ ]:


f, ax = plt.subplots(figsize=(15, 5))
sns.countplot(y="visit", data=df_route, color="c");


# ### Latitude / Longitude

# In[ ]:


import folium
southkorea_map = folium.Map(location=[36.55,126.983333 ], zoom_start=8,tiles='Stamen Toner')

for lat, lon in zip(df_route['latitude'], df_route['longitude']):
    folium.CircleMarker([lat, lon],
                        radius=5,
                        color='red',
                        
                        fill_color='red',
                        fill_opacity=0.7 ).add_to(southkorea_map)
southkorea_map


# ## Time

# **Columns**
# 
# 1. **date** YYYY-MM-DD
# 2. **time** Time (0 = AM 12:00 / 16 = PM 04:00)
# 3. **test** the accumulated number of tests
# 4. **negative** the accumulated number of negative results
# 5. **confirmed** the accumulated number of positive results
# 6. **released** the accumulated number of releases
# 7. **deceased** the accumulated number of deceases

# In[ ]:


df_time.head()


# In[ ]:


df_time.info()


# In[ ]:


df_time.describe()


# # To be continued...
