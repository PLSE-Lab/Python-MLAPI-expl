#!/usr/bin/env python
# coding: utf-8

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


# # Coronavirus(2019)
# Understanding coronavirus growth, distribution by country and state / region
# 
# # Introduction
# 
# The 2019-nCoV is a contagious coronavirus that hailed from Wuhan, China. This new strain of virus has striked fear in many countries as cities are quarantined and hospitals are overcrowded. How coronavirus have grow in 2019/2020 and what is his cases distribution in South Korea and other countries are questions to be answered by this kernel.

# ![](http://www.thesun.co.uk/wp-content/uploads/2020/01/tp-graphic-corona-virus-2155.jpg)

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta


# In[ ]:


route_df = pd.read_csv("/kaggle/input/coronavirusdataset/route.csv")
patient_df = pd.read_csv("/kaggle/input/coronavirusdataset/patient.csv")
time_df = pd.read_csv("/kaggle/input/coronavirusdataset/time.csv")


# # DATA DESCRIPTION
# 
# id - the ID of the patient (n-th confirmed patient)
# 
# sex - the sex of the patient
# 
# birth_year - the birth year of the patient
# 
# country - the country of the patient
# 
# region - the region of the patient
# 
# group - the collective infection
# 
# infection_reason - the reason of infection
# 
# infection_order - the order of infection
# 
# infected_by - the ID of who has infected the patient
# 
# contact_number - the number of contacts with people
# 
# confirmed_date - the date of confirmation
# 
# released_date - the date of discharge
# 
# deceased_date - the date of decease
# 
# state - isolated / released / deceased

# In[ ]:


route_df.head(2)


# In[ ]:


route_df.shape


# In[ ]:


patient_df.head(2)


# In[ ]:


patient_df.shape


# In[ ]:


time_df.head(2)


# In[ ]:


time_df.shape


# In[ ]:


patient_df.head(1)


# In[ ]:


route_df.head(1)


# In[ ]:


time_df.head()


# In[ ]:


train_data_merge = pd.merge(time_df,route_df,how="inner",on="date")


# In[ ]:


train_data_merge = pd.merge(route_df,patient_df,how="left",on="id")


# In[ ]:


train_data_merge.info()


# In[ ]:


train_data_merge.head(2)


# In[ ]:


train_data_merge.shape


# In[ ]:


train_data_merge.drop_duplicates(keep="first", inplace=True)


# In[ ]:


train_data_merge.info()


# In[ ]:


train_data_merge.isna().sum()


# In[ ]:


percent_missing = train_data_merge.isnull().sum() * 100 / len(patient_df)

missing_value_df = pd.DataFrame({'column_name': train_data_merge.columns,
                                 'percent_missing': percent_missing})


# In[ ]:


missing_value_df.sort_values('percent_missing', inplace=True)


# In[ ]:


missing_value_df


# In[ ]:


train_data_merge.head(2)


# In[ ]:


train_data_merge.describe()


# In[ ]:


train_data_merge.group.value_counts()


# In[ ]:


train_data_merge.infection_reason.value_counts()


# In[ ]:


train_data_merge.infection_order.value_counts()


# In[ ]:


train_data_merge.country.value_counts()


# In[ ]:


train_data_merge.province.value_counts()


# In[ ]:


temp = train_data_merge.groupby(['country', 'state'])['disease'].max()
temp


# In[ ]:


train_data_merge['group'] = train_data_merge['group'].fillna((train_data_merge['group'].mode()))


# In[ ]:


train_data_merge['infection_reason'] = train_data_merge['infection_reason'].fillna((train_data_merge['infection_reason'].mode()))


# In[ ]:


train_data_merge['infection_order'] = train_data_merge['infection_order'].fillna((train_data_merge['infection_order'].median()))


# In[ ]:


import plotly.express as px


# In[ ]:


states = pd.DataFrame(train_data_merge["state"].value_counts())
states["status"] = states.index
states.rename(columns={"state": "count"}, inplace=True)

fig = px.pie(states,
             values="count",
             names="status",
             title="Current state of patients",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="value+percent+label")
fig.show()


# In[ ]:


dead = train_data_merge[train_data_merge.state == 'deceased']
dead.head()


# In[ ]:


male_dead = dead[dead.sex=='male']
female_dead = dead[dead.sex=='female']


# In[ ]:


plt.figure(figsize=(15, 5))
plt.title('sex')
dead.sex.value_counts().plot.bar();


# In[ ]:


plt.figure(figsize=(15,5))
plt.title('Infection reason')
train_data_merge.infection_reason.value_counts().plot.bar();


# # State of Patient

# In[ ]:


sns.set(rc={'figure.figsize':(10,10)})
sns.countplot(x=train_data_merge['state'].loc[
    (train_data_merge['infection_reason']=='contact with patient')
])


# # State of male patient

# In[ ]:


sns.set(rc={'figure.figsize':(5,5)})
sns.countplot(x=train_data_merge['state'].loc[(train_data_merge['sex']=="male")])


# # State of female patient

# In[ ]:


sns.set(rc={'figure.figsize':(5,5)})
sns.countplot(x=train_data_merge['state'].loc[(train_data_merge['sex']=="female")])


# In[ ]:


# type casting : object -> datetime
train_data_merge.confirmed_date = pd.to_datetime(train_data_merge.confirmed_date)

# get daily confirmed count
daily_count = train_data_merge.groupby(train_data_merge.confirmed_date).id.count()

# get accumulated confirmed count
accumulated_count = daily_count.cumsum()


# In[ ]:


daily_count.plot()
plt.title('Daily Confirmed Count');


# In[ ]:


accumulated_count.plot()
plt.title('Accumulated Confirmed Count');


# In[ ]:


train_data_merge['age'] = 2020 - train_data_merge['birth_year'] 


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


train_data_merge["age_range"] = train_data_merge["age"].apply(group_age)


# In[ ]:


age_ranges = sorted(set([ar for ar in train_data_merge["age_range"] if ar != "Unknown"]))


# In[ ]:


date_cols = ["confirmed_date", "released_date", "deceased_date"]
for col in date_cols:
    train_data_merge[col] = pd.to_datetime(train_data_merge[col])


# In[ ]:


train_data_merge["time_to_release_since_confirmed"] = train_data_merge["released_date"] - train_data_merge["confirmed_date"]

train_data_merge["time_to_death_since_confirmed"] = train_data_merge["deceased_date"] - train_data_merge["confirmed_date"]
train_data_merge["duration_since_confirmed"] = train_data_merge[["time_to_release_since_confirmed", "time_to_death_since_confirmed"]].min(axis=1)
train_data_merge["duration_days"] = train_data_merge["duration_since_confirmed"].dt.days
age_ranges = sorted(set([ar for ar in train_data_merge["age_range"] if ar != "Unknown"]))
train_data_merge["state_by_gender"] = train_data_merge["state"] + "_" + train_data_merge["sex"]


# In[ ]:


age_gender_hue_order =["isolated_female", "released_female", "deceased_female",
                       "isolated_male", "released_male", "deceased_male"]
custom_palette = sns.color_palette("Reds")[3:6] + sns.color_palette("Blues")[2:5]

plt.figure(figsize=(12, 8))
sns.countplot(x = "age_range",
              hue="state_by_gender",
              order=age_ranges,
              hue_order=age_gender_hue_order,
              palette=custom_palette,
              data=train_data_merge)
plt.title("State by gender and age", fontsize=16)
plt.xlabel("Age range", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc="upper right")
plt.show()


# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(train_data_merge, hue = 'state', size = 10).map(plt.scatter, 'age', 'region').add_legend()
plt.title('Region by age and state')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




