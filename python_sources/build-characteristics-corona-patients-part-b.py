#!/usr/bin/env python
# coding: utf-8

# # Features engineering and visual exploration of the data

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from control import *
from  graphs import *
from jengineer import *


# In[ ]:


data = pd.read_csv("/kaggle/input/covid19-re/Characteristics_Corona_patients-13.6.20.csv",
                   parse_dates=['confirmed_date', 'deceased_date',"released_date","return_date", "date_onset_symptoms"])


# # Feature Engineering

# Turn all date data into numbers
# (Measuring distance from 1.11.19)
# 

# In[ ]:


# 1min 2s
for j in ['confirmed_date', 'deceased_date',"released_date","return_date", "date_onset_symptoms"]:
    name = j+"_D"
    zero = pd.to_datetime("1.11.2019", dayfirst=True)
    data[name] = data[j].apply(lambda x: JEngineer.com(x, zero ) )


# interval days

# In[ ]:


data["id"] = range(0,len(data.confirmed_date))

JEngineer.interval_days(data,"id", "date_onset_symptoms","return_date", "return_date_until_date_onset_symptoms")
print(3)
JEngineer.interval_days(data, "id", "confirmed_date", "date_onset_symptoms", "date_onset_symptoms_until_confirmed_date")
print(3)
JEngineer.interval_days(data, "id","released_date","confirmed_date", "confirmed_date_until_released_date")
print(3)
JEngineer.interval_days(data, "id","deceased_date","confirmed_date", "confirmed_date_until_deceased_date")
print(3)


# severity_illness_infectious_person

# In[ ]:


indexs = data.index[data.infected_by.notnull()]

data["severity_illness_infectious_person"] = np.nan

for indx in indexs:
    i  = data.infected_by[indx]
    i = i.split(",")
    
    if len(i) == 1:
        ind = int(i[0])
        data.loc[indx, "severity_illness_infectious_person"] =  data.severity_illness[ind]
    
    elif len(i) > 1:
        pass


# # visual exam 

# In[ ]:


fig, ax = plt.subplots(9, 2, figsize=(15,45))
palette_severity_illness = {"asymptomatic": "#C77DE7", "good":"yellowgreen", "critical":"red", "deceased": "black", "cured": "#0BEFFF"}

# 0
# severity_illness
Graphs.basic_countplot(data.severity_illness, 'severity_illness', ax, (0,0), "y", palette_severity_illness)

# severity_illness_infectious_person
Graphs.basic_countplot(data.severity_illness_infectious_person, 'severity_illness_infectious_person', 
                ax, (0,1), "y", palette_severity_illness)


# line 1
# sex
Graphs.basic_countplot(data.sex, 'sex',  ax, (1,0), "y", None)

# treatment
Graphs.basic_countplot(data.treatment, 'treatment',  ax, (1,1), "y", None)


# line 2
# age_band
Graphs.hist_graf(data.age_band, 13, "age band", ax,(2,1), "y")

# age
Graphs.hist_graf(data.age, 120, "age", ax ,(2,0), "y")


# line 3
# return_date
Graphs.time_plot(data.return_date, "return_date", ax ,(3,0), "y", "blue")
Graphs.hist_graf(data.confirmed_date_d, 100, "age band", ax,(3,1), "y")

# line 4
# date_onset_symptoms
Graphs.time_plot(data.date_onset_symptoms, "date_onset_symptoms", ax ,(4,0), "y", "blue")

# confirmed_date
Graphs.time_plot(data.confirmed_date, "confirmed_date", ax ,(4,1), "y", "blue")


# line 5
# deceased_date
Graphs.time_plot(data.deceased_date, "deceased_date", ax ,(5,0), "y", "red")

# released_date
Graphs.time_plot(data.released_date, "cured_date", ax ,(5,1), "y", "blue")


#  line 6
# background_diseases_binary
Graphs.basic_countplot(data.background_diseases_binary, 'background_diseases_binary', 
                ax, (6,0), "y", None)

# smoking
Graphs.basic_countplot(data.smoking, 'smoking', ax, (6,1), "y", None)


#  line 7
# return_date_until_date_onset_symptoms
Graphs.hist_graf(data.return_date_until_date_onset_symptoms, 120,
          "return_date_until_date_onset_symptoms", ax, (7,0), "y")

# date_onset_symptoms_until_confirmed_date
Graphs.hist_graf(data.date_onset_symptoms_until_confirmed_date, 100,
          "date onset symptoms until confirmed date", ax,(7,1), "y")

#  line 8
# confirmed_date_until_released_date
Graphs.hist_graf(data.confirmed_date_until_released_date, 100,
          "confirmed_date_until_released_date", ax,(8,0), "y")

# confirmed_date_until_deceased_date
Graphs.hist_graf(data.confirmed_date_until_deceased_date, 100,
          "confirmed_date_until_deceased_date", ax,(8,1), "y")

fig.tight_layout()
plt.show()


# # solve problem 

# In[ ]:


data.deceased_date.apply(lambda x: print(x) if x.year < 2019 else None)


# In[ ]:


data["ind"] =range(len(data.sex))
delete =  []
data["ind"].apply(lambda ind: delete.append(ind) if data.deceased_date[ind].year < 2019 else None )
data =  data.drop(delete)


# In[ ]:


fig, ax = plt.subplots(9, 2, figsize=(15,45))
palette_severity_illness = {"asymptomatic": "#C77DE7", "good":"yellowgreen", "critical":"red", "deceased": "black", "cured": "#0BEFFF"}

# 0
# severity_illness
Graphs.basic_countplot(data.severity_illness, 'severity_illness', ax, (0,0), "y", palette_severity_illness)

# severity_illness_infectious_person
Graphs.basic_countplot(data.severity_illness_infectious_person, 'severity_illness_infectious_person', 
                ax, (0,1), "y", palette_severity_illness)


# line 1
# sex
Graphs.basic_countplot(data.sex, 'sex',  ax, (1,0), "y", None)

# treatment
Graphs.basic_countplot(data.treatment, 'treatment',  ax, (1,1), "y", None)


# line 2
# age_band
Graphs.hist_graf(data.age_band, 13, "age band", ax,(2,1), "y")

# age
Graphs.hist_graf(data.age, 120, "age", ax ,(2,0), "y")


# line 3
# return_date
Graphs.time_plot(data.return_date, "return_date", ax ,(3,0), "y", "blue")

# line 4
# date_onset_symptoms
Graphs.time_plot(data.date_onset_symptoms, "date_onset_symptoms", ax ,(4,0), "y", "blue")

# confirmed_date
Graphs.time_plot(data.confirmed_date, "confirmed_date", ax ,(4,1), "y", "blue")


# line 5
# deceased_date
Graphs.time_plot(data.deceased_date, "deceased_date", ax ,(5,0), "y", "red")

# released_date
Graphs.time_plot(data.released_date, "cured_date", ax ,(5,1), "y", "blue")


#  line 6
# background_diseases_binary
Graphs.basic_countplot(data.background_diseases_binary, 'background_diseases_binary', 
                ax, (6,0), "y", None)

# smoking
Graphs.basic_countplot(data.smoking, 'smoking', ax, (6,1), "y", None)


#  line 7
# return_date_until_date_onset_symptoms
Graphs.hist_graf(data.return_date_until_date_onset_symptoms, 120,
          "return_date_until_date_onset_symptoms", ax, (7,0), "y")

# date_onset_symptoms_until_confirmed_date
Graphs.hist_graf(data.date_onset_symptoms_until_confirmed_date, 100,
          "date onset symptoms until confirmed date", ax,(7,1), "y")

#  line 8
# confirmed_date_until_released_date
Graphs.hist_graf(data.confirmed_date_until_released_date, 100,
          "confirmed_date_until_released_date", ax,(8,0), "y")

# confirmed_date_until_deceased_date
Graphs.hist_graf(data.confirmed_date_until_deceased_date, 100,
          "confirmed_date_until_deceased_date", ax,(8,1), "y")

fig.tight_layout()
plt.show()


# In[ ]:


jnujnj


# In[ ]:


data.to_csv("Characteristics_Corona_patients_version_5 13-6-20.csv", index = False)
data.to_csv()

