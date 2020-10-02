#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import time 

from collections import Counter

from  graphs import *


# In[ ]:


data = pd.read_csv("/kaggle/input/characteristics-corona-patients/Characteristics_Corona_patients_version_5 13-6-20.csv",
                   parse_dates=['confirmed_date', 'deceased_date',"released_date","return_date", "date_onset_symptoms"])


# # The distribution of properties in the dataset

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
Graphs.hist_graf(data.confirmed_date_until_released_date, 10,
          "confirmed_date_until_released_date", ax,(8,0), "y")

# confirmed_date_until_deceased_date
Graphs.hist_graf(data.confirmed_date_until_deceased_date, 100,
          "confirmed_date_until_deceased_date", ax,(8,1), "y")

fig.tight_layout()
plt.show()


# many j

# In[ ]:


fig, ax = plt.subplots(3, 1, figsize=(15,45))
Graphs.h_bar_with_limit(data, 'symptoms', 20 , "symptoms", ax, 0 , "x", "blue")

Graphs.h_bar_with_limit(data, 'background_diseases', 300 , 'background_diseases', ax, 1 , "x", "blue")

Graphs.h_bar_with_limit(data, 'country', 4000 , 'country', ax, 2 ,  "x", "blue")

plt.show()

