#!/usr/bin/env python
# coding: utf-8

# Make the information number (read model)

# In[ ]:


import numpy as np
import pandas as pd 
from control import *
from clean import *
from format_data import *


# In[ ]:


data = pd.read_csv("/kaggle/input/characteristics-corona-patients/Characteristics_Corona_patients version5 6-7-20.csv",
                   parse_dates=['confirmed_date', 'deceased_date',"released_date","return_date", "date_onset_symptoms"])


# In[ ]:


data.shape


# In[ ]:


[i for i in data.columns]


# In[ ]:


[Control.examining_values_by_col ([data], [""],j ) for j in data.columns ]


# ## make categories to number 

# In[ ]:


f = Format.category_col_to_num_col(data, "severity_illness")

data["severity_illness_infectious_person"] = data["severity_illness_infectious_person"].replace(f)


# In[ ]:


for j in  [ "sex", "treatment", "country" ]:
    l = Format.category_col_to_num_col(data, j)
    print(l)


# ## to hot vector

# In[ ]:


#  Format
def turn_hotvec(df, input_col):
    name_col = get_name_of_categorized_value(df, input_col)
    for j in name_col:
        name = j.replace(" ", "_")
        df[input_col + "_"+ name] = df[input_col][df[input_col].notnull()].apply(lambda i: 1 if j in i else 0)
        print(df[input_col + "_"+ name].value_counts())


# In[ ]:


get_ipython().run_cell_magic('time', '', "Format.turn_hotvec(data, 'symptoms')\nFormat.turn_hotvec(data, 'background_diseases')")


# ## Delete all features not for model

# In[ ]:


data = data.drop(['city', 'infection_place','region', 'symptoms', 'region', 
           'symptoms_no_symptom', 'background_diseases', 'date_onset_symptoms',  'deceased_date',
                  'released_date',  'return_date',  "infected_by",'background_diseases_', "confirmed_date",], axis=1)


# In[ ]:


data = data.drop([] , axis=1)


# In[ ]:


[Control.examining_values_by_col ([data], [""],j ) for j in data.columns ]


# In[ ]:


data.shape


# In[ ]:


[i for i in data.columns]


# ## to_csv

# In[ ]:


data.to_csv("Characteristics_Corona_patients_version_6 - 6-7-2020.csv", index = False)
data.to_csv()

