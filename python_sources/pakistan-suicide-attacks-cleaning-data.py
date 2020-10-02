#!/usr/bin/env python
# coding: utf-8

# # Pakistan Suicide Attacks Cleaning data

# In[ ]:


import re
import math
import calendar
import numpy as np
import pandas as pd

from pprint import pprint
from datetime import datetime, date

csv = pd.read_csv("../input/PakistanSuicideAttacks Ver 4.csv", encoding="latin1")

csv.head()


# In[ ]:


a = list(set(csv["Latitude"]))[0]
b = list(set(csv["Latitude"]))[1]

#remove rows with empty latitude or empty longitude
clean_data_set = csv[csv["Latitude"].isnull() != True]
clean_data_set = clean_data_set[clean_data_set["Longitude"].isnull() != True]


# In[ ]:


#set the timestamp at the date place
def get_day_name(s):
    return s.split('-')[0]

month = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12
}

def get_month(s):
    try:
        return month[(s.split("-")[1].split()[0])]
    except:
        try:
            return month[(s.split("-")[1].split("-")[0])]
        except:
            return -1

def get_day(s):
    try:
        return int(s.split("-")[1].split()[1])
    except:
        try:
            return int(s.split("-")[2])
        except:
            return -1
    
def get_year(s):
    try:
        return int(s.split("-")[-1])
    except:
        return -1

def get_timestamp(s):
    try:
        return int(date(get_year(s), get_month(s), get_day(s)).strftime("%s"))
    except:
        return -1


clean_data_set["day"] = clean_data_set["Date"].apply(get_day_name)
clean_data_set["timestamp"] = clean_data_set["Date"].apply(get_timestamp)

clean_data_set = clean_data_set.drop(["Date", "Time", "Islamic Date", "Temperature(F)", "Province"], axis=1)

clean_data_set.head()

#clean_data_set[clean_data_set["timestamp"] == -1] ##all worked


# In[ ]:


print(set(clean_data_set["Blast Day Type"]))

#make the holiday weekends as a weekend

def is_WE(row):
    if row["day"] in ["Saturday", "Sunday"] or row["Holiday Type"] == "Weekend":
        return "Weekend"
    return row["Blast Day Type"]

clean_data_set["Blast Day Type"] = clean_data_set.apply(is_WE, axis=1)
clean_data_set = clean_data_set.drop(["Holiday Type"], axis=1)
clean_data_set["Blast Day Type"] = clean_data_set["Blast Day Type"].fillna('Working Day')

clean_data_set.head()


# In[ ]:


clean_data_set.keys()


# In[ ]:


#fill the nan values of nb killed: estimate the number of persons killed

min_number_killed_average = clean_data_set["Killed Min"].mean()
max_number_killed_average = clean_data_set["Killed Max"].mean()

def estimate_number_killed(row):
    if math.isnan(row["Killed Min"]) and math.isnan(row["Killed Max"]):
        return 0
    elif math.isnan(row["Killed Min"]):
        if min_number_killed_average < row["Killed Max"]:
            return (min_number_killed_average + row["Killed Max"] * 2) / 3. #set bigger coeff to the know value
        else:
            return row["Killed Max"] #if the average is bigger than the maxx kill, return the max kill
    elif math.isnan(row["Killed Max"]):
        if max_number_killed_average > row["Killed Min"]:
            return (max_number_killed_average + row["Killed Min"] * 2) / 3.
        else:
            return row["Killed Min"]
    else:
        return (row["Killed Max"] + row["Killed Min"]) / 2.


clean_data_set["nb_killed"] = clean_data_set.apply(lambda r: int(estimate_number_killed(r)), axis=1)
clean_data_set = clean_data_set.drop(["Killed Min", "Killed Max", "Hospital Names"], axis=1)

clean_data_set.head()


# In[ ]:


#clean the max injured column
def val_2_float(v):
    try:
        if math.isnan(v):
            return v
    except:
        non_decimal = re.compile(r'[^\d.]+')
        return float(non_decimal.sub('', v))
    
clean_data_set["Injured Max"] = clean_data_set["Injured Max"].apply(val_2_float)    


#estimate the nb of injured people
min_number_inj_average = clean_data_set["Injured Min"].mean()
max_number_inj_average = clean_data_set["Injured Max"].mean()

def estimate_number_injured(row):
    if math.isnan(row["Injured Min"]) and math.isnan(row["Injured Max"]):
        return 0
    elif math.isnan(row["Injured Min"]):
        if min_number_inj_average < row["Injured Max"]:
            return (min_number_inj_average + row["Injured Max"] * 2) / 3. 
        else:
            return row["Injured Max"]
    elif math.isnan(row["Injured Max"]):
        if max_number_inj_average > row["Injured Min"]:
            return (max_number_inj_average + row["Injured Min"] * 2) / 3.
        else:
            return row["Injured Min"]
    else:
        return (row["Injured Max"] + row["Injured Min"]) / 2.


clean_data_set["nb_injured"] = clean_data_set.apply(lambda r: int(estimate_number_injured(r)), axis=1)
clean_data_set = clean_data_set.drop(["Injured Min", "Injured Max"], axis=1)

clean_data_set.head()


# In[ ]:


#clean the No. of Suicide Blasts column: need integers
mean_nb_suicide_blast = int(clean_data_set["No. of Suicide Blasts"].mean())

clean_data_set["nb_kamikaze"] = clean_data_set["No. of Suicide Blasts"].apply(
    lambda v: mean_nb_suicide_blast if math.isnan(v) else v
)

clean_data_set = clean_data_set.drop(["No. of Suicide Blasts", "Location Sensitivity", "Influencing Event/Event"], axis=1)

clean_data_set.head()


# In[ ]:


s = list(set(clean_data_set["Explosive Weight (max)"]))
print(len(s)) #not enought data
print(s)      #to normalize !!!
#==> not use this column (and it's not really relevant)

clean_data_set = clean_data_set.drop(["Explosive Weight (max)"], axis=1)


# In[ ]:


print(set(clean_data_set["Targeted Sect if any"]))
print(len(clean_data_set[clean_data_set["Targeted Sect if any"] == "Shiite/sunni"])) #as ther is only one case, just affect to shiite

def get_target_sect(v):
    try:
        if math.isnan(v):
            return 'None'
    except:
        if v in ["Shiite", "shiite", "Shiite/sunni"]:
            return "Shiite"
        return v

print(set(clean_data_set["Targeted Sect if any"].apply(get_target_sect)))

clean_data_set["religious_target"] = clean_data_set["Targeted Sect if any"].apply(get_target_sect)
clean_data_set = clean_data_set.drop(["Targeted Sect if any", "Open/Closed Space"], axis=1)

clean_data_set.head()


# In[ ]:


#pprint(set(clean_data_set["Location Category"]))
#pprint(set(clean_data_set["Target Type"]))

#simplify the target type and remove the location category
def get_target_type(s):
    if s in ['Anti-Militants', 'Children/Women', 'Civilian', 'Media', 'Civilian & Police', 'civilian']:
        return "Civilian"
    elif s in ['Army', 'Frontier Corps ', 'Military', 'Police', 'Police & Rangers', 'Rangers', 'police']:
        return "Army"
    elif s in ['advocates (lawyers)', 'Judges & lawyers', 'Civilian Judges']:
        return "Law"
    elif s in ['Government official', 'Government Official']:
        return "Government"
    elif s in ['Foreigner', 'foreigner']:
        return "Foreign"
    elif s in ['religious', 'Shia sect', 'Religious']:
        return "Religious"
    return "Unknow"
    
pprint(list(set(map(get_target_type, set(clean_data_set["Target Type"])))))

clean_data_set["target_type"] = clean_data_set["Target Type"].apply(get_target_type)
clean_data_set = clean_data_set.drop(["Target Type", "Location Category"], axis=1)

clean_data_set.head()


# In[ ]:




