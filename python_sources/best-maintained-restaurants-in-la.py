#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import time, os, warnings
color = sns.color_palette()
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')

inspections = pd.read_csv("../input/restaurant-and-market-health-inspections.csv")
violations = pd.read_csv("../input/restaurant-and-market-health-violations.csv")


# # Inspections

# ## General Info of Inspections

# In[ ]:


inspections.head()


# In[ ]:


inspections.describe()


# In[ ]:


inspections.info()


# This dataset is very clean, just program_name misses 132 values.

# ## Overview of Features

# ### Score

# In[ ]:


fig, ax  = plt.subplots(2, 1, figsize = (10, 8))
sns.boxplot(inspections["score"], ax = ax[0])
ax[0].set_title("Box plot of Score", fontsize = 14)
ax[0].set_xlabel("")
sns.distplot(inspections["score"], kde = True, bins = 20, ax = ax[1])
ax[1].set_xlabel("score")
ax[1].set_title("Distribution of Score", fontsize = 14)
plt.show()


# At the first glance, there are about $60$% of restaurants and markets received score 90 or above.

# ### Activity Date

# In[ ]:


inspections["activity_date"] = pd.to_datetime(inspections["activity_date"])

inspect_date = pd.DataFrame({"date":inspections["activity_date"].value_counts().index, 
                             "values":inspections["activity_date"].value_counts().values}).sort_values(by = "date")
plt.figure(figsize = (10, 5))
plt.plot(inspect_date["date"], inspect_date["values"])
plt.title("Inspections Overview By Date", fontsize = 14)
plt.show()


# In[ ]:


inspections["year"] = inspections["activity_date"].dt.year
inspections["month"] = inspections["activity_date"].dt.month
inspections["day"] = inspections["activity_date"].dt.day


# In[ ]:


fig, ax = plt.subplots(3, 1, figsize = (10, 15))

for idx, time in enumerate(["year", "month", "day"]):
    temp = inspections[time].value_counts()
    sns.barplot(temp.index, temp.values, order = temp.index, ax = ax[idx])
    ax[idx].set_xlabel(time)
    ax[idx].set_ylabel("frequency")
    ax[idx].set_title("Inspection {} Frequency".format(time), fontsize = 14)
    rects = ax[idx].patches
    labels = temp.values
    for rect, label in zip(rects, labels):
        ax[idx].text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
                     label, ha = "center", va = "bottom")
plt.show()


# We found:
# 
# * 1016 has the most Inspections
# * Inspections schedule in March more often
# * Inspections schedule in day 7, 13, 20, and 6 of each month

# ### Grade

# In[ ]:


inspections["grade"].unique()


# It seems some missing values are filled with empty strings. This implies that the dataset aren't clean actually and we will fill those missing value with 'Unknown'.

# In[ ]:


inspections.loc[inspections["grade"] == ' ', "grade"] = "Unknown"


# In[ ]:


inspections["grade"].unique()


# In[ ]:


plt.figure(figsize = (10, 5))
grade = inspections["grade"].value_counts()
ax = sns.barplot(grade.index, grade.values)
plt.xlabel("grade")
plt.ylabel("frequency")
plt.title("Inspection Grade Frequency", fontsize = 14)
rects = ax.patches
labels = grade.values
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
           label, ha = "center", va = "bottom")
plt.show()


# There is only one restaurant/market with an unknown grade! Let's see who it is.

# In[ ]:


inspections[inspections["grade"] == "Unknown"]


# Without loss of generality, this restaurant has a grade C since it scored 65. 

# In[ ]:


inspections.loc[inspections["grade"] == "Unknown", "grade"] = "C"


# ### Service Code

# In[ ]:


plt.figure(figsize = (10, 5))
inspections["service_code"].astype(str)
service_code = inspections["service_code"].value_counts()
ax = sns.barplot(service_code.index, service_code.values)
plt.xlabel("service code")
plt.ylabel("frequency")
plt.title("Top 20 Service Code Frequency", fontsize = 14)
rects = ax.patches
labels = service_code.values
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
           label, ha = "center", va = "bottom")
plt.show()


# ### Service Description

# In[ ]:


inspections["service_description"].unique()


# In[ ]:


plt.figure(figsize = (10, 5))
service_description = inspections["service_description"].value_counts()
ax = sns.barplot(service_description.index, service_description.values)
plt.xlabel("service description")
plt.ylabel("frequency")
plt.title("Service Description Frequency", fontsize = 14)
rects = ax.patches
labels = service_description.values
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
           label, ha = "center", va = "bottom")
plt.show()


# About $99$% of restaurant and merket owners statisfied the routine inspection. About $1$% don't, and they rescheduled a second inspection in order to get a better score.

# #### Service Description: Owner Initiated Routine Inspect

# In[ ]:


second_inspect = inspections[inspections["service_description"] == "OWNER INITIATED ROUTINE INSPECT."]


# In[ ]:


plt.figure(figsize = (10, 5))
second_grade = second_inspect["grade"].value_counts()
ax = sns.barplot(second_grade.index, second_grade.values)
plt.xlabel("grade")
plt.ylabel("frequency")
plt.title("Second Inspection Grade Frequency", fontsize = 14)
rects = ax.patches
labels = second_grade.values
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
           label, ha = "center", va = "bottom")
plt.show()


# Since we don't know what grade these restaurants or markets received in the first inspection, we assume some of them did get a better grade in the second round.

# ### Facility Zip Code

# In[ ]:


len(inspections["facility_zip"].unique())


# Many zip code contain their local zip code, which make the analysis more complicated, so we decide to use their primary zip code instead.  

# In[ ]:


inspections["facility_zip_pri"] = inspections["facility_zip"].apply(lambda x:x[:5])
inspections["facility_zip_pri"].unique()


# In[ ]:


plt.figure(figsize = (10, 5))
facility_zip = inspections["facility_zip_pri"].value_counts()
ax = sns.barplot(facility_zip.index[:20], facility_zip.values[:20], order = facility_zip.index[:20])
plt.xlabel("facility zip")
plt.xticks(rotation = 90)
plt.ylabel("frequency")
plt.title("First 20 Facility Zip Frequency", fontsize = 14)
rects = ax.patches
labels = facility_zip.values[:20]
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
           label, ha = "center", va = "bottom")
plt.show()


# At the first glance, most restaurants and markets are located in 90012, 90045, and 90028

# ### Facility Name

# In[ ]:


inspections["facility_name"].unique()


# In[ ]:


len(inspections["facility_name"].unique())


# In[ ]:


plt.figure(figsize = (10, 5))
facility_name = inspections["facility_name"].value_counts()
ax = sns.barplot(facility_name.index[:20], facility_name.values[:20], order = facility_name.index[:20])
plt.xlabel("facility name")
plt.xticks(rotation = 90)
plt.ylabel("frequency")
plt.title("Top 20 Facility Name Frequency", fontsize = 14)
rects = ax.patches
labels = facility_name.values[:20]
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
           label, ha = "center", va = "bottom")
plt.show()


# We found:
# 
# * Facility names are diverse, some name contain its store name and its store number/location

# ### Program Name

# In[ ]:


inspections["program_name"].unique()


# In[ ]:


len(inspections["program_name"].unique())


# In[ ]:


plt.figure(figsize = (10, 5))
program_name = inspections["program_name"].value_counts()
ax = sns.barplot(program_name.index[:20], program_name.values[:20], order = program_name.index[:20])
plt.xlabel("program name")
plt.xticks(rotation = 90)
plt.ylabel("frequency")
plt.title("Top 20 Program Name Frequency", fontsize = 14)
rects = ax.patches
labels = program_name.values[:20]
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
           label, ha = "center", va = "bottom")
plt.show()


# We found:
# * Program name is the name of a restaurant or a market.
# * And there are 12447 stores participated in the inspection. 
# * Some names are diverse, for example, subway and subway sandwiches, McDonald's and Mc Donalds (need to clean).

# ### Program Status

# In[ ]:


inspections["program_status"].unique()


# In[ ]:


plt.figure(figsize = (10, 5))
program_status = inspections["program_status"].value_counts()
ax = sns.barplot(program_status.index, program_status.values)
plt.xlabel("program status")
plt.ylabel("frequency")
plt.title("Program Status Frequency", fontsize = 14)
rects = ax.patches
labels = program_status.values[:15]
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
           label, ha = "center", va = "bottom")
plt.show()


# There are 6832 programs inactive, this tells us these restaruants or markets don't have a inspection in the activity date.

# ### Pe-Description

# In[ ]:


plt.figure(figsize = (10, 5))
pe_description = inspections["pe_description"].value_counts()
ax = sns.barplot(pe_description.index, pe_description.values, order = pe_description.index)
plt.xlabel("pe description")
plt.xticks(rotation = 90)
plt.ylabel("frequency")
plt.title("Pe Description Frequency", fontsize = 14)
rects = ax.patches
labels = pe_description.values
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
           label, ha = "center", va = "bottom")
plt.show()


# We found small restaurants are difficult to maintain clean.

# ## Regression Analysis

# Without loss of generality, inspectors go to the same restaurants or markets each year. Below, we will start analyzing the grade against with different features in each year.

# ### Activity Date vs Grade

# In[ ]:


plt.figure(figsize = (10, 5))
sns.countplot(x = "year", hue = "grade", data = inspections)
plt.xlabel("year")
plt.ylabel("frequency")
plt.title("Inspections Grade by Year", fontsize = 14)
plt.show()


# We found:
# 
# * In 2015, not many restaurants/markets participated in inspections
# * In 2017, there were less participants than in 2016. Less owner received grade A. 
#   * A possibility is that some restaurants/markets don't require inspection each year

# In[ ]:


years = sorted(inspections["year"].unique())


# In[ ]:


fig, ax = plt.subplots(2, 2, figsize = (15, 10))

for i, year in enumerate(years):
    temp = inspections[inspections["year"] == year]
    sns.countplot(x = "month", hue = "grade", data = temp, ax = ax[int(i/2)][i%2])
    ax[int(i/2)][i%2].set_xlabel("month")
    ax[int(i/2)][i%2].set_ylabel("frequency")
    ax[int(i/2)][i%2].set_title("Inspections Grade per Month, {}".format(year), fontsize = 14)
plt.show()


# We found:
# 
# * The inspections started in July, 2015.
# * July has less inspections in 2015, 2016,  and 2017

# ### Pe Descrition vs Grade

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize = (15, 5))

for i, year in enumerate(years[:2]):
    temp = inspections[inspections["year"] == year]
    sns.countplot(x = "pe_description", hue = "grade", data = temp, ax = ax[i])
    ax[i].set_xlabel("pe description")
    ax[i].set_ylabel("frequency")
    ax[i].set_title("Inspections Grade by Pe description, {}".format(year), fontsize = 14)

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation = 90)
    
plt.show()


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize = (15, 5))

for i, year in enumerate(years[2:]):
    temp = inspections[inspections["year"] == year]
    sns.countplot(x = "pe_description", hue = "grade", data = temp, ax = ax[i])
    ax[i].set_xlabel("pe description")
    ax[i].set_ylabel("frequency")
    ax[i].set_title("Inspections Grade by Pe description, {}".format(year), fontsize = 14)

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation = 90)
    
plt.show()


# Restaurants (0-30) seats high risk receive A grade.

# ### Program Status vs Grade

# In[ ]:


fig, ax = plt.subplots(2, 2, figsize = (15, 18))

for i, year in enumerate(years):
    temp = inspections[inspections["year"] == year]
    sns.countplot(x = "program_status", hue = "grade", data = temp, ax = ax[int(i/2)][i%2])
    ax[int(i/2)][i%2].set_xlabel("program status")
    ax[int(i/2)][i%2].set_title("Inspections Program Status in {}".format(year), fontsize = 14)
plt.show()


# ### Mean Grade Per Day

# In[ ]:


fig, ax = plt.subplots(2, 2, figsize = (15, 18))

for i, year in enumerate(years):
    temp = inspections[inspections["year"] == year]
    temp = temp.groupby(["activity_date", "grade"]).score.mean()
    temp.unstack().plot(stacked = False, colormap = plt.cm.Set3,
                        grid = False, legend = True, ax = ax[int(i/2)][i%2])
    ax[int(i/2)][i%2].set_xlabel("activity date")
    ax[int(i/2)][i%2].set_title("Inspections Average Grade in {}".format(year), fontsize = 14)
plt.show()


# ### Service Description vs Grade

# In[ ]:


fig, ax = plt.subplots(2, 2, figsize = (15, 10))

for i, year in enumerate(years):
    temp = inspections[inspections["year"] == year]
    sns.countplot(x = "service_description", hue = "grade", data = temp, ax = ax[int(i/2)][i%2])
    ax[int(i/2)][i%2].set_xlabel("service description")
    ax[int(i/2)][i%2].set_ylabel("frequency")
    ax[int(i/2)][i%2].set_title("Inspections Service Description in {}".format(year), fontsize = 14)
plt.show()


# Owners who recheculed a second inspection did get a grade A in most cases.

# ### Facility Zip vs Grade

# Ideas:
# * Classify zipcode by district 
# * Apply GeoPy to find long and lat by address

# ## Conclusion:

# In[ ]:




