#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# --------------------
# 1. [ Terminology ](#1)
# 2. [Packages Import & Util functions ](#2)
# 3. [Data Import  & Data Head](#3)
# 4. [Analysis for Each Year](#4)
# 5. [Analysis Through Time](#5)

# # Terminology <a id="1"></a>
# ----------------------------
# ## Area Id
# Oakland is divided into areas, the following map shows the divison of the city into area codes:
# 
# ![area map](http://ww2.kqed.org/news/wp-content/uploads/sites/10/2013/10/Screen-Shot-2013-10-14-at-8.28.43-PM-659x591-e1381880503593.png)
# 
# ## Beat  
# In police terminology, a beat is the territory and time that a police officer patrols.
# Beat police typically patrol on foot or bicycle which provides more interaction between police and community members.
# 
# ### Oakland Beats Map
# ![Beat map](http://www.panil.org/ncpc/beat9x_map.jpg)
# 
# ## Priority
#    ### PRIORITY 1:
#    > Emergency call which requires immediate response and there is reason to believe that an
#    > immediate threat to life exists.
#    ### PRIORITY 2: 
#    > Emergency call which requires immediate response and there exists an immediate and
#    > substantial risk of major property loss or damage.
#    
# 

# # Packages Import <a id="2"></a>
# ---------------------------

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import json
import ast
import datetime as dt
from dateutil import tz
from dateutil import parser
import matplotlib.pyplot as plt
import time

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print('\n'.join(os.listdir("../input")))

# Any results you write to the current directory are saved as output.


# ## Util functions

# In[ ]:


intervals = (
    ('weeks', 604800),  # 60 * 60 * 24 * 7
    ('days', 86400),    # 60 * 60 * 24
    ('hours', 3600),    # 60 * 60
    ('minutes', 60),
    ('seconds', 1),
    )


def get_nlargest_incident_id(n, df):
    return df.groupby(by="Incident Type Id",sort=True, as_index=False).count().nlargest(n, 'Create Time')["Incident Type Id"].values

def display_time(seconds, granularity=10):
    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:granularity])

def map_x(x):
    if x.hour < 6:
        return "00AM-6AM"
    if x.hour < 12 and x.hour > 6:
        return "6AM-12PM"
    if x.hour >= 12 and x.hour<18:
        return "12PM-6PM"
    if x.hour > 18:
        return "6PM-00AM"
    
def prep_data(df):
    df['Create Time'] = df['Create Time'].fillna(df['Closed Time'])
    df['Closed Time'] = df['Closed Time'].fillna(df['Create Time'])
    df["time_between_creation_and_closed_seconds"] = df.apply(lambda x: abs((parser.parse(x["Closed Time"]) - parser.parse(x["Create Time"])).seconds), axis=1)
    df["time_of_the_day"] = df["Create Time"].map(lambda x:map_x(parser.parse(x)))
    df.replace(r'', np.nan, regex=True, inplace=True)
    df["Area Id"].fillna(-1, inplace=True)
    df["Beat"].fillna("Unknown", inplace=True)
    df["Priority"].fillna("-1", inplace=True)
    df["Priority"].astype(int)
    df.drop(["Agency", "Event Number"], inplace=True, axis=1)
    df["day_of_the_month"] = df["Create Time"].apply(lambda x: parser.parse(x).day)
    df["day_of_the_week"] = df["day_of_the_month"].apply(lambda x: (x % 7) + 1)
    df["month_of_the_year"] = df["Create Time"].apply(lambda x: parser.parse(x).month)
    return df


# # Data Head <a id="3"></a>
# -------------------------------------

# ## 2011
# 

# In[ ]:


crimes_2011 = pd.read_csv("../input/records-for-2011.csv")
crimes_2011.drop(index=[180015], inplace=True)
crimes_2011 = prep_data(crimes_2011)
crimes_2011.rename(index=str, columns={"Location": "address"}, inplace=True)
crimes_2011["Priority"].replace(0.0, 1.0, inplace=True)
crimes_2011["Priority"] = crimes_2011["Priority"].astype(int)
crimes_2011.head(2)


# ## 2012
# 

# In[ ]:


crimes_2012 = pd.read_csv("../input/records-for-2012.csv")
crimes_2012.dropna(thresh=9, inplace=True)
crimes_2012["needs_recoding"] = crimes_2012["Location 1"].apply(lambda x:
                                                                     ast.literal_eval(x)["needs_recoding"])
crimes_2012["address"] = crimes_2012["Location 1"].apply(lambda x:
                                                                     ast.literal_eval(ast.literal_eval(x)["human_address"])["address"])
crimes_2012["address"] = crimes_2012["address"].apply(lambda x:x.replace("&amp;", "&"))
crimes_2012.drop(columns=["Location 1"], inplace=True)
crimes_2012 = prep_data(crimes_2012)
crimes_2012["Priority"] = crimes_2012["Priority"].astype(int)
crimes_2012.head(2)


# ## 2013
# 

# In[ ]:


crimes_2013 = pd.read_csv("../input/records-for-2013.csv")
crimes_2013.dropna(thresh=9, inplace=True)
crimes_2013 = prep_data(crimes_2013)
crimes_2013.rename(index=str, columns={"Location ": "address"}, inplace=True)
crimes_2013["Area Id"] = crimes_2013["Area Id"].astype(int)
crimes_2013["Priority"].replace(0.0, 1.0, inplace=True)
crimes_2013["Priority"] = crimes_2013["Priority"].astype(int)
crimes_2013.head(2)


# ## 2014

# In[ ]:


crimes_2014 = prep_data(pd.read_csv("../input/records-for-2014.csv"))
crimes_2014["needs_recoding"] = crimes_2014["Location 1"].apply(lambda x:
                                                                     ast.literal_eval(x)["needs_recoding"])
crimes_2014["address"] = crimes_2014["Location 1"].apply(lambda x:
                                                                     ast.literal_eval(ast.literal_eval(x)["human_address"])["address"])
crimes_2014["address"] = crimes_2014["address"].apply(lambda x:x.replace("&amp;", "&"))
crimes_2014.drop(columns=["Location 1"], inplace=True)
crimes_2014.head(2)


# ## 2015

# In[ ]:


crimes_2015 = prep_data(pd.read_csv("../input/records-for-2015.csv"))
crimes_2015.rename(index=str, columns={"Location": "address"}, inplace=True)
crimes_2015["Priority"].replace(0.0, 1.0, inplace=True)
crimes_2015["Priority"] = crimes_2015["Priority"].astype(int)
crimes_2015.head(2)


# ## 2016

# In[ ]:


crimes_2016 = pd.read_csv("../input/records-for-2016.csv")
crimes_2016.dropna(thresh=9, inplace=True)
crimes_2016 = prep_data(crimes_2016)
crimes_2016.rename(index=str, columns={"Location": "address"}, inplace=True)
crimes_2016["Priority"] = crimes_2016["Priority"].astype(int)
crimes_2016.head(2)


# # Analysis for Each Year <a id="4"></a>
# --------------------------

# In[ ]:


fig, ax = plt.subplots(nrows=2, ncols=3)
plt.subplots_adjust(left=0, right=2.5, top=3, bottom=1)
crimes_list = [crimes_2011, crimes_2012, crimes_2013, crimes_2014, crimes_2015, crimes_2016]
i = 0
for row in ax:
    for col in row:
        col.set_title(str(2011 + i))
        sns.countplot(data=crimes_list[i], x="Priority", ax=col, palette="Set1")
        i+=1


# ## Priority Reports Count With Respect to the Time of the Day

# In[ ]:


fig, ax = plt.subplots(nrows=2, ncols=3)
plt.subplots_adjust(left=0, right=2.5, top=3, bottom=1)
i = 0
for row in ax:
    for col in row:
        col.set_title(str(2011 + i))
        sns.countplot(data=crimes_list[i], x="Priority", hue="time_of_the_day", palette="Set1", ax=col)
        i+=1


# ## Incident type ID
# 

# In[ ]:


fig, ax = plt.subplots(nrows=2, ncols=3)
plt.subplots_adjust(left=0, right=2.5, top=3, bottom=1)
i = 0
nlargest = [set(get_nlargest_incident_id(10, x)) for x in crimes_list]
print("From 2011 to 2016 Top 10 Common Incident Types are: {}".format(str(set.intersection(*nlargest))))
for row in ax:
    for col in row:
        col.set_title(str(2011 + i))
        sns.countplot(data=crimes_list[i].loc[crimes_list[i]['Incident Type Id'].isin(nlargest[i])], x="Incident Type Id", hue="Priority", palette="Set1", ax=col)
        i += 1


# # Time It Takes to Handle a Report
# -----------------
# Showing the maximum and average time from creation (Create Time) to closing it (Closed Time). 

# In[ ]:


for i, crime_year in enumerate(crimes_list):
    avg_response_prio_1 = display_time(crime_year[crime_year["Priority"] == 1]["time_between_creation_and_closed_seconds"].mean())
    max_response_prio_1 = display_time(crime_year[crime_year["Priority"] == 1]["time_between_creation_and_closed_seconds"].max())

    avg_response_prio_2 = display_time(crime_year[crime_year["Priority"] == 2]["time_between_creation_and_closed_seconds"].mean())
    max_response_prio_2 = display_time(crime_year[crime_year["Priority"] == 2]["time_between_creation_and_closed_seconds"].max())
    print("Year " + str(2011 + i) +":\n")
    print("""The average time it takes to close a call (Priority 1):\n\t Max: {}\n\t Average: {} \n\n 
The average time it takes to close a call (Priority 2): \n\t max: {}\n\t average: {}""".format(max_response_prio_1, avg_response_prio_1, max_response_prio_2, avg_response_prio_2))
    print("=======================================================================================\n")


# # Do People Commit Less Crimes on Weekends \ Speific Months ?
# 
# The decrease of crimes committed on the 31th is becuase not every month has 31 days, same for september since september has 28 days, so there are 2 to 3 less days  to commit crimes.
# 
# ## We can observe that people do commit less crimes on weekends !

# In[ ]:


fig, ax = plt.subplots(nrows=6, ncols=3)
plt.subplots_adjust(left=0, right=3, top=12, bottom=0)
i_list = 0
for i, row in enumerate(ax):
    for j, col in enumerate(row):
        year_string = str(2011 + i)
        
        if j == 1:
            month_or_day = 'Day of The Month'
            title = year_string+ '\n' + month_or_day +'\n Crime Count'
            col.set_title(title)
            col.set_xticklabels(col.get_xticklabels(), rotation=90)
            sns.countplot(data=crimes_list[i_list], x="day_of_the_month" ,palette="Set1", ax=col)
        elif j == 2:
            month_or_day = 'Month of The Year'
            title = year_string + '\n' + month_or_day +'\n Crime Count'
            col.set_title(title)
            sns.countplot(data=crimes_list[i_list], x="month_of_the_year",palette="Set1", ax=col)
        else:
            month_or_day = 'Day of The Week'
            title = year_string+ '\n' + month_or_day +'\n Crime Count'
            col.set_title(title)
            sns.countplot(data=crimes_list[i_list], x="day_of_the_week" ,palette="Set1", ax=col)
            
    i_list += 1


# # Top Dangerous Beats
# --------------------
# 

# In[ ]:


fig, ax = plt.subplots(nrows=2, ncols=3)
plt.subplots_adjust(left=0, right=2.5, top=3, bottom=1)
i = 0
for row in ax:
    for col in row:
        col.set_title(str(2011 + i))
        #sns.countplot(data=crimes_list[i].loc[crimes_list[i]['Incident Type Id'].isin(nlargest[i])], x="Incident Type Id", hue="Priority", palette="Set1", ax=col)
        temp = crimes_list[i].groupby(by=["Beat", "Priority"],sort=True, as_index=False).count().rename(index=str, columns={"Create Time": "Count"})[["Beat", "Priority", "Count", "time_of_the_day"]]
        beats_prio_1 = list(temp[temp["Priority"] == 1].nlargest(5, "Count")["Beat"].values)
        beats_prio_2 = list(temp[temp["Priority"] == 2].nlargest(5, "Count")["Beat"].values)
        print("Year " + str(2011 +i ) +":\n")
        print("The Beats With the Most Reports (Priority 1, Decending Order): {} \nThe Beats With the Most Reports (Priority 2, Decending Order): {} \nUnique Beats: {}".format(str(beats_prio_1), str(beats_prio_2), str(list(set(beats_prio_1)|set(beats_prio_2)))))
        print("Common Beats: {}".format(str(list(set(beats_prio_1) & set(beats_prio_2)))))
        sns.barplot(data=temp[temp["Beat"].isin(beats_prio_1 + beats_prio_2)], x="Beat", y="Count", hue="Priority",palette="Set1", ax=col)
        print("=======================================================================================\n")
        i += 1


# In[ ]:


fig, ax = plt.subplots(nrows=2, ncols=3)
plt.subplots_adjust(left=0, right=2.5, top=3, bottom=1)
i = 0
for row in ax:
    for col in row:
        col.set_title(str(2011 + i))
        temp = crimes_list[i].groupby(by=["Beat", "Priority"],sort=True, as_index=False).count().rename(index=str, columns={"Create Time": "Count"})[["Beat", "Priority", "Count", "time_of_the_day"]]
        beats_prio_1 = list(temp[temp["Priority"] == 1].nlargest(5, "Count")["Beat"].values)
        beats_prio_2 = list(temp[temp["Priority"] == 2].nlargest(5, "Count")["Beat"].values)
        sns.countplot(data=crimes_list[i][crimes_list[i]["Beat"].isin(beats_prio_1 + beats_prio_2)], x="Beat", hue="time_of_the_day",palette="Set1", ax=col)
        i += 1


# ## Top 3 Worst Streets

# In[ ]:


fig, ax = plt.subplots(nrows=2, ncols=3)
plt.subplots_adjust(left=0, right=3, top=3, bottom=1)
i = 0
for row in ax:
    for col in row:
        col.set_title(str(2011 + i))
        temp = crimes_list[i].groupby(by=["address", "Priority"],sort=True, as_index=False).count().rename(index=str, columns={"Create Time": "Count"})[["address", "Priority", "Count", "time_of_the_day"]]
        bad_streets = temp.nlargest(3,"Count")["address"].values
        print("Year " + str(2011 + i) +": \n")
        print("Top 3 Worst Streets are:\n\t" + '\n\t'.join(bad_streets))
        print("=========================================================")
        sns.countplot(data=crimes_list[i][crimes_list[i]["address"].isin(bad_streets)], x="address", hue="Priority", palette="Set1", ax=col)
        i += 1


# # Analysis Through Time <a id="5"></a>
# -----------------------------------------------
# 

# In[ ]:


for i, x in enumerate(crimes_list):
    x["Year"] = 2011 + i
combined = crimes_2011
for x in range(1,len(crimes_list)):
    combined = combined.append(crimes_list[x], ignore_index=True)
combined.tail(5)


# In[ ]:


temp = combined.groupby(by=["Year", "Priority"]).mean()
prio_1 = temp.loc[list(zip(range(2011,2017),[1.0] * 6))]["time_between_creation_and_closed_seconds"]
prio_2 = temp.loc[list(zip(range(2011,2017),[2.0] * 6))]["time_between_creation_and_closed_seconds"]
plt.plot(range(2011, 2017),prio_1, marker='o', markerfacecolor='black', markersize=8, color='skyblue', linewidth=2, label="Avg Closing Time Priority 1")
plt.plot(range(2011, 2017), prio_2, marker='*',color="red", markersize=10, markerfacecolor='black', linewidth=2, label="Avg Closing Time Priority 2")
plt.legend()


# # Worst Beats Through Time
# 
# 

# In[ ]:


beats_list = combined.groupby(by=["Beat"], as_index=False).count().nlargest(5, "Create Time")["Beat"].values
sns.catplot(x="Beat",hue="Year",  y="Create Time",
                 data=combined[combined["Beat"].isin(beats_list)].groupby(by=["Year", "Beat"], as_index=False).count()[["Year", "Beat", "Create Time"]].reset_index(),
                 kind="bar")


# # Top 3 Worst Streets Through Time

# In[ ]:


address_list = combined.groupby(by=["address"], as_index=False).count().nlargest(3, "Create Time")["address"].values
sns.catplot(y="address",hue="Year",  x="Create Time",
                 data=combined[combined["address"].isin(address_list)].groupby(by=["Year", "address"], as_index=False).count()[["Year", "address", "Create Time"]].reset_index(),
                 kind="bar",orient="h")


# # Your Constructive Criticism is Welcome
# -------------------------
# I am a newbie and I have big passion towards  data science ! 
# 
# Any little piece of advice means a lot to me !
