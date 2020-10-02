#!/usr/bin/env python
# coding: utf-8

# Hello Kagglers!
# Welcome to my Kernal for the exercise [Dashboarding with Notebooks](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-1?utm_medium=email&utm_source=intercom&utm_campaign=dashboarding-event).  
# 
# **Daashboards** are useful tools which convery changes in data over any specified time interval. To construct a relevent dashboard, one must draw draw from a regularly-updated source, identify metrics which are important important to the mission of the company or goal, and show information which is relevant to decision-making. Dashboards can also show the results of changes which have been made internally in order to support rapid implementation of data driven decisions. 
# 
# The dataset I have chosen is the [New York Bus Breakdown and Delays](http://https://www.kaggle.com/new-york-city/ny-bus-breakdown-and-delays) dataset. I chose this dataset because I am passionate about the educational industry as a whole, including the often neglected everyday hereos in the operational components. 
#  
#  As a part of this dashboard exercise and as a fictional representative of the school bus industry, I will delineate the following topics:
#  
# * The goals of your organization (like users or measures of pollution)
#     The goal of my organization is to determine the trends in school bus break-down rates by location and service, in order to improve and quote accurate commute times of students in New York state.  
# 
# * Things that you (or your colleagues) can change to affect those goals (like advertising spending or the number of factory inspections)
# My colleagues and I can change the number and quality of busses, the route taken, the lines of communcation, and the training of bus-driving personnel. 
# 
# * Thing you can't change but that will affect the outcome (like the school year, or weather conditions)
# My colleagues and I cannot change the drop-off times, traffic congestion, random mechanical failures, school year dates, weather & climate, and school district budgets. 
# 
# **Let's begin** by loading in the lastest updated dataset. Thank you for reading!
# 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


## Load in dataset, and see information ##
df = pd.read_csv("../input/bus-breakdown-and-delays.csv")
print(df.info())


# With 44.4MB of memory, this isn't a gigantic dataset, but it's often useful to inspect each column to get a feel for the data types, and appropriately categorize them. This has the benefit of reducing memory and speeding up certain processes. 

# In[ ]:


print(df.head(10))


# In[ ]:


for c in df.columns:
    print("---",c,"---")
    unq = df[c].value_counts() 
    print(unq)
 


# Some columns, such as school year and run type, are clearly categorical. Others, such as Occured on, should be datetimes. Many columns, unfortunately, need to be cleaned up, such as How_Long_Delayed.  With a sample of each column, and a descrption of the elements within each column, it's possible to now define a datatype structure. 

# In[ ]:


dtype_structure = {
    "category":["School_Year", "Busbreakdown_ID", "Run_Type", "Bus_No", "Route_Number", "Reason", 
                "Schools_Serviced", "Boro", "Bus_Company_Name", "Incident_Number", 
                "Breakdown_or_Running_Late", "School_Age_or_PreK"],
    #"float":   ["How_Long_Delayed"], # This will need to be cleaned later
    "int":     ["Number_Of_Students_On_The_Bus"],
    "datetime64":["Occurred_On", "Created_On", "Informed_On", "Last_Updated_On"],
    "bool":    ["Has_Contractor_Notified_Schools", "Has_Contractor_Notified_Parents", "Have_You_Alerted_OPT"],
    "object":  []    
}


# Before I can actually convert to the relevent dtypes - I notice that How_Long_Delayed is a string entry, when it actually makes more sense for this to be a float type, representing the number of minutes the bus will be delayed. Let's clean that up now.  

# In[ ]:


print(df["How_Long_Delayed"][0:10])


# With entries like "20 min", "30MINS", "1 hour", "20minutos", and "10-15", this won't be easy. Elements such as "/" will make this difficult as well because there are "1/2 hour" and "20/25 MINS"

# In[ ]:


delayed_str = df["How_Long_Delayed"].values
trial = ["-", "hr", "min", "/"]
for t in trial:
    matching = [s for s in delayed_str if t in str(s)]
    print(matching[0:6])
    
"""
Let's come back and clean this up later.
"""


# Once the columns have been cleaned up, I will finally convert to a clean Dataframe. 

# In[ ]:


for dtp, col in dtype_structure.items():
    df[col] = df[col].astype(dtp)
    
print(df.info())


# Great! Our dataframe is now 33.2 MB and properly dtype-d. Let's make some basic visualizations. 
# 
# What are the most common type of delays?               

# In[ ]:


rsn = df.groupby("Reason").size()
lth = range(len(rsn))
plt.bar(lth, rsn)
plt.xticks(lth, rsn.index, rotation=90)
plt.show()


# Heavy traffic account for a majority of the delays. Perhaps alternate routes can be explored on days which are known to have haviest traffic. 
# 
# When do delays occur most often?
# 

# In[ ]:


delay_time = df.groupby(df["Occurred_On"].map(lambda t: t.hour)).size()
lth = range(len(delay_time))
plt.bar(lth, delay_time)
plt.xticks(lth, delay_time.index, rotation=90)
plt.title("Number of Delays by Hour of the Day")
plt.ylabel("Number of delays reported")
plt.xlabel("Hour of the day")
plt.show()


# As expected, most of the delays occur between 5am and 9am, and 1pm to 5pm, times when school is beginning and ending. 
# 
# Which days are worst for traffic?

# In[ ]:


traff_day = df[df["Reason"] == "Heavy Traffic"].groupby(
                                                df["Occurred_On"].map(lambda t: t.weekday())).size()
wkd = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
lth = range(len(traff_day))
plt.bar(lth, traff_day)
plt.xticks(lth, wkd, rotation=90)
plt.title("Occurances of Heavy Traffic by Day of the Week")
plt.ylabel("Heavy Traffic Reports")
plt.show()


# It appears that Tuesdays have the most number of reports of delay by heavy traffic, altough it's likely not statistically significant. 
# 
# What aboout other types of delays?

# In[ ]:


wkd = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]#, "Saturday", "Sunday"]
nrows = 4
ncols = 3

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 6), dpi=80)
plt.suptitle("Occurance of Delay Reason by Weekday")
i = 1
for rsn in df["Reason"].unique():
    reason_day = df[df["Reason"] == rsn].groupby(
                                        df["Occurred_On"].map(lambda t: t.weekday())).size()
    lth = range(len(reason_day))
    plt.subplot(nrows,ncols,i)
    plt.bar(lth, reason_day)
    plt.xticks(lth, wkd, rotation=90)
    #plt.title("Occurances of {r} by Day of the Week".format(r=rsn))
    plt.ylabel(str(rsn))
    i+=1 

# delete empty axes
# axes[5,0].set_axis_off()
plt.show()


# 

# Interestingly, more delays due to Weather Conditions occur on Fridays. 
# 
# Now, I'd like to investigate the relatioship between a delay due to "Mechanical Problem" and Bus_Company_Name. Perhaps some companies have faulty busses in general. Bus_Company_Name is a bit messy, so let's clean it up first. 

# In[ ]:


bus_comp_val = df["Bus_Company_Name"].value_counts()

print(bus_comp_val[0:4])
print("...And so on")


# There are 117 different entries for bus companies. By inspection, however, I see that there are entries like "G.V.C., LTD" , "G.V.C. LTD. (B2192)", and "gvc", which all seem like the same company to me. 
# combine all entries which have the same first five characters, and then exclude companies with less than two hundred reports of any type of delay. Afterwards, I'll show the number of mechanical failures by company as a percentage of that company's overall delay occurances. 

# In[ ]:


merge_char = 5
exclude_cnt = 200

df["Short_Bus_Comp_Name"] = df["Bus_Company_Name"].str[0:merge_char]
# print(df["Short_Bus_Comp_Name"].value_counts())

bus_comp_cnt = df["Short_Bus_Comp_Name"].value_counts().tolist()
bus_comp_bns = df["Short_Bus_Comp_Name"].value_counts().index.tolist()

keep_cnt_lst = [k for k in bus_comp_cnt if k>exclude_cnt]
keep_bus_lst = bus_comp_bns[0:len(keep_cnt_lst)]
# print(keep_bus_lst) #61 elements, down from 117


grp_mech = df[df["Reason"] == "Mechanical Problem"].groupby("Short_Bus_Comp_Name").size() # Number of delays due to mechanical problems by company
grp_dely = df.groupby("Short_Bus_Comp_Name").size() # Number of any delays for any reason by company
# print(len(grp_mech), len(grp_dely)) The lengths don't match, some short companies don't have mech failures reported


mech_rates = {}
for comp in grp_mech[keep_bus_lst].index:
   mech_rates[comp] = grp_mech[comp]/grp_dely[comp] * 100
    
sorted_lst = sorted(mech_rates.items(), reverse=True, key=lambda x: x[1])

n = 10
top_n = sorted_lst[0:n]
lth = range(n)

# Let's narrow to the top 10:

plt.bar(lth, [v for k,v in top_n])
plt.xticks(lth, [k for k,v in top_n], rotation=90)
plt.show()




# To be fair, I will express the number of mechanics 

# In[ ]:





# In[ ]:





# In[ ]:




