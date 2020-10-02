#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/3-Airplane_Crashes_Since_1908.txt',                   infer_datetime_format = True, parse_dates = ["Date"])
data.info()


# In[ ]:


data.head()


# In[ ]:


# Top10 cases by "Route"
data["Route"].value_counts().nlargest(10).plot(kind = "barh").invert_yaxis()


# In[ ]:


# Top10 airplane type
data["Type"].value_counts().nlargest(10).plot(kind = "barh").invert_yaxis()


# In[ ]:


# Top10 Location
data["Location"].value_counts().nlargest(10).plot(kind = "barh").invert_yaxis()


# In[ ]:


# fatalities over time
plt.figure()
data.set_index(data["Date"])["Fatalities"].plot()


# # RFF exploration
# RFF here means Recency, Frequency and Fatalities, derived from the customer segmentation model RFM.
# Using this it should be checked if there is a difference between the number of incidents per
# airline and the total # of fatalities as well as the last incident. Indicating that although
# an airline has a lot of incidents it maybe became saver over time.

# In[ ]:


# create date for recency time difference
today = datetime.datetime(2016, 7, 16)

# get recency
data["Recency"] = data["Date"].apply(lambda x: (today - x).days)


# In[ ]:


# aggregation per airline for RFF
airline = data.groupby("Operator").agg({"Recency": "min",
                                        "Date": "count",
                                        "Fatalities": lambda x: x.sum(skipna = True)})
airline["Fatalities"] = airline["Fatalities"].astype(np.int32)
airline.rename(columns = {"Date": "Frequency", "Fatalities": "TotalFatal"}, inplace = True)
airline["FatPerIncident"] = round(airline["TotalFatal"] / airline["Frequency"], 0).astype(np.int32)
airline.head()


# ### Top15 most recent incidents
# (Recency in days)

# In[ ]:


sns.barplot(y = "Operator", x = "Recency", data = airline.nsmallest(15, columns = "Recency").reset_index())
plt.xlabel("Days since last incident")


# In[ ]:


airline.nsmallest(15, columns = "Recency").reset_index()


# ### Top15 airlines by frequencies
# This has also already been shown by other nice kernels.

# In[ ]:


sns.barplot(y = "Operator", x = "Frequency", data = airline.nlargest(15, columns = "Frequency").reset_index())
plt.xlabel("total # of incidents")


# In[ ]:


airline.nlargest(15, columns = "Frequency").reset_index()


# Looks like a diverse picture at the top of the total # of incidents. Some airlines with fewer
# incidents but more total # of fatalities and some airlines like Lufthansa or China National
# Aviation Corporation with their last incidents much longer ago.

# ### Top15 total fatalities

# In[ ]:


sns.barplot(y = "Operator", x = "TotalFatal", data = airline.nlargest(15, columns = "TotalFatal").reset_index())
plt.xlabel("Total # of fatalities")


# In[ ]:


airline.nlargest(15, columns = "TotalFatal").reset_index()


# ### Top15 average fatalities per incident

# In[ ]:


sns.barplot(y = "Operator", x = "FatPerIncident", data = airline.nlargest(15, columns = "FatPerIncident").reset_index())
plt.xlabel("Average # of fatalities per incident")


# In[ ]:


airline.nlargest(15, columns = "FatPerIncident").reset_index()


# In[ ]:


# checking for airlines with more than 1 total incidents
airline.loc[airline["Frequency"] > 1, :].nlargest(15, columns = "FatPerIncident").reset_index()


# ## Investigating Aeroflot and Military - U.S. Air Force
# Checking if there is a difference in the incidents by the planes they use.

# In[ ]:


# Top20 airplane types for "Aeroflot" incidents
aero20 = data.loc[data["Operator"] == "Aeroflot", "Type"]
aero20 = aero20.loc[aero20.isin(aero20.value_counts().nlargest(20).index)]
sns.countplot(y = aero20)


# In[ ]:


# Top20 location for "Aeroflot" incidents
aero20l = data.loc[data["Operator"] == "Aeroflot", "Location"]
aero20l = aero20l.loc[aero20l.isin(aero20l.value_counts().nlargest(20).index)]
sns.countplot(y = aero20l)


# In[ ]:


# Aeroflot incidents over time
plt.figure(figsize = (17, 6))
sns.countplot(x = data.loc[data["Operator"] == "Aeroflot", "Date"].apply(lambda x: x.year))
plt.title("Aeroflot incidents over time")


# In[ ]:


# Top20 airplane types for "Military - U.S. Air Force" incidents
us20 = data.loc[data["Operator"] == "Military - U.S. Air Force", "Type"]
us20 = us20.loc[us20.isin(us20.value_counts().nlargest(20).index)]
sns.countplot(y = us20)


# In[ ]:


# Top20 location for "Military - U.S. Air Force" incidents
us20l = data.loc[data["Operator"] == "Military - U.S. Air Force", "Location"]
us20l = us20l.loc[us20l.isin(us20l.value_counts().nlargest(20).index)]
sns.countplot(y = us20l)


# In[ ]:


# Military - U.S. Air Force incidents over time
plt.figure(figsize = (17, 6))
sns.countplot(x = data.loc[data["Operator"] == "Military - U.S. Air Force", "Date"].apply(lambda x: x.year))
plt.title("Military - U.S. Air Force incidents over time")


# ## Summary
# Although Aeroflot and Military - U.S. Air Force were the Top2 for the # of incidents and fatalities
# the plots show that their numbers keep steadily falling which is in line with the overall development
# of incidents and # of fatalities.
