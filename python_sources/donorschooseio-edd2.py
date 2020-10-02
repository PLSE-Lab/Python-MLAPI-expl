#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import plotly.plotly as py1
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools

import cufflinks as cf
cf.go_offline()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


donors_data = pd.read_csv("../input/Donors.csv")


# **Total number of donors**

# In[ ]:


donors_data.count()


# **Donations Dataset EDD**
# * Loading the donations Dataset

# In[ ]:


# Loading the donations dataset
donations_data = pd.read_csv("../input/Donations.csv")
donations_data.head()


# ** Donors who is Teacher or not **

# In[ ]:


donor_is_teacher = donors_data['Donor Is Teacher'].value_counts()
df = pd.DataFrame({'labels': donor_is_teacher.index,
                   'values': donor_is_teacher.values })
df.iplot(kind='pie',
         labels='labels',
         values='values', 
         title='Not Teacher vs Teacher')


# **Shows the Donations Dataset**

# In[ ]:


donations_data.head()


# **Top Donation Amount by a Donor** 

# In[ ]:


#donations_data[["Donor ID","Donation Amount"]]
donations_sum = donations_data["Donation Amount"].sort_values()
donations_data["Donation Amount"].sort_values()[-10:].plot(kind="Bar",title="Top Donation Amount by a Donor")


# **Least Donation Amount by a Donor** 

# In[ ]:


# Least 10 Donorsi
#print(donations_sum.head(10))
donation_5 = donations_data[donations_data["Donation Amount"] <= 10]
#donation_10 = donations _data[(donations_data["Donation Amount"] > 10) & (donations_data["Donation Amount"] <= 50)] 
print(donation_5.shape)
print()
#leastdonations = donations_sum[:30]
#df = pd.DataFrame({'labels': leastdonations.index,
#                   'values': leastdonations.values })
#print(df.values)
#df.iplot(kind='bar',
#         labels='labels',
#         values='values', 
#         title="")
#leastdonations.iplot(kind="Bar",title="Least Donation Amount by a Donor")
donations_data.shape


# In[ ]:


donation_10 = donations_data[(donations_data["Donation Amount"] > 10) & (donations_data["Donation Amount"] <= 100)] 
print(donation_10.shape)


# In[ ]:


donation_1000 = donations_data[(donations_data["Donation Amount"] > 100) & (donations_data["Donation Amount"] <= 1000)] 
print(donation_1000.shape)


# In[ ]:


donations_percent = {"10":donation_5.shape[0], "100":donation_10.shape[0],"1000":donation_1000.shape[0]}


# In[ ]:


donation_amount = donations_data[(donations_data["Donation Amount"] > 1000) & (donations_data["Donation Amount"] <= 10000)]
donations_percent["10000"]=donation_amount.shape[0]


# In[ ]:


donation_amount = donations_data[(donations_data["Donation Amount"] > 10000) & (donations_data["Donation Amount"] <= 30000)]
donations_percent["30000"]=donation_amount.shape[0]


# In[ ]:


donation_amount = donations_data[(donations_data["Donation Amount"] > 30000) & (donations_data["Donation Amount"] <= 50000)]
donations_percent["50000"]=donation_amount.shape[0]


# In[ ]:


donation_amount = donations_data[(donations_data["Donation Amount"] > 50000)]
donations_percent["100000"]=donation_amount.shape[0]


# In[ ]:


#import plotly.plotly as py
#import plotly.graph_objs as go

#labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']
#values = [4500,2500,1053,500]

#trace = go.Pie(labels=labels, values=values)

#offline.plotly.iplot([trace], filename='basic_pie_chart')


# In[ ]:


#iplot(trace, filename='basic_pie_chart')


# In[ ]:


total = donations_data.shape[0]
donations_percent
#for k,v 
print(donations_sum.shape[0])
print(total)
df = pd.Series(data=donations_percent)#.values(),index=donations_percent.keys())
#df = pd..from_dict(donations_percent)
df.plot(kind="bar")
df.head(10)
#df.iplot(kind='pie',title='Distribution of Donation Amount wise ')


# **Highest Donation amount by a Donor**

# In[ ]:


donations_data[donations_data["Donor ID"]== "8f70fc7370842e0709cd9af3d29b4b0b"]["Donation Amount"].sum()


# **Donation by each Donor**

# In[ ]:


donations_data.groupby("Donor ID").sum()


# **Loading the Projects Dataset**

# In[ ]:


projects_data = pd.read_csv("../input/Projects.csv")


# In[ ]:


projects_data.head()


# **Merging the Donation and Projects**

# In[ ]:


donations_merge_projects = donations_data.merge(projects_data,how="inner",on="Project ID")


# **Shows the dataset after merging the Donations and Projects dataset**

# In[ ]:


donations_merge_projects.head()


# In[ ]:


#donations_merge_projects.shape


# ** No.of Donations for a  particular  "Project Subject Category Tree" **

# In[ ]:


donations_merge_projects.groupby("Project Subject Category Tree").count()["Donation Amount"].plot(kind="Bar",figsize=(15,10),title="Donations Towards different Projects")


# In[ ]:



donations_merge_projects.groupby("Project Subject Category Tree").sum()["Donation Amount"].sort_values(ascending=False)[0:30].plot(kind="Bar",figsize=(15,10),title="Donations Towards different Projects")


# In[ ]:


schools_data = pd.read_csv("../input/Schools.csv")


# In[ ]:


schools_data["School District"].unique().size
print(schools_data.shape)
schools_data[0:400]


# In[4]:


resources_data = pd.read_csv("../input/Resources.csv")


# In[5]:


del reosurces_data


# In[14]:


resources_data.head()


# In[18]:


temp = resources_data["Resource Item Name"].value_counts()
data = [go.Bar(
            x = temp.index.values[0:15],
            y = temp.values[0:15],
            marker=dict(color="blue"),
)]
layout = go.Layout(
    title='January 2013 Sales Report',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename="baic-bar")

