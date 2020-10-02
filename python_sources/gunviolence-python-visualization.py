#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns 
import matplotlib.pyplot as plt
import os 
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


# In[ ]:


# Read data file
gundata = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")


# In[ ]:


# Enlists the column names 
gundata.columns.values


# In[ ]:


# Study the contents of the data by using head, to get the feel of the data
gundata.head(3)


# In[ ]:


# Gives dimensions of the data
gundata.shape


# In[ ]:


#Describes the data
gundata.describe()


# In[ ]:


# To study how many columns have null values , and its count
pd.isnull(gundata).sum()


# In[ ]:


# Drop the unnecessary columns 
gundata.drop(['latitude','source_url','incident_url','longitude','incident_url_fields_missing','notes','gun_stolen','sources','state_house_district','state_senate_district'], axis = 1, inplace = True)


# In[ ]:


gundata.columns.values


# In[ ]:


gundata.shape


# In[ ]:


# Some feature engineering to introduce new columns , and process data
gundata['date'] = pd.to_datetime(gundata['date'])


# In[ ]:


gundata['year'] = gundata['date'].dt.year


# In[ ]:


gundata['month'] = gundata['date'].dt.month


# In[ ]:


gundata['weekday'] = gundata['date'].dt.weekday


# In[ ]:


def generic_split(n) :                    
    generic_rows = []               
    generic_row = str(n).split("||")     
    for i in generic_row :              
        g_row = str(i).split("::")              
        if len(g_row) > 1 :         
            generic_rows.append(g_row[1])           
    return generic_rows


# In[ ]:


# Split the participant gender to count no of Males/Females
gender_series = gundata.participant_gender.apply(generic_split)
gundata["total_ppl"] = gender_series.apply(lambda x: len(x))
gundata["male"] = gender_series.apply(lambda i: i.count("Male"))
gundata["female"] = gender_series.apply(lambda i: i.count("Female"))


# In[ ]:


# We further participant_age_group split as per age group 
gundata_agegrp = gundata.participant_age_group.apply(generic_split)
gundata["Child"] = gundata_agegrp.apply(lambda i: i.count("Child 0-11"))
gundata["Teen"] = gundata_agegrp.apply(lambda i: i.count("Teen 12-17"))
gundata["Adult"] = gundata_agegrp.apply(lambda i: i.count("Adult 18+"))


# In[ ]:


# Written generic function - needs bit fine tuning , not used here
def create_extra_cols(generic_series):   
   i = 0;
   for r in range(generic_series.count()):       
        lst = pd.Series(generic_series[r], dtype="category").cat.categories.tolist()      
        for s in range(len(lst)):            
            labelname = lst[s];            
            gundata.loc[i,labelname] = pd.Series(generic_series[r], dtype="category").value_counts()[labelname]            
        i = i + 1        


# In[ ]:


# Post feature engineering is done , lets now begin with data visualization in python
# Distinct state names having crime scenarios
gundata.state.unique()


# In[ ]:


# Distinct city names having crime scenarios
gundata.city_or_county.unique()


# In[ ]:


# Let us study how many people were killed in incidents - statewise 
data_state_killed=gundata.groupby(gundata["state"]).apply(lambda x: pd.Series(dict(state_killed=x.n_killed.sum())))


# In[ ]:


data_state_killed


# In[ ]:


data_state_killed_plot= sns.barplot(x=data_state_killed.index, y=data_state_killed.state_killed, data=data_state_killed,label="Victims killed per state")
data_state_killed_plot.set_xticklabels(data_state_killed_plot.get_xticklabels(),rotation=90)
data_state_killed_plot.set_xlabel('State')
data_state_killed_plot.set_ylabel('Count')
data_state_killed_plot.set_title("Victims killed per state")


# In[ ]:


# Let us see how many people were injured in incidents - statewise
data_state_injury=gundata.groupby(gundata["state"]).apply(lambda x: pd.Series(dict(state_injury=x.n_injured.sum())))


# In[ ]:


data_state_injury


# In[ ]:


data_state_injured_plot= sns.pointplot(x=data_state_injury.index, y=data_state_injury.state_injury, data=data_state_injury,label="Victims injured per state")
data_state_injured_plot.set_xticklabels(data_state_injured_plot.get_xticklabels(),rotation=90)
data_state_killed_plot.set_xlabel('State')
data_state_killed_plot.set_ylabel('Count')
data_state_injured_plot.set_title("Victims injured per state")


# In[ ]:


# Top 10 State with highest no of incidents 
top_10_state = gundata['state'].value_counts().keys().tolist()[0:9]
top_10_values = gundata['state'].value_counts().tolist()[0:9]
top_10_state_plot= sns.barplot(x=top_10_state, y=top_10_values, label="Top 10 states with highest number of incidents")
top_10_state_plot.set_xlabel('State')
top_10_state_plot.set_ylabel('Count')

del top_10_state,top_10_values


# In[ ]:


# We can also use pie chart to understand the same
state_crime = gundata['state'].value_counts().head(30)
state_crime
plt.pie(state_crime, labels=state_crime.index,autopct='%1.1f%%', shadow=True)


# In[ ]:


# Top 10 Cities with highest no of incidents 
top_10_city = gundata['city_or_county'].value_counts().keys().tolist()[0:9]
top_10_city
top_10_values = gundata['city_or_county'].value_counts().tolist()[0:9]
top_10_city_plot= sns.barplot(x=top_10_city, y=top_10_values, label="Top 10 cities with highest number of incidents")
top_10_state_plot.set_xlabel('City')
top_10_state_plot.set_ylabel('Count')


# In[ ]:


#  Year wise - count of criminal incidents
Yearly_incidents_label = gundata['year'].value_counts().keys()
Yearly_incidents_count = gundata['year'].value_counts().tolist()
yearly_incident_plot= sns.pointplot(x =Yearly_incidents_label, y =Yearly_incidents_count, label="Number of incidents happening per year")
yearly_incident_plot.set_xticklabels(yearly_incident_plot.get_xticklabels(),rotation=45)
yearly_incident_plot.set_title("Year wise incident count")
yearly_incident_plot.set_xlabel("Year")
yearly_incident_plot.set_ylabel("Incident Count")


# In[ ]:


# Yearly male/ female ppl involved yearwise
yearly_data = gundata[["male","female"]].groupby(gundata["year"]).sum()


# In[ ]:


yearly_data


# In[ ]:


yearly_data.plot(kind='bar')


# In[ ]:


#  Month wise - count of criminal incidents
monthly_incidents_count = gundata['month'].value_counts()
monthly_incidents_count
monthly_incident_plot= sns.barplot(x =monthly_incidents_count.index, y =monthly_incidents_count, label="Number of incidents happening per month")
monthly_incident_plot.set_xticklabels(monthly_incident_plot.get_xticklabels(),rotation=45)
monthly_incident_plot.set_title("Month wise incident count")
monthly_incident_plot.set_xlabel("Month")
monthly_incident_plot.set_ylabel("Incident Count")


# In[ ]:


# To understand the overall proportion in comparison for ppl killed vs ppl injured , we draw density plot
yearly_data_killed_injured = gundata[["n_killed","n_injured"]].groupby(gundata["year"]).sum()
yearly_data_killed_injured
p1=sns.kdeplot(yearly_data_killed_injured['n_killed'], shade =True , color="r" )
p1=sns.kdeplot(yearly_data_killed_injured['n_injured'], shade =True, color="b" )


# In[ ]:


# We draw barchart for understanding what age group was involved in higher proportions yearwise
age_grp_data = gundata[["Child","Teen","Adult"]].groupby(gundata["year"]).sum()
age_grp_data.plot(kind='bar')


# In[ ]:


# Create joint plot
g = sns.jointplot("male","n_killed",gundata , dropna=True , kind="scatter" ,color = "r" , edgecolor="black")


# In[ ]:


g = sns.jointplot("female","n_killed",gundata , dropna=True , kind="scatter" ,color = "g" , edgecolor="black")

