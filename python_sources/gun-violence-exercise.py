#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for base plotting
import seaborn as sns #easier plotting.
import matplotlib as mpl

import os
#Show graphs  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


gunviolence_data = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
gunviolence_data.head()


# In[ ]:


gunviolence_data.shape


# In[ ]:


gunviolence_data.describe()


# In[ ]:


gunviolence_data.info()


# In[ ]:


gunviolence_data.dtypes 


# In[ ]:


gunviolence_data.columns.values


# In[ ]:


gunviolence_data.columns


# In[ ]:


#unique states
gunviolence_data['state'].unique()


# In[ ]:


#No of states
state_vs_crimecount=gunviolence_data['state'].value_counts().head(10)
state_vs_crimecount


# In[ ]:


plt.pie(state_vs_crimecount,labels=state_vs_crimecount.index,shadow=True)
plt.title("Top 10 High Crime Rate State")
plt.axis("equal")


# In[ ]:


state_vs_crimecount=sns.countplot(x=gunviolence_data["state"],data=gunviolence_data,order=gunviolence_data["state"].value_counts().index)
state_vs_crimecount.set_xticklabels(state_vs_crimecount.get_xticklabels(),rotation=90)
state_vs_crimecount.set_title("State Vs Crime Rate")


# In[ ]:


#split datetime columns to its components
gunviolence_data['date']=pd.to_datetime(gunviolence_data['date'])
gunviolence_data.dtypes


# In[ ]:


gunviolence_data['date_month'] = gunviolence_data['date'].dt.month
gunviolence_data['date_year'] = gunviolence_data['date'].dt.year
gunviolence_data['date_day'] = gunviolence_data['date'].dt.weekday
gunviolence_data['date_week'] = gunviolence_data['date'].dt.week
gunviolence_data.shape


# In[ ]:


gunviolence_data.head()


# In[ ]:


#Handle missing data
gunviolence_nandata_sum=gunviolence_data.isna().sum()
gunviolence_nandata_count=gunviolence_data.isna().count()
gunviolence_nandata=pd.concat([gunviolence_nandata_sum,gunviolence_nandata_count],axis=1)
gunviolence_nandata


# In[ ]:


#Remove data not required 
gunviolence_data.drop([
    "incident_url",
    "source_url",
    "incident_url_fields_missing",
    "latitude",
    "location_description",
    "longitude",
    "participant_relationship"
], axis=1, inplace=True)


# In[ ]:


gunviolence_data.head()


# In[ ]:


city_vs_crimerate=gunviolence_data['city_or_county'].value_counts().head(10)
city_vs_crimerate=sns.barplot(x=city_vs_crimerate.index,y=city_vs_crimerate)
city_vs_crimerate.set_xticklabels(city_vs_crimerate.get_xticklabels(),rotation=45)
city_vs_crimerate.set_title("Top 10 High Crime Rate Cities")


# In[ ]:


#State Vs Killed
state_vs_killed=gunviolence_data.groupby(gunviolence_data["state"]).apply(lambda x: pd.Series(dict(No_Killed=x.n_killed.sum())))
state_vs_killed


# In[ ]:


#State Vs No of People Killed
#no NAN in State and n_killed
sns.boxplot('state','n_killed',data=gunviolence_data)


# In[ ]:


#State Vs Injured
state_vs_injured=gunviolence_data.groupby(gunviolence_data["state"]).apply(lambda x: pd.Series(dict(No_Injured=x.n_injured.sum())))
state_vs_injured


# In[ ]:


#State vs injured Plot
sns.boxplot('state','n_injured',data=gunviolence_data)


# In[ ]:


#Yearly no of Killed 

yearly_vs_killed=gunviolence_data.groupby(gunviolence_data["date_year"]).apply(lambda x: pd.Series(dict(No_Killed=x.n_killed.sum())))
yearly_vs_killed_plot=sns.pointplot(x=yearly_vs_killed.index, y=yearly_vs_killed.No_Killed, data=yearly_vs_killed,label="yearly_vs_killed")
yearly_vs_killed


# In[ ]:


#Yearly No of Injured
yearly_vs_injured=gunviolence_data.groupby(gunviolence_data["date_year"]).apply(lambda x: pd.Series(dict(No_Injured=x.n_injured.sum())))
yearly_vs_injured_plot=sns.pointplot(x=yearly_vs_injured.index, y=yearly_vs_injured.No_Injured, data=yearly_vs_injured,label="yearly_vs_injured")
yearly_vs_injured


# In[ ]:


#Monthly Killed People
monthly_vs_killed=gunviolence_data.groupby(gunviolence_data["date_month"]).apply(lambda x: pd.Series(dict(No_Killed=x.n_killed.sum())))

monthly_vs_killed_plot=sns.pointplot(x=monthly_vs_killed.index, y=monthly_vs_killed.No_Killed, data=monthly_vs_killed,label="monthly_vs_killed")
monthly_vs_killed


# In[ ]:


#Monthly Killed People
monthly_vs_injured=gunviolence_data.groupby(gunviolence_data["date_month"]).apply(lambda x: pd.Series(dict(No_Injured=x.n_injured.sum())))

monthly_vs_injured_plot=sns.pointplot(x=monthly_vs_injured.index, y=monthly_vs_injured.No_Injured, data=monthly_vs_injured,label="monthly_vs_injured")
monthly_vs_injured


# In[ ]:


#unique gun type
gunviolence_data['gun_type'].unique()


# In[ ]:


#handle nan values
gunviolence_data['gun_type'] = gunviolence_data['gun_type'].fillna(value="0::Unknown")
gunviolence_data['gun_stolen']=gunviolence_data['gun_stolen'].fillna(value="0::Unknown")
gunviolence_data['n_guns_involved'] = gunviolence_data['n_guns_involved'].fillna(value=0)


# In[ ]:


#Split participant gender
gunviolence_data["participant_gender"] = gunviolence_data["participant_gender"].fillna("0::Unknown")
    
def gender(n) :                    
    gender_rows = []               
    gender_row = str(n).split("||")    
    for i in gender_row :              
        g_row = str(i).split("::")  
        if len(g_row) > 1 :         
            gender_rows.append(g_row[1])    

    return gender_rows

gender_series = gunviolence_data.participant_gender.apply(gender)
gunviolence_data["total_participant"] = gender_series.apply(lambda x: len(x))
gunviolence_data["male_participant"] = gender_series.apply(lambda i: i.count("Male"))
gunviolence_data["female_participant"] = gender_series.apply(lambda i: i.count("Female"))
gunviolence_data["unknown_participant"] = gender_series.apply(lambda i: i.count("Unknown"))


# In[ ]:


gunviolence_data.head(3)


# In[ ]:


#Number of Male vs killed
sns.jointplot("male_participant",
             "n_killed",
             gunviolence_data,
             kind="scatter",
             s=100, color="m",edgecolor="red",linewidth=2)


# In[ ]:


#Number of Male vs injured
sns.jointplot("male_participant",
             "n_injured",
             gunviolence_data,
             kind="scatter",
             s=100, color="m",edgecolor="red",linewidth=2)


# In[ ]:


#Number of person killed vs incident
sns.jointplot("incident_id",
             "n_killed",
             gunviolence_data,
             kind="scatter",
             s=100, color="m",edgecolor="red",linewidth=2)


# In[ ]:


#Number of Person Injured vs incident_id 
sns.jointplot("incident_id",
             "n_killed",
             gunviolence_data,
             kind="scatter",
             s=100, color="m",edgecolor="red",linewidth=2)


# In[ ]:


#Density plot for yearly incident 
yearly_impact = gunviolence_data[["n_killed", "n_injured"]].groupby(gunviolence_data["date_year"]).sum()
density_plot=sns.kdeplot(yearly_impact['n_killed'],shade=True,color="r")
density_plot=sns.kdeplot(yearly_impact['n_injured'],shade=True,color="b")
del(yearly_impact)


# In[ ]:


#Density plot for yearly participant 
yearly_participant = gunviolence_data[["total_participant","male_participant", "female_participant"]].groupby(gunviolence_data["date_year"]).sum()
density_plot=sns.kdeplot(yearly_participant['total_participant'],shade=True,color="r")
density_plot=sns.kdeplot(yearly_participant['male_participant'],shade=True,color="b")
density_plot=sns.kdeplot(yearly_participant['female_participant'],shade=True,color="b")
del(yearly_participant)


# In[ ]:


#Histogram for n_guns_involved yearly 
yearly_guns_involved = gunviolence_data[["n_guns_involved"]].groupby(gunviolence_data["date_year"]).count()
yearly_guns_involved.plot.barh()


# In[ ]:


#Histogram for n_guns_involved monthly 
monthly_guns_involved = gunviolence_data[["n_guns_involved"]].groupby(gunviolence_data["date_month"]).count()
monthly_guns_involved.plot.barh()


# In[ ]:


#Violin Plot for gender impacted yearly
impact_total_gender = gunviolence_data[["total_participant","male_participant","female_participant","unknown_participant"]].groupby(gunviolence_data["date_year"]).sum()
print(impact_total_gender)
impact_total_gender_plot=sns.violinplot(data=impact_total_gender,split=True,inner="box")


# In[ ]:


#Violin Plot for injured,killed impacted yearly
impact_numbers = gunviolence_data[["n_killed","n_injured"]].groupby(gunviolence_data["date_year"]).sum()
print(impact_numbers)
impact_numbers=sns.violinplot(data=impact_numbers,split=True,inner="quartile")

