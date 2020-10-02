#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[17]:


#importing the dataset
dataset=pd.read_csv("../input/311_Service_Requests_from_2010_to_Present.csv")
pd.options.display.max_columns = None
dataset.head(4)


# In[18]:


dataset.info()


# In[ ]:


dataset.dtypes


# In[ ]:


dataset['Created Date']=pd.to_datetime(dataset['Created Date'])
dataset['Closed Date']=pd.to_datetime(dataset['Closed Date'])


# In[ ]:


dataset.head(5)


# In[ ]:


dataset['Request_Closing_Time'] = dataset['Closed Date']-dataset['Created Date']


# In[ ]:


dataset.head(5)


# In[ ]:


dataset.columns


# In[ ]:


dataset.drop(['Incident Address', 'Street Name', 'Cross Street 1', 'Cross Street 2',
       'Intersection Street 1', 'Intersection Street 2','Resolution Description', 
     'Resolution Action Updated Date','Community Board','X Coordinate (State Plane)','School or Citywide Complaint',
    'Vehicle Type','Taxi Company Borough','Taxi Pick Up Location','Garage Lot Name','School Name', 'School Number', 
              'School Region', 'School Code','School Phone Number', 'School Address', 'School City', 'School State',
       'School Zip', 'School Not Found','Ferry Direction', 'Ferry Terminal Name','Unique Key','Bridge Highway Name',
       'Bridge Highway Direction', 'Road Ramp', 'Bridge Highway Segment'],axis=1,inplace=True)


# In[ ]:


dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


dataset.isna().sum()


# In[ ]:


#dropping created and closed date
dataset.drop(['Closed Date','Created Date'],axis=1,inplace=True)


# In[ ]:


dataset.head(5)


# In[ ]:


#dealing with missing values
dataset.isna().sum()


# In[ ]:


dataset['Agency'].value_counts()
dataset['Agency Name'].value_counts()
sns.countplot(dataset['Agency Name'])


# In[ ]:


dataset['Complaint Type'].value_counts().head()
plot=sns.countplot(dataset['Complaint Type'])
plot.set_xticklabels(plot.get_xticklabels(),rotation=90)


# In[ ]:


dataset['Descriptor'].isna().sum()


# In[ ]:


dataset['Descriptor'].describe()


# In[ ]:


dataset['Descriptor'].value_counts().head(5)


# In[ ]:


plot2=sns.countplot(dataset['Descriptor'])
plot2.set_xticklabels(plot.get_xticklabels(),rotation=90)


# In[ ]:


dataset['Location Type'].isna().sum()


# In[ ]:


dataset['Location Type'].value_counts().head()


# In[ ]:


dataset['Location Type'].fillna(value='Street/Sidewalk',inplace =True)
plot3=sns.countplot(dataset['Location Type'])
plot3.set_xticklabels(plot3.get_xticklabels(),rotation=90)


# In[ ]:


dataset['Incident Zip'].value_counts().head()
dataset['Incident Zip'].isna().sum()
dataset['Incident Zip'].fillna(value=11385,inplace=True)


# In[ ]:


dataset['Address Type'].value_counts()
dataset['Address Type'].fillna(value='Address',inplace=True)
sns.countplot(dataset['Address Type'])


# In[ ]:


dataset.drop(['Latitude', 'Longitude','Location','Y Coordinate (State Plane)','Landmark'],axis=1,inplace=True)


# In[ ]:


dataset.isna().sum()


# In[ ]:


dataset['City'].value_counts().head()
dataset['Facility Type'].value_counts().head()


# In[ ]:


dataset['City'].fillna(value='BROOKLYN',inplace=True)
dataset['City'].value_counts().head()
plot4=sns.countplot(x=dataset['City'])
plot4.set_xticklabels(plot.get_xticklabels(),rotation=90)


# In[ ]:


dataset.isna().sum()


# In[ ]:


dataset['Request_Closing_Time'].head()


# In[ ]:


dataset['Request_Closing_Time'].fillna(value=dataset['Request_Closing_Time'].mean(),inplace=True)


# In[ ]:


dataset['Request_Closing_Time'].isna().sum()


# In[ ]:


dataset['Request_Closing_Time'].dtypes


# In[ ]:


dataset.head(10)


# In[ ]:


dataset['Status'].value_counts()


# In[ ]:


sns.countplot(dataset['Status'])


# In[ ]:


#bivariate analysis
#the most common complaint
dataset['Complaint Type'].value_counts().head(6)


# In[ ]:


dataset.head(3)


# In[ ]:


desc=dataset.groupby(by='Complaint Type')['Descriptor'].agg('count')
desc


# In[ ]:


#City with their status
dataset.loc[dataset['City']=='NEW YORK',]['Borough'].value_counts()


# In[ ]:


#Newyork city has how many boroughs and whats their status 
sns.countplot(x=dataset.loc[dataset['City']=='NEW YORK',]['Borough'],hue='Status',data=dataset)


# In[ ]:


#Newyork city has max complaints of which complaint type?
dataset.loc[dataset['City']=='NEW YORK',:]['Complaint Type'].value_counts()


# In[ ]:


#Countplot to show Newyork city has max complaints of which complaint type?
plot=sns.countplot(x=dataset.loc[dataset['City']=='NEW YORK',:]['Complaint Type'])
plot.set_xticklabels(plot.get_xticklabels(),rotation = 90)


# In[ ]:


#Avg time taken to solve a case in Newyork city
dataset.loc[(dataset['City']=='NEW YORK')&(dataset['Status']=='Closed'),:]['Request_Closing_Time'].mean()


# In[ ]:


dataset.loc[(dataset['City']=='NEW YORK')&(dataset['Status']=='Closed'),:]['Request_Closing_Time'].std()


# In[ ]:


dataset['Borough'].value_counts()


# In[ ]:


dataset['Location Type'].value_counts()


# In[ ]:


#Top Location type and their countplot with hues='Borough'
sns.countplot(dataset.loc[dataset['Location Type'].isin(['Street/Sidewalk','Store/Commercial','Club/Bar/Restaurant'])]
              ['Location Type'],data=dataset,hue='Borough')


# In[ ]:


import datetime
dataset['year'] = pd.DatetimeIndex(dataset['Due Date']).year
dataset.head()


# In[ ]:


sns.countplot(dataset['year'],hue='Borough',data=dataset)


# In[ ]:


dataset['Location Type'].value_counts()


# In[ ]:


#Display the complaint type and city together
dataset[['Complaint Type','City']].head()


# In[ ]:


#Find the top 10 complaint types 
dataset['Complaint Type'].value_counts()[0:10,]


# In[ ]:


#Plot a bar graph of count vs. complaint types
plot3=sns.countplot(dataset['Complaint Type'])
plot3.set_xticklabels(plot3.get_xticklabels(),rotation =90)


# In[ ]:


#Display the major complaint types and their count
#top 5 complaint types
series=dataset['Complaint Type'].value_counts()[0:5,]
series.nlargest().index


# In[ ]:


#graph
plot4=sns.barplot(x=series.nlargest().index,y=series.nlargest().values)
plot4.set_xticklabels(plot3.get_xticklabels(),rotation =90)

