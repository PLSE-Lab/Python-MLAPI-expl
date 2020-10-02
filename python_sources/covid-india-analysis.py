#!/usr/bin/env python
# coding: utf-8

# **1. Effect of Lockdown in India** 
# 
# Lockdown in India proved to be effective as it helped India to flatten the curve of positive cases.
# The Below graph shows that while during the Lockdown the testing capacity increased sharply, the more tests helped to lower the stiffness of positive case percentage curve.
# 
# **2. Effect of More Tests** 
# 
# As the Graph shows the more tests helped India to make its positive cases pegged at around 4-5%
# 

# In[ ]:


# 1.2 Display multiple outputs from a jupyter cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


get_ipython().run_line_magic('reset', '-f')
# 1.0 For data manipulation
import numpy as np
import pandas as pd
# 1.1 For plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os


# In[ ]:


testdetail = pd.read_csv("../input/covid19-in-india/ICMRTestingDetails.csv",parse_dates=['DateTime'],dayfirst=True)
testdetail.drop(testdetail[testdetail['TotalIndividualsTested'].isna() & testdetail['TotalPositiveCases'].isna()].index,inplace=True)
testdetail.loc[testdetail['TotalIndividualsTested'].isna()==True,'TotalIndividualsTested']=testdetail.loc[testdetail['TotalIndividualsTested'].isna()==True,'TotalSamplesTested']
testdetail.drop(testdetail[testdetail['TotalPositiveCases'].isna()].index,inplace=True)


# In[ ]:


testdetail['PositivePercentage'] = testdetail['TotalPositiveCases']/testdetail['TotalIndividualsTested']*100
testdetail['Date'] = testdetail['DateTime'].dt.date.apply(lambda x: x.strftime('%Y-%m-%d'))


# In[ ]:


testdetail["lockdown"] = pd.cut(
                       testdetail['DateTime'],
                       bins= [pd.datetime(year=2020, month=3, day=1),pd.datetime(year=2020, month=3, day=25),pd.datetime(year=2020, month=4, day=14),
                              pd.datetime(year=2020, month=5, day=3)],
                       labels= ["No-Lockdown", "Lockdown-1", "Lockdown-2"]
                      )


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10)) 
sns.barplot(x = 'Date',
            y = 'TotalIndividualsTested',
            hue = 'lockdown',   
            ci = 95,
            data =testdetail)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10)) 
sns.barplot(x = 'Date',
            y = 'PositivePercentage',
            hue = 'lockdown',   
            ci = 95,
            data =testdetail,
            ax = ax)
ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45, ha='right')


# In[ ]:


plt.plot('TotalIndividualsTested','PositivePercentage','bo-',data = testdetail)
plt.title("Line Plot for Testing vs PositivePercentage")
plt.xlabel("TotalIndividualsTested")
plt.ylabel("PositivePercentage")


# In[ ]:


#Data Cleansing
Stwisetestdetail = pd.read_csv("../input/covid19-in-india/StatewiseTestingDetails.csv",parse_dates=['Date'])
Stwisetestdetail.drop(Stwisetestdetail[Stwisetestdetail['Date']=='2020-02-16'].index,inplace=True)
Stwisetestdetail.drop(Stwisetestdetail[Stwisetestdetail['Negative'].isna() & Stwisetestdetail['Positive'].isna()].index,inplace=True)
Stwisetestdetail.loc[Stwisetestdetail['Negative'].isna(),'Negative'] = Stwisetestdetail.loc[Stwisetestdetail['Negative'].isna(),'TotalSamples']-Stwisetestdetail.loc[Stwisetestdetail['Negative'].isna(),'Positive']
Stwisetestdetail.loc[Stwisetestdetail['Positive'].isna(),'Positive'] = Stwisetestdetail.loc[Stwisetestdetail['Positive'].isna(),'TotalSamples']-Stwisetestdetail.loc[Stwisetestdetail['Positive'].isna(),'Negative']


# In[ ]:


stateDategrouped = Stwisetestdetail.groupby([Stwisetestdetail['Date'].dt.date, 'State'])
stateDateStatus = stateDategrouped[['TotalSamples','Negative','Positive']].sum()
stateDateStatus = stateDateStatus.reset_index()
stateDateStatus['Date'] = pd.to_datetime(stateDateStatus['Date'])
stateDateStatus['TestingMonth'] = stateDateStatus['Date'].dt.month.map({
                                    4: 'April',
                                    5: 'May'
                                    }
                                )
stateDateStatus['TestingDay'] = stateDateStatus['Date'].dt.day
stateDateStatus['PositivePercent'] = (stateDateStatus['Positive']/stateDateStatus['TotalSamples'])*100
stateDateStatus['Date'] = stateDateStatus['Date'].dt.date


# In[ ]:


#pivot State data Datewise 
state_date = stateDateStatus.pivot(index='Date', columns='State', values='PositivePercent')


# **3. State Performance** 
# 
# As the heatmap shows for some states like TamilNadu,Delhi and Chandigarh the percentage of positive cases were more initially while for other states like Maharastra and Gujarat it was lower initially but increased as the day passes as Mumbai and Ahmedabad proved to be major cities in these two states .For Northeast states,the positive case percentage was always lower.  
# 
# The Statewise position of positive cases percentage for the month of April and May has also been shown for all the states.

# In[ ]:


fig, ax = plt.subplots(figsize=(8,8)) 
sns.heatmap(state_date.fillna(0),cmap="YlOrRd",xticklabels=True,ax=ax)
ax.invert_yaxis()


# In[ ]:


sns.catplot(x = 'TestingDay',
            y = 'PositivePercent',
            row = 'State',
            col = 'TestingMonth',
            kind = 'bar',
            data =stateDateStatus)
#sns.relplot(x='TestingMonth',y='PositivePercent',row = 'State',kind = 'scatter',data=stateDateStatus)


# In[ ]:


#Loading Population data for States
census = pd.read_csv("../input/covid19-in-india/population_india_census2011.csv")
census['AreaInNumbers'] = census['Area'].apply(lambda x : x.split("km2")[0].replace(",", ""))
census['DensityInNumbers'] = census['Density'].apply(lambda x : x.split("/km2")[0].replace(",", ""))
census['AreaInNumbers']=census['AreaInNumbers'].astype('int32')
census['DensityInNumbers']=census['DensityInNumbers'].astype('float')


# In[ ]:


#Latest Covid -19 status of Indian States
stateWiseCovid = Stwisetestdetail.groupby('State')[['TotalSamples','Negative','Positive']].max()


# In[ ]:


#Joining State census data with Covid-19 data of States
stateCensusCovid = pd.merge(census,stateWiseCovid,left_on = 'State / Union Territory',right_on = 'State', how='left')[['State / Union Territory','Population','Rural population','Urban population','Gender Ratio', 'AreaInNumbers',
       'DensityInNumbers','TotalSamples', 'Negative', 'Positive']]
stateCensusCovid = stateCensusCovid.fillna(0)
stateCensusCovid['PositiveinPercent'] = stateCensusCovid['Positive']/stateCensusCovid['TotalSamples']*100
stateCensusCovid['UrbanPopulationInPercent'] = stateCensusCovid['Urban population']/stateCensusCovid['Population']*100
stateCensusCovid['RuralPopulationInPercent'] = 100 -stateCensusCovid['UrbanPopulationInPercent']


# In[ ]:


#Finding Relation between Density of Population and no. of positive cases.
sns.jointplot(x=stateCensusCovid.DensityInNumbers,y='Positive',xlim=(0,2000),ylim=(0,5000),data = stateCensusCovid.fillna(0))


# **4. State Performance (Population Density vs Covid-19 cases)**
# 
# The maximum cases are concentrated in some of the States having  population density more than 500 per SqkM.
# Maximum no. of Staes having Urban population less between 20-40% and for these states positive case percentage is less than 2%

# In[ ]:





# In[ ]:


#sns.jointplot(x=stateCensusCovid['Urban population']/stateCensusCovid['Population'],y='Positive',data = stateCensusCovid.fillna(0),kind="kde")
sns.jointplot(x='UrbanPopulationInPercent',y='PositiveinPercent',xlim=(0,100),data = stateCensusCovid,kind="kde")


# In[ ]:


sns.jointplot(x='DensityInNumbers',y='PositiveinPercent',xlim=(0,1500),data = stateCensusCovid,kind="hex")


# In[ ]:


#Loading State health-Infrastructure data.
statehealthInfra = pd.read_csv("../input/covid19-in-india/HospitalBedsIndia.csv")


# In[ ]:


statehealthInfra['TotalHospitals'] = statehealthInfra['NumRuralHospitals_NHP18'] + statehealthInfra['NumUrbanHospitals_NHP18']
statehealthInfra['TotalBeds'] = statehealthInfra['NumRuralBeds_NHP18'] + statehealthInfra['NumUrbanBeds_NHP18']


# In[ ]:


#Merging Health-Infrastructure data with census and Covid-19 cases
stateCensusCovid = pd.merge(stateCensusCovid,statehealthInfra,left_on = 'State / Union Territory',right_on = 'State/UT', how='left')[['State / Union Territory', 'Population', 'Rural population',
       'Urban population', 'Gender Ratio', 'AreaInNumbers', 'DensityInNumbers',
       'TotalSamples', 'Negative', 'Positive', 'PositiveinPercent',
       'UrbanPopulationInPercent', 'RuralPopulationInPercent','TotalPublicHealthFacilities_HMIS','NumPublicBeds_HMIS', 'NumRuralHospitals_NHP18', 'NumRuralBeds_NHP18',
       'NumUrbanHospitals_NHP18', 'NumUrbanBeds_NHP18', 'TotalHospitals',
       'TotalBeds']]


# In[ ]:


stateCensusCovid['RuralPersonPerBed'] = stateCensusCovid['Rural population']/stateCensusCovid['NumRuralBeds_NHP18']
stateCensusCovid['UrbanPersonPerBed'] = stateCensusCovid['Urban population']/stateCensusCovid['NumUrbanBeds_NHP18']
stateCensusCovid['PersonPerBed'] = stateCensusCovid['Population']/stateCensusCovid['TotalBeds']
stateCensusCovid['PersonPerPublicBed'] = stateCensusCovid['Population']/stateCensusCovid['NumPublicBeds_HMIS']


# **5. State Health Infrastructure Status**
# 
# The Northern states like U.P ,Bihar and Jharkhand having low health Infrastructure having more than 3000 persons per bed,spreading cases in these states may arise a big challenge for the local government.
# 

# In[ ]:


#sns.jointplot(x='Positive',y=stateCensusCovid['Population']/stateCensusCovid['NumPublicBeds_HMIS'],xlim=(0,3000),ylim=(0,3000),data = stateCensusCovid,kind="kde")
fig, ax = plt.subplots(figsize=(10,10)) 
sns.barplot(x = 'PersonPerPublicBed',
            y = 'State / Union Territory',
           data =stateCensusCovid)


# In[ ]:


sns.jointplot(x='UrbanPersonPerBed',y='PositiveinPercent',data = stateCensusCovid,kind="kde")


# In[ ]:


sns.jointplot(x='TotalBeds',y='Positive',xlim=(0,30000),ylim=(0,10000),data = stateCensusCovid,kind="kde")


# In[ ]:




