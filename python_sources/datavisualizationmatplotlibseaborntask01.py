#!/usr/bin/env python
# coding: utf-8

# # $$Introduction$$
# 
# ## Covid-19 22 January 2020 to 11 April 2020 data analysis
# 
# ###  Data Information
# - Number of rows 25353
# - Number of columns 6
#  - column_Name(Id)   Data_Type(int64) Records(25353)
#  - column_Name(Province_State)   Data_Type(object) Records(10773 )
#  - column_Name(Country_Region)   Data_Type(object) Records(25353)
#  - column_Name(Date)   Data_Type(datetime64[ns]) Records(25353)
#  - column_Name(ConfirmedCases)   Data_Type(float64) Records(25353)
#  - column_Name(Fatalities)   Data_Type(float64) Records(25353)

# ## Necessery Liberary Import

# In[ ]:


import pandas as pd # Load data
import numpy as np # Scientific Computing
import seaborn as sns # Data Visualization
import matplotlib.pyplot as plt # Data Visualization
import warnings # Ignore Warnings
warnings.filterwarnings("ignore")
sns.set() # Set Graphs Background


# ## Load Data

# In[ ]:


data = pd.read_csv('../input/datafile/train (3).csv')
data.head()


# ## Information Of Data

# In[ ]:


data.info()


# ## Unique Country

# In[ ]:


data['Country_Region'].unique()


# ## Unique country Count

# In[ ]:


data['Country_Region'].nunique()


# - There have 184 Unique Country present

# ## Barplot For All Data

# In[ ]:


plt.figure(figsize=(40,10))  # For Figure Resize
sns.barplot(x='Country_Region',y='ConfirmedCases', data=data)
plt.xlabel('Country_Region',fontsize = 35)
plt.ylabel('ConfirmedCases',fontsize = 35)
plt.xticks(rotation=90)  #For X label Value_Name rotation
plt.show() # Show The Plotfontsize = 25


# ### Insights of Barplot For All Data
# - This barplot represents Country_Region vs ConfirmedCases
# - Italy First number of ConfirmedCases 
# - Spain Second number of ConfirmedCases
# - Germany Third number of ConfirmedCases 
# - Then Iran,Korea, switzerland, Turkey also

# ## Box Plot

# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(data=data)
plt.show()


# - The Maximum ConfirmedCases & Fatalities Data Out of 75% quantile.

# ## Scatter Plot Date VS ConfirmedCases with Country

# In[ ]:


plt.figure(figsize=(20,8))
plt.scatter(data['Date'],data['ConfirmedCases'])
plt.title('22 January - 11 April Total ConfirmedCases By Date With Country', fontsize=25)
plt.xlabel('Date',fontsize=25)
plt.ylabel('ConfirmedCases',fontsize=25)
plt.xticks(rotation=90)
plt.show()


# - Day By Day Increase The ConfirmedCases

# ## Scatter Plot Date VS Deaths With Country

# In[ ]:


plt.figure(figsize=(20,8))
plt.scatter(data['Date'],data['Fatalities'])
plt.title('22 January - 11 April Total Deaths By Date With Country', fontsize=25)
plt.xlabel('Date', fontsize=25)
plt.ylabel('Deaths',fontsize=25)
plt.xticks(rotation=90)
plt.show()


# - Day By Day Increase The Deaths

# ## Country Records Count

# In[ ]:


data['Country_Region'].value_counts()


# ## Maximum & Minimum Date 

# In[ ]:


print(data['Date'].min())
print(data['Date'].max())


# - Start date 22 january 2020
# - End date 11 april 2020

# ## Group Country & Select Largest 15 Country by ConfirmedCases

# In[ ]:


data_15 = data.groupby('Country_Region', as_index=False)['ConfirmedCases','Fatalities'].sum()
data_15 = data_15.nlargest(15,'ConfirmedCases')
data_15


# ## Create Active/Recover Column

# In[ ]:


data_15['Active/Recover'] = data_15['ConfirmedCases'] - data_15['Fatalities']
data_15


# ## Bar Plot ConfirmedCases For New Data

# In[ ]:


plt.figure(figsize=(15,8))    # For Figure Resize
sns.barplot(x='Country_Region',y='ConfirmedCases', data=data_15) # For Bar Plot
plt.title("22 January - 11 April Total ConfirmedCases By Country",fontsize = 25)   # For Title For the Graph 
plt.xlabel('Country_Region',fontsize = 25)   # For X-axis Name
plt.ylabel('ConfirmedCases',fontsize = 25)   # For Y-axis Name
plt.xticks(rotation=70)   # For X label Value_Name rotation
plt.show()    # Show The Plot


# - The Highest Number of ConfirmedCases in US
# - The lowest Number of ConfirmedCases in Austria

# ## Bar Plot For Fatalities For New Data

# In[ ]:


plt.figure(figsize=(15,8))  # For Figure Resize
sns.barplot(x='Country_Region',y='Fatalities', data=data_15)  # Show The Plot
plt.title("22 January - 11 April Total Deaths By Country",fontsize = 25)  # For Title For the Graph 
plt.xlabel('Country_Region',fontsize = 25)  # For X-axis Name
plt.ylabel('Fatalities',fontsize = 25)   # For Y-axis Name
plt.xticks(rotation=70)   # For X label Value_Name rotation
plt.show()   # Show The Plot


# - The Highest Number of Deaths in Italy
# - The lowest Number of Deaths in Austria

# ## Bar Plot For Active/Recover For New Data

# In[ ]:


plt.figure(figsize=(15,8))  # For Figure Resize
sns.barplot(x='Country_Region',y='Active/Recover', data=data_15)  # Show The Plot
plt.title("22 January - 11 April Total Active/Recover By Country",fontsize = 25)  # For Title For the Graph 
plt.xlabel('Country_Region',fontsize = 25)  # For X-axis Name
plt.ylabel('Active/Recover',fontsize = 25)   # For Y-axis Name
plt.xticks(rotation=70)   # For X label Value_Name rotation
plt.show()   # Show The Plot


# - The Highest Number of Active/Recover in US
# - The lowest Number of Active/Recover in Austria

# ## Country_Region Vs ConfirmedCases & Deaths for 15 Country

# In[ ]:


plt.figure(figsize=(15,8))
sns.barplot(x='Country_Region',y='ConfirmedCases', data=data_15, color='green',label='ConfirmedCases')
sns.barplot(x='Country_Region',y='Fatalities', data=data_15,color='red', label='Deaths')
plt.title("22 January - 11 April Total Country_Region Vs ConfirmedCases & Deaths")
plt.xlabel('Country_Region')
plt.ylabel('ConfirmedCases & Fatalities')
plt.xticks(rotation=70)
plt.legend()
plt.show()


# ## Scatter Plot For Country VS ConfirmedCases For New Data

# In[ ]:


bubbol = np.array(data_15['ConfirmedCases']/1500) # For Bubbol Size
plt.figure(figsize=(15,8))
plt.scatter(data_15['Country_Region'],data_15['ConfirmedCases'],c='green',s=bubbol, alpha=0.6)
plt.title("22 January - 11 April Total ConfirmedCases By Country",fontsize = 20)
plt.xlabel('Country_Region',fontsize = 20)
plt.ylabel('ConfirmedCases',fontsize = 20)
plt.xticks(rotation=70)
plt.show()


# - The Highest Number of ConfirmedCases in US.
# - The lowest Number of ConfirmedCases in Austria.

# ## Scatter Plot For Country VS Active/Recover For New Data

# In[ ]:


bubbol = np.array(data_15['Active/Recover']/1500) # For Bubbol Size
plt.figure(figsize=(15,8))
plt.scatter(data_15['Country_Region'],data_15['Active/Recover'],c='blue',s=bubbol, alpha=0.6)
plt.title("22 January - 11 April Total Active/Recover By Country",fontsize = 20)
plt.xlabel('Country_Region',fontsize = 20)
plt.ylabel('ConfirmedCases',fontsize = 20)
plt.xticks(rotation=70)
plt.show()


# - The Highest Number of Active/Recover in US.
# - The lowest Number of Active/Recover in Austria.

# ## Scatter Plot For Country VS Deaths For New Data

# In[ ]:


bubbol = np.array(data_15['Fatalities']/50) # For Bubbol Size
plt.figure(figsize=(15,8))
plt.scatter(data_15['Country_Region'],data_15['Fatalities'],c='red',s=bubbol, alpha=0.6)
plt.title("22 January - 11 April Total Deaths By Country",fontsize = 25)
plt.xlabel('Country',fontsize = 25)
plt.ylabel('Deaths',fontsize = 25)
plt.xticks(rotation=70)
plt.show()


# - The Highest Number of Deaths in Italy.
# - The lowest Number of Deaths in Austria.

# ## Pie Chart For ConfirmedCases For New Data

# In[ ]:


plt.axis('equal')
plt.pie(data_15['ConfirmedCases'],labels=data_15['Country_Region'], radius=2, autopct='%.0f%%',
        shadow=True)
plt.show()


# - The Highest Number of ConfirmedCases 23% in US & China .
# - The lowest Number of ConfirmedCases 1% in Belgium, Netherlands, Canada & Austria.

# ## Pie Chart For Deaths For New Data

# In[ ]:


plt.axis('equal')
plt.pie(data_15['Fatalities'],labels=data_15['Country_Region'], radius=2, autopct='%.0f%%',
        shadow=True)
plt.show()


# - The Highest Number of Deaths 25% in Italy.
# - The lowest Number of Deaths 0.2% in Austria.

# ## The Date Convert Into YYYY-MM-DD Format

# In[ ]:


data['Date'] = pd.to_datetime(data['Date'])


# ## Create New Dataset For Individual Date

# In[ ]:


data_81 = data.groupby('Date', as_index=False)['ConfirmedCases','Fatalities'].sum()
data_81 = data_81.nlargest(81,'ConfirmedCases')
data_81


# ## Create Active/Recover Column for Individual date

# In[ ]:


data_81['Active/Recover'] = data_81['ConfirmedCases'] - data_81['Fatalities']
data_81


# ## Scatter Plot Date VS ConfirmedCases For New Dataset

# In[ ]:


bubbol = np.array(data_81['ConfirmedCases']/1500) # For Bubbol Size
plt.figure(figsize=(15,8))
plt.scatter(data_81['Date'],data_81['ConfirmedCases'],c='blue',s=bubbol, alpha=0.6)
plt.title("22 January - 11 April Total ConfirmedCases By Date",fontsize=25)
plt.xlabel('Date',fontsize=25)
plt.ylabel('ConfirmedCases',fontsize=25)
plt.show()


# - Day By Day Increase The ConfirmedCases

# In[ ]:





# In[ ]:


bubbol = np.array(data_81['Active/Recover']/1500) # For Bubbol Size
plt.figure(figsize=(15,8))
plt.scatter(data_81['Date'],data_81['Active/Recover'],c='green',s=bubbol, alpha=0.6)
plt.title("22 January - 11 April Total Active/Recover By Date",fontsize = 20)
plt.xlabel('Date',fontsize = 20)
plt.ylabel('Active/Recover',fontsize = 20)
plt.xticks(rotation=70)
plt.show()


# - Day By Day Increase The Active/Recover

# ## Scatter Plot Date VS Deaths For New Dataset

# In[ ]:


bubbol = np.array(data_81['Fatalities']/50) # For Bubbol Size
plt.figure(figsize=(15,8))
plt.scatter(data_81['Date'],data_81['Fatalities'],c='red',s=bubbol, alpha=0.6)
plt.title("22 January - 11 April Total Deaths By Date",fontsize=25)
plt.xlabel('Date',fontsize=25)
plt.ylabel('Deaths',fontsize=25)
plt.show()


# - Day By Day Increase The Deaths

# ## ConfirmedCases & Active/Recover & Deaths By Date

# In[ ]:


plt.figure(figsize=(15,8))
plt.scatter(data_81['Date'],data_81['ConfirmedCases'],c='blue', alpha=0.6, label='ConfirmedCases')
plt.scatter(data_81['Date'],data_81['Active/Recover'],c='green',alpha=0.6, label='Active/Recover')
plt.scatter(data_81['Date'],data_81['Fatalities'],c='red',alpha=0.6, label='Fatalities')
plt.title("Total ConfirmedCases & Active/Recover & Deaths By Date",fontsize=25)
plt.xlabel('Date',fontsize=25)
plt.ylabel('ConfirmedCases Active/Recover & Deaths',fontsize=25)
plt.legend(loc=10)
plt.show()


# - ConfirmedCases & Active/Recover highly increases after 18th March
# - Deaths are increase after 29th March.

# ## Bar Plot Date VS ConfirmedCases For New Dataset

# In[ ]:


fig, ax = plt.subplots(figsize = (20,10))    
fig = sns.barplot(x ="Date", y ="ConfirmedCases", data = data_81)

x_dates = data_81['Date'].dt.strftime('%Y-%m-%d').sort_values()
ax.set_xticklabels(labels=x_dates, rotation=80)
plt.title("22 January - 11 April Total ConfirmedCases By Date",fontsize=25)
plt.xlabel('Date',fontsize=25)
plt.ylabel('ConfirmedCases',fontsize=25)
plt.show()


# - Day By Day Increase The ConfirmedCases.

# ## Bar Plot Date VS Deaths For World

# In[ ]:


fig, ax = plt.subplots(figsize = (20,10))    
fig = sns.barplot(x ="Date", y ="Fatalities", data = data_81)

x_dates = data_81['Date'].dt.strftime('%Y-%m-%d').sort_values()
ax.set_xticklabels(labels=x_dates, rotation=80)
plt.title("22 January - 11 April Total Date VS Deaths",fontsize=25)
plt.xlabel('Date',fontsize=25)
plt.ylabel('Deaths',fontsize=25)
plt.show()


# - Day By Day Increase The Deaths.

# ## Line Plot Date VS ConfirmedCases For World

# In[ ]:


plt.figure(figsize=(15,8))
sns.lineplot(x='Date',y='ConfirmedCases', data=data_81)
plt.title("22 January - 11 April Total ConfirmedCases By Date for world",fontsize=25)
plt.xlabel('Date',fontsize=25)
plt.ylabel('ConfirmedCases',fontsize=25)
plt.show()


# - Day By Day Increase The ConfirmedCases.

# ## Line Plot Date VS Deaths For World

# In[ ]:


plt.figure(figsize=(15,8))
sns.lineplot(x='Date',y='Fatalities', data=data_81)
plt.title("22 January - 11 April Total Deaths By Date for world",fontsize=25)
plt.xlabel('Date',fontsize=25)
plt.ylabel('Deaths',fontsize=25)
plt.show()


# - Day By Day Increase The Number of deaths.

# ## ConfirmedCases & Active/Recover & Deaths By Date for world

# In[ ]:


plt.figure(figsize=(15,8))
sns.lineplot(x='Date',y='ConfirmedCases', data=data_81, label='ConfirmedCases')
sns.lineplot(x='Date',y='Active/Recover', data=data_81, label='Active/Recover')
sns.lineplot(x='Date',y='Fatalities', data=data_81, label='Fatalities')
plt.title("Total ConfirmedCases & Active/Recover & Deaths By Date for world",fontsize=25)
plt.xlabel('Date',fontsize=25)
plt.ylabel('ConfirmedCases & Active/Recover & Deaths',fontsize=25)
plt.legend()
plt.show()


# - ConfirmedCases & Active/Recover highly increases after 18th March
# - Deaths are increase after 25th March.

# ## Create New Dataset For Individual Date & Individual Country

# In[ ]:


data_all = data.groupby(['Date','Country_Region'], as_index=False)['ConfirmedCases','Fatalities'].sum()
data_all


# ## Create Active/Recover Column for Country &  date

# In[ ]:


data_all['Active/Recover'] = data_all['ConfirmedCases'] - data_all['Fatalities']
data_all


# ## Select All United State Data From New Dataset

# In[ ]:


data_usa = data_all.query("Country_Region=='US'")
data_usa


# ## Scatter Plot Date VS ConfirmedCases For United State

# In[ ]:


bubbol = np.array(data_usa['ConfirmedCases']/500) # For Bubbol Size
plt.figure(figsize=(15,8))
plt.scatter(data_usa['Date'],data_usa['ConfirmedCases'],c='blue',s=bubbol, alpha=0.6)
plt.title("22 January - 11 April Total ConfirmedCases By Date For United State",fontsize=25)
plt.xlabel('Date',fontsize=25)
plt.ylabel('ConfirmedCases',fontsize=25)
plt.show()


# - Day By Day Increase The Number of ConfirmedCases In US.

# ## Scatter Plot Date VS Deaths For United State

# In[ ]:


bubbol = np.array(data_usa['Fatalities']/40) # For Bubbol Size
plt.figure(figsize=(15,8))
plt.scatter(data_usa['Date'],data_usa['Fatalities'],c='blue',s=bubbol, alpha=0.6)
plt.title("22 January - 11 April Total Deaths By Date For United State",fontsize=25)
plt.xlabel('Date',fontsize=25)
plt.ylabel('Deaths',fontsize=25)
plt.show()


# - Day By Day Increase The Number of deaths In US.

# ## ConfirmedCases & Active/Recover & Deaths By Date For United State

# In[ ]:


plt.figure(figsize=(15,8))
plt.plot(data_usa['Date'],data_usa['ConfirmedCases'],c='blue', alpha=0.6,label='ConfirmedCases')
plt.plot(data_usa['Date'],data_usa['Active/Recover'],c='green', alpha=0.6, label='Active/Recover')
plt.plot(data_usa['Date'],data_usa['Fatalities'],c='red', alpha=0.6, label='Fatalities')
plt.title("Total ConfirmedCases & Active/Recover & Deaths By Date For United State",fontsize=25)
plt.xlabel('Date',fontsize=25)
plt.ylabel('ConfirmedCases & Active/Recover & Deaths',fontsize=25)
plt.legend(loc=10)
plt.show()


# - ConfirmedCases & Active/Recover highly increases
# - Deaths are increase after 9th April.

# ## Bar Plot Date VS ConfirmedCases For United State

# In[ ]:


fig, ax = plt.subplots(figsize = (20,10))    
fig = sns.barplot(x ="Date", y ="ConfirmedCases", data = data_usa)

x_dates = data_usa['Date'].dt.strftime('%Y-%m-%d').sort_values()
ax.set_xticklabels(labels=x_dates, rotation=80)
plt.title("22 January - 11 April Total ConfirmedCases By Date For United State",fontsize=25)
plt.xlabel('Date',fontsize=25)
plt.ylabel('ConfirmedCases',fontsize=25)
plt.show()


# - Day By Day Increase The Number of ConfirmedCases In US.

# ## Bar Plot Date VS Deaths For United State

# In[ ]:


fig, ax = plt.subplots(figsize = (20,10))    
fig = sns.barplot(x ="Date", y ="Fatalities", data = data_usa)

x_dates = data_usa['Date'].dt.strftime('%Y-%m-%d').sort_values()
ax.set_xticklabels(labels=x_dates, rotation=80)
plt.title("22 January - 11 April Total Deaths By Date For United State",fontsize = 25)
plt.xlabel('Date',fontsize = 25)
plt.ylabel('Deaths',fontsize = 25)
plt.show()


# - Day By Day Increase The Number of Deaths In US.
