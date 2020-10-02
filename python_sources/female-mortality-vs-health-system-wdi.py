#!/usr/bin/env python
# coding: utf-8

# # Female Mortality Vs Health System
# 
# **Research Question**
# * Do health system improvements by making more expediture affects mortality (of females in particular) in South Asian countries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Loading Data
# Data Source: http://databank.worldbank.org/data/download/WDI_csv.zip

# In[ ]:


#browse the folder to see what CSV files are available
get_ipython().system('ls ../input/wbdevelopmentindicator')


# In[ ]:


#Loading WDIData assuming all indicators data shall be in this file
wdi_df = pd.read_csv(r"../input/wbdevelopmentindicator/WDIData.csv")


# # Exploratory Analysis

# In[ ]:


wdi_df.shape


# In[ ]:


#Quick check of the data and columns
wdi_df.head()


# **Data Observation**
# 
# * Data has got 377784 rows and 65 columns
# * All year data is in columns
# * Year range is from 1960 to 2019
# * There is unnamed last column
# * Column names have space

# In[ ]:


#remove spaces from column names
wdi_df.columns = wdi_df.columns.str.replace(' ', '')
#list(wdi_df.columns)


# In[ ]:


#check aggregates
wdi_df.agg( {'CountryCode' : ['nunique'], 'IndicatorCode' : ['nunique']} )


# **Explore indicators realted to Female Mortality & Health System Expenditure**
# 
# Following indicators have been searched from https://data.worldbank.org/
# 1. Mortality rate, adult, female (per 1,000 female adults) - SP.DYN.AMRT.FE
# 2. Current health expenditure per capita (current US$) - SH.XPD.CHEX.PC.CD
# 
# Also, South Asian countries agregates exist with country code as SAS

# In[ ]:


#define a list of filter for adult female mortality and health expenditure per capita
filters_list = ['SP.DYN.AMRT.FE', 'SH.XPD.CHEX.PC.CD']


# In[ ]:


#Get filtered data frame of female mortality and health expenditure per capita in south asia
raw_mor_epc_df = wdi_df[(wdi_df['IndicatorCode'].isin(filters_list)) & (wdi_df['CountryCode']=='SAS')]
raw_mor_epc_df


# In[ ]:


#drop year columns with no data 
mor_epc_df = raw_mor_epc_df.dropna(axis='columns')
mor_epc_df


# In[ ]:


#reshape data from wide format to long format
mor_epc_final_df = mor_epc_df .melt(id_vars=['CountryName', 'CountryCode','IndicatorName', 'IndicatorCode'],
                  var_name='Year',
                  value_name='Value'
                 )

#delete all initial dataframes and release memory
del mor_epc_df
del raw_mor_epc_df
del wdi_df

#peak into final df i.e
# Female Mortality and health expenditure per capita in South Asia
mor_epc_final_df.head()


# In[ ]:


#Verify agreegates and get year range
mor_epc_final_df.agg( {'CountryCode' : ['nunique'], 'IndicatorCode' : ['nunique'], 'Year' : ['min', 'max']} )


# # Analysing Selected Indicators

# In[ ]:


#get sub indicator specific dataframes

#Mortality dataframe
mort_df = mor_epc_final_df[(mor_epc_final_df['IndicatorCode']=='SP.DYN.AMRT.FE')]

#Expenditure dataframe
epc_df = mor_epc_final_df[(mor_epc_final_df['IndicatorCode']=='SH.XPD.CHEX.PC.CD')]


# **Plotting the two Indicators**

# In[ ]:


#get ndarry from dataframe for plotting
years = mort_df['Year'].values
mrate = mort_df['Value'].values
epc = epc_df['Value'].values


# In[ ]:


#Quick plot - Mortality
plt.plot(years,mrate)
plt.show()


# In[ ]:


#Quick plot - Expenditure
plt.plot(years,epc)
plt.show()


# # Visualizations

# **Combine the two plots to analyze the relation**

# In[ ]:


fig, ax = plt.subplots()

#plot mortality
ax.plot(years, mrate, color='blue', marker='o')
ax.set_xlabel('Year')
ax.set_ylabel('Mortality Rate')
ax.set_ylim([0, 250])
ax.grid(True, ls=':')

#get twin object for two different y-axis on the same plot
ax2 = ax.twinx()

#plot health expenditure
ax2.plot(years,epc, color='orange', marker='o')
ax2.set_ylabel('Expenditure in USD')
ax2.set_ylim([0, 100])
ax2.grid(True, ls=':')

#set legend and fig size
fig.set_figheight(6)
fig.set_figwidth(12)
fig.tight_layout()
fig.legend([mort_df['IndicatorName'].iloc[0], epc_df['IndicatorName'].iloc[0]], 
           loc="upper right",
           bbox_to_anchor=(1,1), 
           bbox_transform=ax.transAxes)

plt.title('Female Mortality Vs Health Expenditure in South Asia')
plt.show()


# Above plot showing that expenditure is increasing over a period of 17 years,
# whereas mortality rate is decreasing

# **Evaluating the co-relation between the two indicators**

# In[ ]:


#Visualizing relation through a scatter plot

fig, ax = plt.subplots()
ax.scatter(x=mrate, y=epc, alpha=0.7, color='b')

ax.set_xlabel(mort_df['IndicatorName'].iloc[0], fontsize=12)
ax.set_ylabel(epc_df['IndicatorName'].iloc[0], fontsize=12)
ax.set_title('Female Mortality & Health Expenditure in South Asia', fontsize=15)

# Add correlation line
axs = plt.gca()
m, b = np.polyfit(mrate, epc, 1)
X_plot = np.linspace(axs.get_xlim()[0],axs.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b, alpha=0.5, ls='--')

#set grid and fig size
ax.grid(True, ls=':')
fig.set_figheight(6)
fig.set_figwidth(10)
fig.tight_layout()

plt.show()


# In[ ]:


#Calculating corelation coefficient
np.corrcoef(x=mrate, y=epc )

