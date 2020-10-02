#!/usr/bin/env python
# coding: utf-8

# # **Case Study: Predicting flight delays**
# Originally developed by Fabien Daniel (September 2017)
# 
# ## Overview of the dataset
# Let's read the file that contains the details of all the flights that occured in 2015. I output some informations concerning the types of the variables in the dataframe and the quantity of null values for each variable:

# In[ ]:


import pandas as pd

df = pd.read_csv('../input/flights.csv', low_memory=False)
print('Dataframe dimensions:', df.shape)
#____________________________________________________________
# gives some infos on columns types and number of null values
tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})
tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))
tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100)
                         .T.rename(index={0:'null values (%)'}))
df.info()
#df = df.dropna()
#df = df[['YEAR', 'MONTH', 'DAY', 'AIRLINE', 'FLIGHT_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT' ]].dropna()
tab_info


# Each entry of the `flights.csv` file corresponds to a flight and we see that more than 5'800'000 flights have been recorded in 2015. These flights are described according to 31 variables. A description of these variables can be found [here](https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time) and I briefly recall the meaning of the variables that will be used in this notebook:
# 
# - **YEAR, MONTH, DAY, DAY_OF_WEEK**: dates of the flight <br/>
# - **AIRLINE**: An identification number assigned by US DOT to identify a unique airline <br/>
# - **ORIGIN_AIRPORT** and **DESTINATION_AIRPORT**: code attributed by IATA to identify the airports <br/>
# - **SCHEDULED_DEPARTURE** and **SCHEDULED_ARRIVAL** : scheduled times of take-off and landing <br/> 
# - **DEPARTURE_TIME** and **ARRIVAL_TIME**: real times at which take-off and landing took place <br/> 
# - **DEPARTURE_DELAY** and **ARRIVAL_DELAY**: difference (in minutes) between planned and real times <br/> 
# - **DISTANCE**: distance (in miles)  <br/>
# 
# An additional file of this dataset, the `airports.csv` file, gives a more exhaustive description of the airports:

# In[ ]:


airports = pd.read_csv("../input/airports.csv")
airports


# Also, the `airlines.csv` file, gives a more exhaustive description of the airlines:

# In[ ]:


airlines = pd.read_csv("../input/airlines.csv")
airlines


# In[ ]:


pd.set_option('display.max_columns', 500)
df.head(10)


# In[ ]:


df_Jan = df[df['MONTH'] == 1]
df_Jan


# In[ ]:


df_Jan['DATE'] = pd.to_datetime(df_Jan[['YEAR','MONTH', 'DAY']])


# In[ ]:


df_Jan_nocan = df_Jan[df_Jan['CANCELLED'] == 0]


# In[ ]:


df_Jan['CANCELLED'].unique()


# In[ ]:


df_Jan_nocan.isnull().sum().sort_values()


# In[ ]:


df_Jan_nocan[df_Jan_nocan['DEPARTURE_TIME'].isnull() == True]


# ## Tasks
# 
# ### Cleaning
# 1. Given the large dataset, let's just work with the flights from January 2015th.
# 2. Transform **YEAR, MONTH, DAY, DAY_OF_WEEK ** to datetime
# 3. In the **SCHEDULED_DEPARTURE** variable, the hour of the take-off is coded as a float where the two first digits indicate the hour and the two last, the minutes. This format is not convenient and you should thus convert it. Finally, merge the take-off hour with the flight date.
# 
# ### Exploration
# 
# 1. Visualize and compare flight count per airline
# 2. Visualize the average delay(take-off or landing) per airline
# 3. Find out if there is a relation between airport and delays
# 4. Temporal variability of delays
# 
# ### Modelling	
# 
# 1. Predicting flight delays - per airline and all airports 
# 2. Predicting flight delays - all airline and all airports
# 3. Is there a risk of overfitting, how to handle it?
# 4. Model accuracy calculation
# 

# In[ ]:




