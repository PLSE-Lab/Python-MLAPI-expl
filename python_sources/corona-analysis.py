#!/usr/bin/env python
# coding: utf-8

# # Investigatng the data and exploratory data analysis

# First installing all the libraries that will use in our application. Installing the libraries in the first part because the algorithm we use later and the analysis we make more clearly will be done. Furthermore, investigating the data, presented some visualization and analyzed some features. Lets write it. Importing necessary packages and libraries.

# In[ ]:



import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime


# Now we are uploading our data set using the variable "corona" in the pandas library.

# In[ ]:


corona= pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")


# In[ ]:


#after the loading the data. Next step is to view/see the top 10 rows of the loaded data set

corona.head(10)


# In[ ]:


#last 10 rows of loaded data set

corona.tail(10)


# In[ ]:


corona.describe()


# describe function is a function that allos analysis between the numeric values contained in the the dataset. Using this function count, mean,std, min,max,25%,50%,75%.
# 
# As seen in this section, most values are generally numeric.

# In[ ]:


#information about each var

corona.info()


# In[ ]:


#we will be listing the columns of all the data.
#we will check all columns

corona.columns


# In[ ]:


corona.sample(frac=0.01)


# In[ ]:


#sample: random rows in the dataset
#useful for future analysis
corona.sample(5)


# In[ ]:


#next, how many rows an columns are there in the loaded data set

corona.shape


# In[ ]:


# and, will check null on all the data and if there is any null, getting the sum of all the null data's

corona.isna().sum()


# we can see from the above analysis, there are 462 NaN values from Province/state variable

# In[ ]:


df= corona.groupby('ObservationDate')['Confirmed','Deaths','Recovered'].sum()
df=df.reset_index()
df=df.sort_values('ObservationDate', ascending= True)
df.head(60)


# In[ ]:


df= corona.groupby('Province/State')['Confirmed','Deaths','Recovered'].sum()
df=df.reset_index()
df=df.sort_values('Province/State', ascending= True)
df.head(60)


# In[ ]:


#df=corona[corona['Confirmed'] == corona['Deaths']+['Recovered']]
#df=df[['Province','Confirmed','Deaths','Recovered']]#
#df=df.reset_index()
#df=df.sort_values('Confirmed',ascending= True)
#df.head()


# In[ ]:


df= corona.groupby('ObservationDate')['Confirmed','Deaths','Recovered'].sum()
df.sort_values('ObservationDate',ascending=True)
df.head(10)


# In[ ]:


print(min(corona.Confirmed))
print(max(corona.Confirmed))
print(corona.Confirmed.mean())


# In[ ]:


print(min(corona.Deaths))
print(max(corona.Deaths))
print(corona.Deaths.mean())


# In[ ]:


print(min(corona.Recovered))
print(max(corona.Recovered))
print(corona.Recovered.mean())


# # Working on with the different data i,e confirmed data, deaths data and recovered data

# In[ ]:


#loading the raw data of confirmed, deaths and confirmed

conf=pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
death=pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
recov=pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

print(conf.shape)
print(death.shape)
print(recov.shape)

conf.head()


# In[ ]:


conf2 = pd.melt(conf, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name=['Date'])
death2 = pd.melt(death, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name=['Date'])
recov2 = pd.melt(recov, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name=['Date'])

print(conf2.shape)
print(death2.shape)
print(recov2.shape)

conf2.head()


# In[ ]:


# Converting the new column to dates

conf2['Date'] = pd.to_datetime(conf2['Date'])
death2['Date'] = pd.to_datetime(death2['Date'])
recov2['Date'] = pd.to_datetime(recov2['Date'])


# In[ ]:


#renaming the values to confirmed,death and recivered with respected datasets

conf2.columns=conf2.columns.str.replace('value','Confirmed')
death2.columns=death2.columns.str.replace('value','Deaths')
recov2.columns=recov2.columns.str.replace('value','Recovered')


# In[ ]:


#Finding the sum of NaN values in the columns of respective loaded data set

print(conf2.isna().sum())
#print(death2.isna().sum())
#print(recov2.isna().sum())


# In[ ]:


#Dealing with the Nan values

conf2['Province/State'].fillna(conf2['Country/Region'], inplace=True)
death2['Province/State'].fillna(death2['Country/Region'], inplace=True)
recov2['Province/State'].fillna(recov2['Country/Region'], inplace=True)

conf2.isna().sum()


# In the next step we are going to join/ combine the three data sets, first we will once again printing the shape of the each loaded data set
# 
# 

# In[ ]:


print(conf2.shape)
print(death2.shape)
print(recov2.shape)


# And, in this step we are going to join the three datas i,e full joins
# 

# In[ ]:


join= conf2.merge(death2[['Province/State','Country/Region','Date','Deaths']], 
                                      how = 'outer', 
                                      left_on = ['Province/State','Country/Region','Date'], 
                                      right_on = ['Province/State', 'Country/Region','Date'])

join2= join.merge(recov2[['Province/State','Country/Region','Date','Recovered']], 
                                      how = 'outer', 
                                      left_on = ['Province/State','Country/Region','Date'], 
                                      right_on = ['Province/State', 'Country/Region','Date'])

join2.head()


# In[ ]:


df= join2.groupby('Country/Region')['Confirmed','Deaths','Recovered'].sum()
df=df.reset_index()
df=df.sort_values('Country/Region', ascending= True)
df.head(60)


# In[ ]:


#Verifying is there any null values from the above data

join2.isna().sum()


# So, there is no NaN values from the above dataset

# In[ ]:


#Adding month and year as a new column

join2['Month-Year'] = join2['Date'].dt.strftime('%b-%Y')


# In[ ]:


join2.head(10)


# ## Total confirmed cases, death cases and recovered cases 

# In[ ]:


df= join2.groupby('Month-Year')['Confirmed','Deaths','Recovered'].sum()
df.sort_values('Month-Year',ascending=True)
df.head()


# In[ ]:




