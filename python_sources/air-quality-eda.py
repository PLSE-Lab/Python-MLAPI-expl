#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os


# In[ ]:


#As the date is of object type we need to cnvert it to datetime format hence will reading the data:
abc=pd.read_csv("../input/airquality/data.csv", parse_dates=['date'])
abc.info()


# In[ ]:


abc['type'].value_counts()


# In[ ]:


#as there are alot of duplicate types
#cleaning type column and should have only four columns Industrial,Residential,Sensitive and RIRUO
#Updating the changes to abc data frame
abc.loc[(abc['type']=="Residential, Rural and other Areas"),'type']='Residential'
abc.loc[(abc['type']=="Residential and others"),'type']='Residential'
abc.loc[(abc['type']=="Industrial Area"),'type']='Industrial'
abc.loc[(abc['type']=="Industrial Areas"),'type']='Industrial'
abc.loc[(abc['type']=="Sensitive Area"),'type']='Sensitive'
abc.loc[(abc['type']=="Sensitive Areas"),'type']='Sensitive'
abc['type'].value_counts()


# # Showing Spm and Rspm level state wise

# In[ ]:


#Filling missing values for rspm and spm hence grouping by location and type
grp_location=abc.groupby(['location','type'])
dict_grp_location=dict(list(grp_location))
# dict_grp_location


# In[ ]:


print(abc['rspm'].isnull().sum())
print(abc['spm'].isnull().sum())


# In[ ]:


#Forward filling
grouped_location=pd.DataFrame()
for key in dict_grp_location:
    df1=dict_grp_location[key].sort_values(by='date')
    df1['rspm'].fillna(method='ffill',inplace=True)
    df1['spm'].fillna(method='ffill',inplace=True)
    grouped_location=pd.concat([grouped_location,df1])


# In[ ]:


print(grouped_location['rspm'].isnull().sum())
print(grouped_location['spm'].isnull().sum())


# In[ ]:


#Initially we have grouped by 'location' and 'type' and then did foward fill but some values were not filled hence backward fill
backwardfill=grouped_location.groupby(['location','type'])
backwardfill=dict(list(backwardfill))
backwardfill
grouped_location=pd.DataFrame()
for key in backwardfill:
    df2=backwardfill[key].sort_values(by='date')
    df2['rspm'].fillna(method='bfill',inplace=True)
    df2['spm'].fillna(method='bfill',inplace=True)
    grouped_location=pd.concat([grouped_location,df2])


# In[ ]:


print(grouped_location['rspm'].isnull().sum())
print(grouped_location['spm'].isnull().sum())


# In[ ]:


#now we are grouping it on larger scale that is 'state' and thn by 'type' so as to fill null values
dict_grouped_state=dict(list(grouped_location.groupby(['state','type'])))


# In[ ]:


grouped_state=pd.DataFrame()
for key in dict_grouped_state:
    df3=dict_grouped_state[key]
    df3['rspm'].fillna(df3['rspm'].median(),inplace=True)
    df3['spm'].fillna(df3['spm'].median(),inplace=True)
    grouped_state=pd.concat([grouped_state,df3])


# In[ ]:


print(grouped_state['spm'].isnull().sum())
print(grouped_state['rspm'].isnull().sum())


# In[ ]:


#Now we are grouping by 'type' and replacimg all remaining nan values
grouped_type=grouped_state.groupby('type').median()
grouped_type


# In[ ]:


dataframe=grouped_state


# In[ ]:


dataframe.loc[(dataframe['type']=='Industrial') & (dataframe['rspm'].isnull()),'rspm']=grouped_type['rspm']['Industrial']
dataframe.loc[(dataframe['type']=='RIRUO') & (dataframe['rspm'].isnull()),'rspm']=grouped_type['rspm']['RIRUO']
dataframe.loc[(dataframe['type']=='Residential') & (dataframe['rspm'].isnull()),'rspm']=grouped_type['rspm']['Residential']
dataframe.loc[(dataframe['type']=='Sensitive') & (dataframe['rspm'].isnull()),'rspm']=grouped_type['rspm']['Sensitive']

dataframe.loc[(dataframe['type']=='Industrial') & (dataframe['spm'].isnull()),'spm']=grouped_type['spm']['Industrial']
dataframe.loc[(dataframe['type']=='RIRUO') & (dataframe['spm'].isnull()),'spm']=grouped_type['spm']['RIRUO']
dataframe.loc[(dataframe['type']=='Residential') & (dataframe['spm'].isnull()),'spm']=grouped_type['spm']['Residential']
dataframe.loc[(dataframe['type']=='Sensitive') & (dataframe['spm'].isnull()),'spm']=grouped_type['spm']['Sensitive']


# In[ ]:


print(dataframe['rspm'].isnull().sum())
print(dataframe['spm'].isnull().sum())


# In[ ]:


#adding a new 'year' column from 'date' column
dataframe['year']=dataframe['date'].dt.year
print(dataframe['year'].isnull().sum())


# In[ ]:


#filling null values in year by either doing forward fill or backwadr fill
dataframe['year']=dataframe['year'].fillna(method='ffill')
print(dataframe['year'].isnull().sum())
dataframe['year']=dataframe['year'].astype(int)


# In[ ]:


#ploting states in descending order as per the level of spm
dataframe


# In[ ]:


state=dataframe.groupby('state').median()
state=state[['rspm','spm']]
state=state.sort_values(by='spm',ascending=False)


# In[ ]:


state.plot(kind='bar',figsize=(15,10))


# In[ ]:


# potting a graph in  descending order as per the level of spm
state.sort_values(by='rspm',ascending=False).plot(kind='bar',figsize=(15,10))


# # Looking at top 5 states with high spm values

# In[ ]:


states=state.reset_index().head(5)
top_five_states=states['state']
for i in top_five_states:
    print(i)


# In[ ]:


group_by_state=dict(list(dataframe.groupby('state')))
plot_five_states=pd.DataFrame()
for i in top_five_states:
    df=group_by_state[i][['state','location','spm','rspm','type']]
    plot_five_states=pd.concat([plot_five_states,df])
plot_five_states


# In[ ]:


plot_five_states=plot_five_states.groupby(['state','location','type']).median()
plot_five_states


# In[ ]:


plt.figure(figsize = (20,20))
plt.subplot(3,2,1)
plt.title('Delhi')
a = sns.barplot(x = 'location',y = 'spm',hue = 'type',data = plot_five_states.loc['Delhi'].reset_index())
a.set(ylim = (0,600))
plt.subplot(3,2,2)
plt.title('Haryana')
b = sns.barplot(x = 'location',y = 'spm',hue = 'type',data = plot_five_states.loc['Haryana'].reset_index())
b.set(ylim = (0,600))
plt.subplot(3,2,3)
plt.title('Rajasthan')
c = sns.barplot(x = 'location',y = 'spm',hue = 'type',data = plot_five_states.loc['Rajasthan'].reset_index())
c.set(ylim = (0,600))
plt.subplot(3,2,4)
plt.title('Uttarakhand')
d = sns.barplot(x = 'location',y = 'spm',hue = 'type',data = plot_five_states.loc['Uttarakhand'].reset_index())
d.set(ylim = (0,600))
plt.subplot(3,2,5)
plt.title('Uttar Pradesh')
plt.xticks(rotation = 90)
g = sns.barplot(x = 'location',y = 'spm',hue = 'type',data = plot_five_states.loc['Uttar Pradesh'].reset_index())
g.set(ylim = (0,600))


# The above graphs are of top 5 states showing the spm levels as per their locations.
# In almost all the states, the residential locations are contributing to spm level same as that of industrial locations.
# From the graph, its clear that in states like Delhi and Uttarakhand the spm level in residential areas is more 
# than that of industrial areas which is quite surprising.

# # Which years have not recorded the spm value 

# In[ ]:


states_year=dataframe.groupby(['state','year']).median()['spm']


# In[ ]:


states_year


# In[ ]:


states_year=states_year.reset_index()
states_year['spm'].isnull().sum()


# In[ ]:


pivot=pd.pivot_table(states_year,values='spm',index='state',columns='year')
pivot.fillna(0,inplace= True)


# In[ ]:


pivot


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(data=pivot,annot=True)


# The 0 value represents for that year there is missing spm value/no spm values are recorded for that year.
# 

# # So2 and No2 levels in bangalore over the years

# In[ ]:


#Finding the null values for so2 and no2 for karnataka state and the replacing them with median values.
#1)SO2
karnataka=abc.groupby(['state','type'])
a=dict(list(karnataka))
kar_ind=a[('Karnataka','Industrial')]
print(kar_ind['so2'].isnull().sum())
kar_res=a[('Karnataka','Residential')]
print(kar_res['so2'].isnull().sum())
kar_sensitive=a[('Karnataka','Sensitive')]
print(kar_sensitive['so2'].isnull().sum())
# Karnataka has no RIRUO
# kar_riruo=a[('Karnataka','RIRUO')]
# kar_riruo['so2'].isnull().sum()


# In[ ]:


#now replacing all these null values of So2  with median values  in copy_abc data frame
copy_abc=abc.copy()


# In[ ]:


copy_abc.loc[(copy_abc['state']=='Karnataka') & (copy_abc['type']=='Industrial') & (copy_abc['so2'].isnull()),'so2']=kar_ind.median()['so2']
copy_abc.loc[(copy_abc['state']=='Karnataka') & (copy_abc['type']=='Residential') & (copy_abc['so2'].isnull()),'so2']=kar_res.median()['so2']
copy_abc.loc[(copy_abc['state']=='Karnataka') & (copy_abc['type']=='Sensitive') & (copy_abc['so2'].isnull()),'so2']=kar_sensitive.median()['so2']


# In[ ]:


# Checking if null values are imputed S02 
karnataka=copy_abc.groupby(['state','type'])
a=dict(list(karnataka))
print(a[('Karnataka','Industrial')]['so2'].isnull().sum())
print(a[('Karnataka','Residential')]['so2'].isnull().sum())
print(a[('Karnataka','Sensitive')]['so2'].isnull().sum())


# In[ ]:


#2)Replacing below null values for no2 state of karnataka with median in copy_abc
#No2


# In[ ]:


kar_ind=a[('Karnataka','Industrial')]
kar_ind['no2'].isnull().sum()
kar_ind=a[('Karnataka','Residential')]
kar_ind['no2'].isnull().sum()
kar_sensitive=a[('Karnataka','Sensitive')]
kar_sensitive['no2'].isnull().sum()
# Karnataka has no RIRUO
# kar_riruo=a[('Karnataka','RIRUO')]
# kar_riruo['no2'].isnull().sum()


# In[ ]:


copy_abc.loc[(copy_abc['state']=='Karnataka') & (copy_abc['type']=='Industrial') & (copy_abc['no2'].isnull()),'no2']=kar_ind.median()['no2']
copy_abc.loc[(copy_abc['state']=='Karnataka') & (copy_abc['type']=='Residential') & (copy_abc['no2'].isnull()),'no2']=kar_res.median()['no2']
copy_abc.loc[(copy_abc['state']=='Karnataka') & (copy_abc['type']=='Sensitive') & (copy_abc['no2'].isnull()),'no2']=kar_sensitive.median()['no2']


# In[ ]:


# Checking if null values are imputed No2
karnataka=copy_abc.groupby(['state','type'])
a=dict(list(karnataka))
print(a[('Karnataka','Industrial')]['no2'].isnull().sum())
print(a[('Karnataka','Residential')]['no2'].isnull().sum())
print(a[('Karnataka','Sensitive')]['no2'].isnull().sum())


# In[ ]:


#Now we are grouping the data by just Bangalore state as we have to draw graph of no2 and so2
bangalore=copy_abc.groupby('location')
bangalore=dict(list(bangalore))
bangalore=bangalore['Bangalore'][['so2','no2','date']]
bangalore['year']=bangalore['date'].dt.year
bangalore=bangalore[['so2','no2','year']]
bangalore['year'].isnull().sum()


# In[ ]:


#as we have one null value for year field we have to do forward fill or backward fill.Performing 1st backward fill
bangalore['year']=bangalore['year'].fillna(method='bfill')
bangalore['year']=bangalore['year'].astype('int')
bangalore['year'].isnull().sum()


# In[ ]:


bangalore=bangalore.groupby('year').median()
bangalore=bangalore.reset_index()
bangalore


# In[ ]:


plt.figure(figsize=(15,5))
plt.xticks(np.arange(1980,2016))
plt.yticks(np.arange(5,55,5))
sns.pointplot(bangalore['year'],bangalore['so2'],color='r')
sns.pointplot(bangalore['year'],bangalore['no2'],color='g')
plt.legend(['so2','no2'])


# The above graph shows the level of so2 and no2 level for Bangalore state from year 1987 10 2015.
# Amount of So2 in Bangalore city from the year 1988 to 2015:
#     In the year 1988 to 1992 the amount of So2 in the air had gradually decreased. 
#     The amount of so2 gradually increased from the year 1995 to 1998.
#     The amount of so2 from the year 2000 has never increased beyond 25.
#     The highest amount of So2 is for the year 1998 and 1999.
# Amount of No2 in bangaolre city from the year 1988 to 2015:
#     The amount of no2 is gradually increasing till the year 2000.
#     The year 2004 marks the highest amount of no2 level whereas after 2004 the amount of no2 is gradually decreasing
#     
#     

# In[ ]:





# In[ ]:




