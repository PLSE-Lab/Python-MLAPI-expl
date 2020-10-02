#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#importing the data as a Pandas DataFrame
dataset=pd.read_csv('../input/data.csv',encoding="ISO-8859-1")
dataset.describe()


# The dataset consists primarily 5 different types pollutants measured over the years in different states and cities of India.
# 
# Where SO2 and NO2 are harmful gaseous emmissions; rspm, spm and pm2_5 come under susended air pollutants.
# 
# > The count clearly shows that there are variable number of Non-null entries for each of the pollutants.
# 
# > To understand the dataset further, we will have a look at all the different columns now and store them for future reference.
# 
# 
# 

# In[ ]:


dataset.columns
#Apart from the major pollutants, there are columns that refer to the respective states, agencies, sampling dates and the type.
#We will now have a look at what kind of data each of the columns consists of.


# Understanding the pollutants briefly here.
# 
# **NO2**: Nitrogen Dioxide and is emmitted mostly from combustion from power sources or transport.
# 
# **SO2**: Sulphur Dioxide and is emmitted mostly from coal burning, oil burning, manufacturing of Sulphuric acid.
# 
# **spm**: Suspended particulate matter and are known to be the deadliest form of air pollution. They are microscopic in nature and are found to be suspended in earth's atmosphere.
# 
# **rspm**: Respirable suspended particulate matter. A sub form of spm and are respnsible for respiratory diseases.
# 
# **pm2_5**: Suspended particulate matter with diameters less than 2.5 micrometres. They tend to remain suspended for longer durations and potentially very harmful.
# 
# Let us get back to the data again and see how it is stored.
# 

# In[ ]:



dataset.info()

#Now, we can immediatly see that there are quite a few nulls in various columns, which need work and first need a closer inspection.


# In[ ]:


dataset.head()


# Clearly there are lots of null values, noticeably in stn_code,agency, both of which should therefore be not ncluded further in the analysis.
# 
# > Intuitively, these two columns will hardly add much value to analysis.
# 
# > Now, focusing on the categorical variables, we are left with location_monitoring_station which consists of considerable nulls (approximately 27000). 
# 
# ****It would have been useful to have those values for an in depth analysis, but for now we will keep it out because of the null values and come back later if needed.
# 
# 1. Out of the two dates columns, immediate attention goes to sampling date which has different formats within, highlighting some data input issues.
# 1. While, it is importnat to have this metric, more useful is to go back to the origin of the dataset and ask relevant questions,as to why are there different formats? Is it a human error or error due to incorporating different formats.For now, we will keep it out and only have the date column.

# In[ ]:



dataset.drop(['stn_code','agency','sampling_date','location_monitoring_station'],axis=1,inplace=True)
dataset.info()
dataset.head()


# In[ ]:


#Fixing the missing values firstly for all the pollutants.
#We will consider taking mean for all the pollutants columns and make use of the Imputer class
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(dataset.iloc[:, 3:8].values)
dataset.iloc[:,3:8] = imputer.transform(dataset.iloc[:, 3:8].values)
dataset.info()
dataset.head()


# In[ ]:


#Fixing the missing values in the column 'type'
dataset['type'].describe()
#With 10 Unique labels, we will fill the null values by the most common type, which is 'Residential, Rural and Other Areas'.
common_value='Residential,Rural and other Areas'
dataset['type']=dataset['type'].fillna(common_value)
dataset.info()


# We have fixed the missing values now and made the dataset much shorter to focus on the key variables.
# > We should start with some preliminary visualisations, starting foremost with those of the pollutants

# In[ ]:


#Importing libraries
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#We will start with pairplots to undestand the statistics and get a general idea about the interdependence of pollutants.
sns.pairplot(dataset[['so2','no2','pm2_5']])


# **The idea is to understand through data, whether NO2 and SO2 have a role in particulate formation or not?
# > Clearly there is some interdependence between SO2 and NO2, but no clear trend can be seen for pm2_5.
# > Although, making fair asumptions would be premature as there were large number of missing values for pm2_5 as seen earlier.

# In[ ]:


sns.pairplot(dataset[['so2','no2','spm']])


# ****So, now we see some trend. spm emmissions do demonstrate some relation with No2 and So2 emmissions. 
# * As the emmissions for NO2 increase, the spm emmissions demonstrate a slight increase.
# > The right questions, here, therefore could be: Is there any percentage of No2 that gets converted to spm?
# > It will be useful to get this answer as spm are known to have quite a harmful effect on human health.

# In[ ]:


sns.pairplot(dataset[['so2','no2','rspm']])


# Again, nothing in this plot, that can be singled out here as a defining trend. 
# > Either the data is not enough or clearly rspm presence in atmosphere is independent of NO2 or SO2 presence.

# In[ ]:


sns.pairplot(dataset[['rspm','spm','pm2_5']])


# **Atleast according to the data, all the three different suspended particulates seem to have very less interdependence.
# * Suggestive of different origins for each of them or different methods of sampling them.
# * We always need to be mindful of the fact that pm2_5 has a lot of nulls and a large part of it was computed by taking mean.

# ****It is good to have all the emmissions athe same place by making use of subplots.
# > Making subplots for emmissions data.

# In[ ]:



fig, axarr = plt.subplots(2, 2, figsize=(12, 8))
sns.distplot(dataset['no2'],hist=True,kde=True,
             color='darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth':4},
             ax=axarr[0][0])

sns.distplot(dataset['so2'],hist=True,kde=True,
             color='red',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth':4},
             ax=axarr[0][1])

sns.distplot(dataset['rspm'],hist=True,kde=True,
             color='green',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth':4},
             ax=axarr[1][0])

sns.distplot(dataset['spm'],hist=True,kde=True,
             color='black',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth':4},
             ax=axarr[1][1])


# In[ ]:


sns.heatmap(
    dataset.loc[:, ['state','so2', 'no2', 'rspm', 'spm', 'pm2_5']].corr(),
    annot=True
)


# ****Interesting outcome. From the correlations, spm and rspm show a high value, followed by that of rspm and NO2.
# * It could again highlight some important aspect of qualitative analysis that should be added here. 
# > Can we say something about spm and rspm? or rspm and NO2?

# # Grouping the emmissions by state.
# > Having looked at the pollutants distributions, we now would focus on how these emmissions are stacked across the indian states.
# > We will use groupby on the dataset DataFrame and store it in another DataFrame as statewise_emmissions.

# In[ ]:



statewise_emmissions = dataset.groupby('state').mean()[['so2', 'no2', 'rspm', 'spm', 'pm2_5']]
statewise_emmissions.plot.area()
 


# ***The highest emmissions are for spm, for each of the states.*

# > Getting the statistics for highest emmissions, when grouped statewise.

# In[ ]:


statewise_emmissions.describe()


# In[ ]:


Top10States_with_highest_No2=statewise_emmissions.sort_values(by='no2',ascending=False).head(10)
Top10States_with_highest_No2_sorted=Top10States_with_highest_No2.loc[:,['no2']]
Top10States_with_highest_No2_sorted.head()


# 1. West Bengal and Delhi show the highest NO2 emmissions over the years.
# 1. Questions to ask: Have the Vehicles emmiting NOx and NO2 have been monitored well in these two states?
# 1. What type of Industrial waste is being generated in these two states?

# In[ ]:


Top10states_with_highest_So2=statewise_emmissions.sort_values(by='so2',ascending=False).head(10)
Top10states_with_highest_So2_sorted=Top10states_with_highest_So2.loc[:,['so2']]
Top10states_with_highest_So2_sorted.head()


# 1. With Uttaranachal and Jharkand right at the top, it would be wise to ask whether the coal mining industry in these two states regulated well, as a substantial amount of SO2 emmission can come from the combustion of coal.
# > Importantly, there are no states from South India.

# In[ ]:


Top10states_with_highest_rspm=statewise_emmissions.sort_values(by='rspm',ascending=False).head(10)
Top10states_with_highest_rspm_sorted=Top10states_with_highest_rspm.loc[:,['rspm']]
Top10states_with_highest_rspm_sorted.head()


# In[ ]:


Top10states_with_highest_spm=statewise_emmissions.sort_values(by='spm',ascending=False).head(10)
Top10states_with_highest_spm_sorted=Top10states_with_highest_spm.loc[:,['spm']]
Top10states_with_highest_spm_sorted.head()


# ****The distribution for spm is generally on the higher side, but Delhi and Uttar Pradesh show the highest presence of suspended Particulate matter.
# 
# 1. Are their specific insustries that contribute higher to the spm concentration in and around Delhi?
# > To be underlined is the non-presence of Southern and North eastern states.

# In[ ]:


Top10states_with_highest_pm2_5=statewise_emmissions.sort_values(by='pm2_5',ascending=False).head(10)
Top10states_with_highest_pm2_5_sorted=Top10states_with_highest_pm2_5.loc[:,['pm2_5']]
Top10states_with_highest_pm2_5_sorted.head()


# > The data output can be misleading owing to the missing values of pm2_5 values and subsequent mean.
# > Neverthless, Delhi still shows highest measured value of pm2_5.
# 

# In[ ]:


#Getting the statistics citywise for the pollutants
locationwise_emmissions=dataset.groupby('location').mean()[['so2','no2','rspm','spm','pm2_5']]


# In[ ]:


Top10Cities_with_highest_NO2=locationwise_emmissions.sort_values(by='no2',ascending=False).head(10)
Top10Cities_with_highest_NO2_sorted=Top10Cities_with_highest_NO2.loc[:,['no2']]
Top10Cities_with_highest_NO2_sorted.head()


# In[ ]:



Top10Cities_with_highest_So2=locationwise_emmissions.sort_values(by='so2',ascending=False).head(10)
Top10Cities_with_highest_So2_sorted=Top10Cities_with_highest_So2.loc[:,['so2']]
Top10Cities_with_highest_So2_sorted.head()


# In[ ]:


Top10Cities_with_highest_rspm=locationwise_emmissions.sort_values(by='rspm',ascending=False).head(10)
Top10Cities_with_highest_rspm_sorted=Top10Cities_with_highest_rspm.loc[:,['rspm']]
Top10Cities_with_highest_rspm_sorted.head()


# In[ ]:


Top10Cities_with_highest_spm=locationwise_emmissions.sort_values(by='spm',ascending=False).head(10)
Top10Cities_with_highest_spm_sorted=Top10Cities_with_highest_spm.loc[:,['spm']]
Top10Cities_with_highest_spm_sorted.head()


# In[ ]:


Top10Cities_with_highest_pm2_5=locationwise_emmissions.sort_values(by='pm2_5',ascending=False).head(10)
Top10Cities_with_highest_pm2_5_sorted=Top10Cities_with_highest_pm2_5.loc[:,['pm2_5']]
Top10Cities_with_highest_pm2_5_sorted.head()


# In[ ]:


#Visualising the emmissions according to the type and getting the relevant statistics
type_emmissions=dataset.groupby('type').mean()[['so2','no2','rspm','spm','pm2_5']]
type_emmissions.head()


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(18,14))
ax = sns.barplot("so2", y="type",
                 data=dataset,
                 ax=axes[0,0]
                )
ax = sns.barplot("no2", y="type",
                 data=dataset,
                 ax=axes[0,1]
                )
ax = sns.barplot("rspm", y="type",
                 data=dataset,
                 ax=axes[1,0]
                )
ax = sns.barplot("spm", y="type",
                 data=dataset,
                 ax=axes[1,1]
                )


# * Largely it is the Industrial affluents that contribute highest percentage of all the pollutants in India.

# In[ ]:


#Understanding the emmissions with time
dataset['date'].describe()


# * There are exactly seven missing date values and as we can see there are multiple measurements for the same date.
# * While dropping the missing dates might seem easiest, we might loose out on some important information.
# > Therefore, we will fix this by filling in the missing values by the most frequent values.

# In[ ]:


dataset.head()
common_value_date='2015-03-19'
dataset['date']=dataset['date'].fillna(common_value_date)
dataset.tail()


# * Visualising the Emmissions over the years, by grouping the dataset datewise and creating a new DataFrame for each of the five listed pollutants.

# In[ ]:


datewise_emmissions_SO2=dataset.groupby('date').mean()['so2']
datewise_emmissions_NO2=dataset.groupby('date').mean()['no2']
datewise_emmissions_rspm=dataset.groupby('date').mean()['rspm']
datewise_emmissions_spm=dataset.groupby('date').mean()['spm']



# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(14,10))
datewise_emmissions_SO2.plot(style='k.',legend=True,ax=axes[0,0])
datewise_emmissions_NO2.plot(style='b.',legend=True,ax=axes[0,1])
datewise_emmissions_rspm.plot(style='r.',legend=True,ax=axes[1,0])
datewise_emmissions_spm.plot(style='g.', legend=True,ax=axes[1,1])


# * So2 emmissions it appears showed a lot of variable readings in the 1990's when the measurements started, but since, the emmissions have been localised within a range and even started showing slight downward trend lately.
# 
# * NO2 emmissions it appears shows higher variations in the emmissions readings across India.
# > Are there some particular months or times of the year which shows higher No2 emmissions?
# 
# * Largely distributed within a certain range of values over the years, although rspm emmissions should be investigated by the different times of year. Are there any further and localised trends?
# 
# * The spm measurements early in the 1990's showed huge spikes, are these real or these are measurement issues?
# > The good part is the gradual reduction in spm presence in India's atmosphere, but missing data recently really doen not help in further investigation.

# In[ ]:


dataset.groupby('so2').max()[['state','date']].tail(20)


# *The 20 highest SO2 emmissions measurements have been recorded in West Bengal, most importantly majority of them in 2011.*
# > What was the significance of these measurements? 
# > Are these centred around any coal combustion in 2011 or these reflect a sudden surge in the numbers for resons yet unknown?

# In[ ]:


dataset.groupby('no2').max()[['state','date']].tail(20)


# 1. Uttar Pradesh and West bengal again feature as the states responsible for highest N02 emmissions. 
# 1. While West Bengal measurements date way back in the 1990s, the more recent ones are in Uttar Pradesh.
# 1. The highest measurements are as early as 2014, from Rajasthan.
# > Are these primarily due to vehicular exhaust or the new industries with NO2 exhausts?

# In[ ]:


dataset.groupby('spm').max()[['state','date']].tail(20)


# *Interestingly a lot of high spm measurements have been reported from the state of Rajasthan. *
# > Are Dust storms the primary reasons for suspended particulates in Rajasthan?

# In[ ]:


dataset.groupby('rspm').max()[['state','date']].tail(20)


# Getting the statistics for highest emissions,when the date column is parsed and is recorded as a datetime, instead of an object.

# In[ ]:


dataset['date'] = pd.to_datetime(dataset.date, format='%Y-%m-%d')
dataset.info()

#As it can be seen now, the date column is converted into datetime, instead of an object. This method is useful for 
#anlysing trends with time within the dataset.


# In[ ]:


#Making the date column as the index of the dataframe to make plotting and visulaisation easier.
dataset=dataset.set_index('date')
dataset.head()


# Resampling the dataset, 
# * yearly 
# * monthly 
# * weekly
# * daily.
# We will resample by taking the mean of all the measurements within the resampling timeframe, i.e. 'yearly', 'monthly' etc and 
# consequently store each of the results in separate dataframes.

# In[ ]:



yearly = dataset.resample('Y').mean()

monthly=dataset.resample('M').mean()

weekly=dataset.resample('W').mean()

daily=dataset.resample('D').mean()


# In[ ]:


#All the above dataframes will be grouped together and plotted together in a sinlge frame using subplots.

fig,axes=plt.subplots(nrows=2,ncols=2, figsize=(14,10))
yearly.plot(style=[':', '--', '-','.','*'],
            ax=axes[0,0],
            title='Yearly Emmissions')

monthly.plot(style=[':', '--', '-','.','*'],
             ax=axes[0,1],
             title='Monthly Emmissions')

weekly.plot(style=[':', '--', '-','.','*'],
            ax=axes[1,0],
            title='Weekly Emmissions')

daily.plot(style=[':', '--', '-','.','*'],
            ax=axes[1,1],
            title='Daily Emmissions')


# > The most important thing to note here is the sudden increase in spm emmissions after 2003 (**Daily Emmissions plot**), indicating of some spm emmission mode that increased drastically in India.
# > It would be very useful to check for some resource online that indicates a change in the environment policy around 2003, specifically towards spm measurements. 

# In[ ]:


#Putting together all the emissions data, datewise and visualising the data distributions, outliers and median values
fig,axes2=plt.subplots(nrows=2,ncols=2, figsize=(14,10))
yearly.plot.box(
                ax=axes2[0,0],
                title='Yearly Emmissions Distribution')

monthly.plot.box(
                ax=axes2[0,1],
                title='Monthly Emmissions Distribution')

weekly.plot.box(
                ax=axes2[1,0],
                title='Weekly Emmissions Distribution')

daily.plot.box(
                ax=axes2[1,1],
                title='Daily Emmissions Distribution')


# > Again, the daily emmissions distributions shows a high number of outliers for spm measurements. Is it truely that variable across India or there are still different ways of mensurements OR
#  different times of measurments?

# Getting the statistics out of each of the yearly merged datasets. The idea is to get a firsthand understanding of whether the years with highest pollutant emmisions have any relation to the changing evironment policy structures or the growth in the industrial and vehicular presence.

# In[ ]:



Top5Years_highest_SO2=yearly.sort_values(by='so2', ascending=False).head(5)
Top5Years_highest_SO2.loc[:,'so2']



# In[ ]:


Top5Years_highest_NO2=yearly.sort_values(by='no2', ascending=False).head(5)
Top5Years_highest_NO2.loc[:,'no2']


# In[ ]:



Top10Years_highest_spm=yearly.sort_values(by='spm', ascending=False).head(10)
Top10Years_highest_spm.loc[:,'spm']


# In[ ]:


Top10Years_highest_rspm=yearly.sort_values(by='rspm', ascending=False).head(10)
Top10Years_highest_rspm.loc[:,'rspm']


# The most identifiable trend here, is that 'rspm' measurements have increased dratically after 2000, while all the other three pollutants were recorded highest in 1980s and 1990s.

# Getting the statistics out of each of the mothly merged datasets, now. The idea is to get a firsthand understanding of whether the moths with highest pollutant emmisions have any relation to the changing of the seasons?

# In[ ]:


#Getting the statistics out of each of the monthly merged datasets
Top10Months_highest_SO2=monthly.sort_values(by='so2', ascending=False).head(10)
Top10Months_highest_SO2.loc[:,'so2']





# In[ ]:


Top10Months_highest_NO2=monthly.sort_values(by='no2', ascending=False).head(10)
Top10Months_highest_NO2.loc[:,'no2']


# In[ ]:


Top20Months_highest_spm=monthly.sort_values(by='spm', ascending=False).head(20)
Top20Months_highest_spm.loc[:,'spm']


# In[ ]:


Top20Months_highest_rspm=monthly.sort_values(by='rspm', ascending=False).head(20)
Top20Months_highest_rspm.loc[:,'rspm']


# > SO2 emmissions demonstrate highest occurence during the winter season in India and that is espeiaaly true for rspm measurements.
# * Most importantly, rspm level increase in the winters needs a cross examination with the prevelant winter activities like crop cutting etc.
# > Other than these defingin trends, much less can be inferred about spm measurements with seasonal changes, indicating other potential sources of pollution.
