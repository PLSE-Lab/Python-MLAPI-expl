#!/usr/bin/env python
# coding: utf-8

# This analysis will provide some answers about crime in the city of Chicago. However, murders will not be covered because the data for this the city is found in a different dataset.
# All data used was provided from the different Polices Districts of the city.
# 
#  [**1.Importing and exploring data**](#19)
#    * [1.1 Importing the data and load packages <b>](#1)
#    * [1.2 Description <b>](#2)
#    * [1.3 Modification and exploring<b>](#3)
#        * [1.3.1 Police Districts<b>](#4)
#        * [1.3.2 Community Areas<b>](#5)
#        * [1.3.3 Primary Type<b>](#6)
#        * [1.3.4 Location Description<b>](#7)
#        * [1.3.5 Arrest<b>](#8)
#        * [1.3.6 Transformation of the datatime</b>](#9)
# 
# [**2.Time Analysis**](#18)
#    * [2.1  Annual Analysis<b>](#10)
#    * [2.2 Monthly Analysis<b>](#11)
#    * [2.3 Daily Analysis <b>](#12)
#       * [2.3.1 Analysis by hours<b>](#20)
# 
# 
#     
# 
# [<b>**4.Community Areas Analysis **<b>](#13)
# 
# [<b>**5.Polices Districts Analysis **<b>](#14)
# 
# [<b>**6.Arrest Analysis **<b>](#15)
# 
# [<b>**7.Primary Type Analysis **<b>](#16)
# 
# [<b>**8.Correlations**](#17)
# * [**8.1 Community Areas vs Polices Districts**](#21)
# * [**8.2 Community Areas vs Crimes**](#22)

# <a id=19></a>
# 
#  # <h1> 1.Importing and exploring data </h1>  #
#     
#    <a id=1></a>
# **<h2>1.1 Importing the data and load packages</h2>**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plots
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


csv_files = ['../input/Chicago_Crimes_2001_to_2004.csv',
            '../input/Chicago_Crimes_2005_to_2007.csv',
            '../input/Chicago_Crimes_2008_to_2011.csv',
            '../input/Chicago_Crimes_2012_to_2017.csv',
            ]

frames = []
for csv in csv_files:
    df = pd.read_csv(csv ,usecols = ['Date','Primary Type','Location Description','District','Community Area','Arrest'])
    frames.append(df)
    
crime = pd.concat(frames)   
crime.head()


# <a id=2></a>
# **<h2>1.2 Description</h2>**
# 
# * **Date** : day when the crime was committed. This column will be modified to facilitate the analysis.
# * **Primary Type** :  type of crime.
# * **Location Description** : place where the crime was committed.
# * **Arrest** : if the person was arrested or not,True or False.
# * **District**: is the number of the police dristrict.[https://home.chicagopolice.org/community/districts//](http://)
# * **Community Area**:  is the number of the diferent community areas in this city. [http://www.thechicago77.com/chicago-neighborhoods/](http://)

# <a id=3 ></a>
# ** <h2> 1.3 Modification and exploring </h2>**

# In[ ]:


crime.shape


# In[ ]:


# Exploring the missing values:
print("Are There Missing Data? :",crime.isnull().any().any())     
print(crime.isnull().sum())


# I have no idea how to complete these missing values. Let's remove them and see how the analysis is affected :

# In[ ]:


dfcrime = crime.dropna()
dfcrime.shape


# In[ ]:


count_data_origin = crime['Date'].count()
count_data_modify = dfcrime['Date'].count()
Value = (count_data_modify/count_data_origin)*100
print('The analysis will be carried on with:',"%.2f" % Value,'% of the total data')


# In[ ]:


dfcrime.info(null_counts = True)


# In[ ]:


# It's more comfortable to do the next transformation for the doing analysis.
dfcrime.columns=[each.replace(" ","_") for each in dfcrime.columns]
dfcrime.columns


# <a id=4 ></a>
# <h3> 1.3.1 Police Dristrict </h3>

# In[ ]:


dfcrime.District.unique()


# In[ ]:


dfcrime[(dfcrime['District'] == 'Beat')]


# It looks like this row does not follow the pattern of the others. For that reason,  I am pretty sure that at some point of this row  could cause some problems with the analysis therefore it will be erased.
# 

# In[ ]:


dfcrime = dfcrime[dfcrime.District != 'Beat']


# In[ ]:


# Changing the type 
dfcrime['District'] = dfcrime['District'].astype(float).astype(int)
dfcrime.District.unique()


# <a id=5 ></a>
# <h3> 1.3.2 Community Area </h3>

# In[ ]:


dfcrime.Community_Area.unique()


# In[ ]:


# Changing the type
dfcrime['Community_Area'] = dfcrime['Community_Area'].astype(float).astype(int)
dfcrime['Community_Area'].unique()


# <a id=6 ></a>
# <h3> 1.3.3 Primary Type</h3>

# In[ ]:


dfcrime.Primary_Type.unique()


# We can see that 'NON-CRIMINAL' has several formats. I will unify it into just one format.

# In[ ]:


dfcrime['Primary_Type'] = dfcrime['Primary_Type'].replace(['NON - CRIMINAL',
                                                           'NON-CRIMINAL (SUBJECT SPECIFIED)'], 
                                                          'NON-CRIMINAL')
dfcrime.Primary_Type.unique()


# <a id=7 ></a>
# <h3> 1.3.4 Location Description </h3>

# In[ ]:


dfcrime.Location_Description.unique()


# In[ ]:


dfcrime['Location_Description'].value_counts().head()


# <a id=8 ></a>
# <h3> 1.3.5 Arrest </h3>

# In[ ]:


dfcrime.Arrest.unique()


# In[ ]:


dfcrime.Arrest.value_counts()


# In[ ]:


dfcrime.Arrest = dfcrime.Arrest.astype('bool')
dfcrime.Arrest.unique()


# <a id=9 ></a>
# <h3> 1.3.6 Transformation of the datatime  </h3> 

# In[ ]:


# Splitting the Date columns will make the analysis easier. 
dfcrime['date'] = dfcrime['Date'].str[:11]
dfcrime['time'] = dfcrime['Date'].str[12:]


# The intention of the next transformation is to convert the columns 'date' and 'time' to ** datetime ** . Like has already been seen in the first explotation in 'Disctrict' had a row with different features. For that reason when I imported the data I didn't use **pd.to_datatime**  It would give an error.
# 
# There is another reason for splitting the 'Date' column into two. It makes working with this format more confortable  when  doing the hours analysis.

# In[ ]:


dfcrime['date'] = pd.to_datetime(dfcrime['date'])
dfcrime['time'] = pd.to_datetime(dfcrime['time'])


# In[ ]:


# Establish 'date' as index in the dataframe.
dfcrime.index = dfcrime.date
del dfcrime['Date']
del dfcrime['date']
dfcrime.head(2)


# <a id=18 ></a>
# # <h1> 3.Time Analysis </h1>#
# 
# <a id=10 ></a>
# <h2> 3.1 Annual Analysis  </h2>
# 
# The main purpose of the annual analysis is to find out how the amount of crimes have developed during the years in Chicago.
# 
# Is criminal activity increasing in the city ?  Is the city taking measures for improving the security of the citizens year over year?.

# In[ ]:


crime_year = dfcrime.resample('Y').count()
crime_year = pd.DataFrame(crime_year.iloc[:,0])
crime_year.columns = ['Total_crime_per_year']
print(crime_year.head())
print(crime_year.tail())


# In the year 2001 and 2017 the amount of data we have is not enough, there is no doubts that in comparison to other years the criminal activity shown is not accurate.
# 
# Therefore, the analysis will covers the years 2002 to 2016.

# In[ ]:


dfcrime = dfcrime['2002' : '2016']


# In[ ]:


crime_year = crime_year['2002' : '2016']

a = crime_year.index
b = np.arange(2002,2018)

grid = sns.barplot(x = a ,y = 'Total_crime_per_year', data = crime_year, color = 'black')

grid.set_xticklabels(b, rotation = 60)
plt.ylabel('Total Crime')
plt.xlabel('Year')
plt.title('Crime per Year')
plt.axhline(crime_year['Total_crime_per_year'].mean())
fig=plt.gcf()
fig.set_size_inches(10,5)
plt.show()


# Notice in the graphic  how in the year 2006 there was a considerable increase in the crime activity. Maybe this year corresponds with the global crisis that also affected United Estates. This could  explain why in 2008 the criminal activity got the highest rate.  <b>
#     
# Over the last few yars, the city has been reducing its crime level year by year, specially in the year 2011 in comparision with 2010.<b>
#     
# <b> A very good question that we could ask is: how was it possible to reduce the amount of crime in 2007 and in the following year crime rate increase significantly? 

# In[ ]:


print('- Crime Activity 2005-2006')
Value_a = ((crime_year['2006'].values- crime_year['2005'].values)/crime_year['2006'].values)*100
print('%.2f' % Value_a, '% has been the crime activity increment from 2005 to 2006', '\n')

print('- Crime Activity 2010-2011')
Value_b = ((crime_year['2010'].values- crime_year['2011'].values)/crime_year['2010'].values)*100
print("%.2f" % Value_b, '% has been the crime activity increment from 2010 to 2011', '\n')


# This analysis could be relevant for two different situations. First, to study the factors which led the city to have the huge amount of crimes in the year 2006. Secondly,  what were the measures taken for reducing the amount of crime in the year 2011 by almost 50% compare to the previous year.
# 
# <a id=11 ></a>
# <h2> 3.2 Monthly Analysis  </h2>
# 
# In this part of the analysis presents a study of the evolution of crime, month by month, between 2002 to 2016  as in the previous case.
# 
# Using a graph I will try to identify patterns during a year, I mean it is possible to know what months are the most dangerous. This could answer questions like:<b> 
# * What months are the safest during the year in the city of Chicago? <b> 
# * In the situation that you wish to visit that city, apart from the weather, what is the best month?

# In[ ]:


crime_month = dfcrime.resample('M').count()
crime_month = pd.DataFrame(crime_month.iloc[:,0])
crime_month.columns = ['Total_crime_per_month']

crime_month.plot()
plt.xlabel('Time')
plt.ylabel('Total crime')
plt.title('Crime per mes')
plt.axhline(crime_month['Total_crime_per_month'].mean(), color = 'black')
fig=plt.gcf()
fig.set_size_inches(10,5)


# As can be seen, there is a yearly pattern. There are some months during the year when there is more crime activity and another with less. I going to analyze these months. </b>
# 
# Group for differents period of year this pattern can be seen more clearly.</b>
# 
# ** Analysis of the months with more crime activity  **

# In[ ]:


print('Top 5 months with more crime activity from 2002 to 2005')
print(crime_month['2002':'2005'].sort_values('Total_crime_per_month', ascending = False).head(), '\n')

print('Top 5 months with more crime activity from 2006 to 2010')
print(crime_month['2006':'2010'].sort_values('Total_crime_per_month', ascending = False).head(),  '\n')

print('Top 5 months with more crime activity from 2011 to 2013')
print(crime_month['2011':'2013'].sort_values('Total_crime_per_month', ascending = False).head(), '\n')

print('Top 5 months with more crime activity from 2014 to 2016')
print(crime_month['2014':'2016'].sort_values('Total_crime_per_month', ascending = False).head(), '\n')


#  In the last four tables it can be seen how the months when the maximun index crime correspond to July and August .<br/>
# Although one of the very curious thing we can find here is that the maximun crime rate during the period from 2015 to 2017 is in the month of Febrary, a month which does not appear in the other periods of time.
# 
# 
# **Analysis of the months with less criminal activity  **

# In[ ]:


print('Top 10 months with less criminal activity from 2003 to 2005')
print (crime_month['2002':'2005'].sort_values('Total_crime_per_month').head(5), '\n')

print('Top 10  months with less criminal activity from 2006 to 2010')
print(crime_month['2006':'2010'].sort_values('Total_crime_per_month').head(5), '\n')

print('Top 5 months with less criminal activity from 2011 to 2013')
print(crime_month['2011':'2013'].sort_values('Total_crime_per_month').head(), '\n')

print('Top 5 months with less criminal activity from 2015 to 2016')
print(crime_month['2015':'2016'].sort_values('Total_crime_per_month').head(), '\n')


# The months with less criminal activity that we find in the analysis are those in winter and autumn. For this reason, it was very unsual to get January with a maximun in the previous study.</b>
#     
# What really happened this month in the city of Chicago for this unsual criminal activity?</b>
# 
# Let's look at the previous analysis in a plot:

# In[ ]:


fig, ax = plt.subplots()

ax.plot(crime_month['2002'].values, color = 'Blue', label = '2002')
ax.plot(crime_month['2003'].values, color = 'Green', label = '2003')
ax.plot(crime_month['2004'].values, color = 'Brown', label = '2004')
ax.plot(crime_month['2005'].values, color = 'Magenta', label = '2005')
ax.plot(crime_month['2006'].values, color = 'Yellow', label = '2006')
ax.plot(crime_month['2007'].values, color = 'red', label = '2007')
ax.plot(crime_month['2008'].values, color = 'cyan', label = '2008')
ax.plot(crime_month['2009'].values, color = 'orange', label = '2009')
ax.plot(crime_month['2010'].values, color = 'hotpink', label = '2010')
ax.plot(crime_month['2011'].values, color = 'lime', label = '2011')
ax.plot(crime_month['2012'].values, color = 'm', label = '2012')
ax.plot(crime_month['2013'].values, color = 'silver', label = '2013')
ax.plot(crime_month['2014'].values, color = 'olive', label = '2014')
ax.plot(crime_month['2015'].values, color = 'salmon', label = '2015')
ax.plot(crime_month['2016'].values, color = 'dimgray', label = '2016')

plt.xlabel('Month')
plt.ylabel('Total crimes')
plt.title('Criminal Activity during a year')
plt.axhline(crime_month['Total_crime_per_month'].mean(), color = 'black')
plt.xlabel('Month')

c = ['January','January','March','May','July','September','November']
ax.set_xticklabels(c)

plt.legend(bbox_to_anchor=(1, 0, .3, 1), loc=2,
           ncol=2, mode="expand", borderaxespad=0)

fig=plt.gcf()
fig.set_size_inches(10,5)


# In the graphic, as it was concluded analytically , it can be seen how the summer months are when the criminal activity is higher.</b>
# Let's see now the amount of crimes per month  in total.

# In[ ]:



Sum_m = [(crime_month['2002'].values + crime_month['2003'].values + crime_month['2004'].values + 
     crime_month['2005'].values + crime_month['2006'] + crime_month['2007'].values + crime_month['2008'].values 
                + crime_month['2009'].values + crime_month['2010'].values + crime_month['2011'].values + 
                crime_month['2012'].values +crime_month['2013'].values + crime_month['2014'].values + 
                crime_month['2015'].values + crime_month['2016'].values)]

Sum_m = Sum_m[0]
Sum_m = Sum_m.reset_index()
Sum_m.index = [['January','Febreary','March','April','May','June',
                'July','August','September','October','November','December']]
del Sum_m['date']
print('Top 5 months with more criminal activity in total')
print(Sum_m.sort_values('Total_crime_per_month', ascending= False).head(), '\n')

print('Top 5 safer months')
print(Sum_m.sort_values('Total_crime_per_month').head(), '\n')


# In[ ]:


Sum_m.plot(kind = 'bar')
fig=plt.gcf()
fig.set_size_inches(10,5)

plt.xlabel('Month')
plt.ylabel('Total crime')
plt.title('Total crime per month')


# Effectively, it was the last study to prove that the more dangerous months in the city of chicago are in summer. This analysis gives us other viewpoint. In the previous explorations the month of May was not the most frequent for time, but in total, May is the month with more criminal activity historically.</b>
# 
# Looking at the previous plot it reveals a huge increase in some years, like 2002, 2006 and 2007, of this parameter.</b>
# 
# Finally, to answer the question what it is the safest time to go sightseeing.I would travel to this city in winter, but I have already said before the weather may not be suitable for touring.

# <a id=12 ></a>
# <h2> 3.3 Daily Analysis  </h2>
# 
# In this analysis will show the criminal activity daily average and also some exceptional days.

# In[ ]:


crime_day = dfcrime.resample('d').count()
crime_day = pd.DataFrame(crime_day.iloc[:,0])
crime_day.columns = ['Total_crime_per_day']
crime_day = crime_day['2002' : '2016']

median_day = crime_day.Total_crime_per_day.mean()
print('The average of criminal activity in the city fo Chicago is:',"%.2f" % median_day)


# In[ ]:


crime_day.plot()

plt.ylabel('Total crime')
plt.title('Crime per day')
plt.axhline(crime_day.Total_crime_per_day.mean(), color = 'black')

fig=plt.gcf()
fig.set_size_inches(20,10)


# In[ ]:


max_day = crime_day['Total_crime_per_day'].max()
print('The day with more crimes in the city of Chicago was:')
crime_day[(crime_day['Total_crime_per_day'] == max_day)]


# In[ ]:


min_day = crime_day['Total_crime_per_day'].min()
print('The day with less crimes in the city of Chicago was:')
crime_day[(crime_day['Total_crime_per_day'] == min_day)]


# <a id=20 ></a>
# # <h3> 3.3.1 Analysis by hours </h3>
# In this section will show how the crimes involved during a day, I mean what hours are common to have more criminal activity.

# In[ ]:


dfhours = dfcrime.reset_index()
dfhours.index = dfhours.time
dfhours.head(1)


# In[ ]:


dfhours = dfhours.resample('h').count()
dfhours = dfhours[['time']]
dfhours.columns = ['Sum_Crimes_per_Hour']
dfhours['Crime_per_hour_median'] = dfhours['Sum_Crimes_per_Hour']/(365*15)
dfhours


# In[ ]:


median_hour = dfhours.Crime_per_hour_median.median()
print('The mean crime per hour is:',"%.2f" % median_hour,'\n')


# ** In this section there is clearly a failure,  at 9 and at 10 in the morning the number of crimes committed is 0. I have not been able to locate this error or how to solve it.
# The average of crimes per hour is  65.87 but without any mistake the real which is 54.91. **

# In[ ]:


x = dfhours.index
y = dfhours.Crime_per_hour_median
a = np.arange(0,24)

grid = sns.pointplot(x = x, y = y, data = dfhours)
grid.set_xticklabels(a)
plt.axhline(dfhours.Crime_per_hour_median.mean(), color = 'black')
fig = plt.gcf()
fig.set_size_inches(20,10)
plt.grid()


# The hours after midday and after midnight have more activities. Although, the graph due to the error was not entirely accurate, it could be assumed that at 10 a.m and at 11 a.m, when it does not collect data, there will be a slight increase following the trend at 6 a.m until the hour that reaches its maximum.

# In[ ]:


print('Top 5 hours with more crime activity')
print(dfhours.sort_values('Crime_per_hour_median', ascending= False).head(), '\n')

print('Top 5 hours with less crime activity')
print(dfhours.sort_values('Crime_per_hour_median').head(7))


# The hours of the night are those that less crimes are registered, this could be due to two factors: </b>
# + There is not so much police activity so that the crimes happend without the police districts being aware of it.
# + Really most happend during the day.
# 
# <a id=13 ></a>
# # 4. Community areas #
# In this section will analyze which are the most dangerous areas and which are the safest. </b>
# 
# This study responds to solutions to questions such as: </b>
# * In case of going sightseeing in this city, what are the areas I should avoid? </b>
# * If I wanted to buy an apartment to live in this city, which areas are the least appropriate?  </b>
# 
# You will see how is the evolution of the most conflictive areas over the years in a graphic 

# In[ ]:


dfcrime['Community_Area'].value_counts()


# In[ ]:


plt.figure(figsize=(20,25))
sns.countplot(y = dfcrime['Community_Area'])
plt.axvline(dfcrime['Community_Area'].value_counts().mean(), color = 'black', alpha = 0.5) 
plt.xlabel('Total crime')
plt.ylabel('Community Area')
plt.title('Crime per Community Area')


# In[ ]:


c_a = pd.DataFrame(dfcrime['Community_Area'].value_counts())
c_a.columns = ['Number_of_crimes']
c_a.index.name = 'Community_Area'

print('The most dangerous areas in the city of Chicago are:')
print(c_a.sort_values('Number_of_crimes', ascending = False).head(), '\n')

print('The safest areas in the city of chicago are:')
print(c_a.sort_values('Number_of_crimes').head(8))


# It really surprises that the criminal activity of area 25 is almost double than the second most conflictive area (8) or is almost the sum of the second (8) and third (43) together. </b>
# 
# It must be clarified that the area 0 does not really belong to the city of Chicago ... so the area 9 could be considered the safest in Chicago. </b>
# 
# Let's see what percentage of the total criminal activity represent the most dangerous areas of the city.

# In[ ]:


c_a['Percent_%'] = round((c_a['Number_of_crimes']/c_a['Number_of_crimes'].sum())*100,2)

print(c_a.sort_values('Percent_%', ascending = False).head(), '\n')


# As it was shown in the annual study, criminal activity has remained stable and has not changed. But, how has been the evolution over the past 5 years in the differents areas?

# In[ ]:


c_a_last_five = pd.DataFrame(dfcrime['2012':'2016'])
c_a_last_five = pd.DataFrame(c_a_last_five['Community_Area'].value_counts())
c_a_last_five.columns = ['Number_of_crimes']
c_a_last_five.index.name = 'Community_Area'

print('The most dangerous areas in the city of Chicago in the last 5 years (2012 to 2016):')
c_a_last_five['Percent_%'] = round((c_a_last_five['Number_of_crimes']/c_a_last_five['Number_of_crimes'].sum())*100,2)
print(c_a_last_five.sort_values('Percent_%', ascending = False).head(), '\n')

print('The safest areas in the city of Chicago in the last 5 years (2012 to 2016):')
print(c_a_last_five.sort_values('Number_of_crimes').head(8))


# The rankings have changed quite a bit ... let's take a look at the most recent date I have, in 2016

# In[ ]:


c_a_last_year = dfcrime['2016']
c_a_last_year = pd.DataFrame(c_a_last_year['Community_Area'].value_counts())
c_a_last_year.columns = ['Number_of_crimes']
c_a_last_year.index.name = 'Community_Area'

print('The more dangerous areas in the city of Chicago in 2016:')
c_a_last_year['Percent_%'] = round((c_a_last_year['Number_of_crimes']/c_a_last_year['Number_of_crimes'].sum())*100,2)
print(c_a_last_year.sort_values('Percent_%', ascending = False).head(), '\n')

print('The safest areas in the city of Chicago in 2016:')
print(c_a_last_year.sort_values('Number_of_crimes').head(8))


# If we compare the different analyzes carried out, we would have to congratulate Area 43, which in the last year has left the top5 of the areas with the most criminal activity. Area 25 is also reducing this rate. In contrast, Areas 8 and 32 have increased it, area 32 did not appear in the top 5 of the last 5 years. </b>
# 
# The graph represents the evolution of the 15 areas with more registered cases.

# In[ ]:


c_a.index[:15]


# In[ ]:


top_c_a = dfcrime.groupby('Community_Area').resample('Y').count()
top_c_a = top_c_a[['Community_Area']]
top_c_a.columns = ['Sum_C_A']
top_c_a = top_c_a.reset_index()


# In[ ]:


top_c_a= top_c_a.set_index(['date','Community_Area'])
top_c_a.head(1)


# In[ ]:


top_c_a = top_c_a[top_c_a.index.get_level_values('Community_Area').isin([25, 43, 8, 23, 67, 
                                                                         24, 71, 28, 29, 68, 49, 
                                                                         66, 69, 32, 22])]


# In[ ]:


top_c_a.head(1)


# In[ ]:


top_c_a.unstack(level=1).plot(kind='line')
fig=plt.gcf()
fig.set_size_inches(20,10)

plt.xlabel('Year')
plt.ylabel('Total crime')
plt.title('top most dangerous community areas during the years')

plt.legend( loc = 'best')
plt.show()


# As you can see, it follows the same distribution as the annual plot. </b>
# 
# In the previous analytical analysis, it was predicted how some disciplines were evolving, in the visual analysis it can be confirmed that district 25 is decreasing its criminal activity year after year, while areas like 8 and 32 have a tendency to increase it</b>
# 
# In these areas it would not be very good idea to buy an apartment to live in, although the prices may be lower, than  homes of the 9, 47 or 12 areas.
# 
# <a id=14> </a>
# # 5. POLICE DISTRICTS #
# 
# This analysis consists of seeing the amount of work of  each police district

# In[ ]:


dfcrime.District.value_counts()


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x = dfcrime['District'])
plt.axhline(dfcrime['District'].value_counts().mean(), color = 'black', alpha = 0.5)
plt.xlabel('District')
plt.ylabel('Total crime')
plt.title('Crime per police district ')


# It is observed how districts 8 and 11 are those that have more registered criminal activities. </b>
# Let's see what percentage of crime activity has each district.

# In[ ]:


crime['District'].value_counts()[:22].plot(kind='pie',autopct='%1.1f%%')
# Here the decision has been made to use 22 districts since the others did not have almost anything relevant
plt.title('Distribution per district')
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.show()


# Finally, in what refers to police districts, let's see the evolution of each of them over the years.

# In[ ]:


district = dfcrime.groupby('District').resample('Y').count()
district = district[['District']]
district.columns = ['Sum_D']


# In[ ]:


district.unstack(level=0).plot(kind='line')
fig=plt.gcf()
fig.set_size_inches(15,8)
plt.legend( bbox_to_anchor=(0.7, 0, .3, 1), loc=2,
           ncol=2, mode="expand", borderaxespad=0)
plt.xlabel('Year')
plt.ylabel('Total crime')
plt.title('District police evolution over the years')
plt.show()


# This graph shows how dictrite 8 has been the one with the most criminal activity. Although It decreased considerably with respect to others, even going so far as to equalize this rate with district 7 in 2016. </b> 
# 
# During 2016, the district with the largest number of crimes is 11. </b>
# 
# In a later analysis it will be seen what relation these police districts have to the most dangerous community areas.
# 
# <a id=15 ></a>
# # 6.Arrests #
# 
# 
# 

# In[ ]:


crime['Arrest'].value_counts()[:2].plot(kind='pie',autopct='%1.1f%%')
plt.title('Arrests')
fig=plt.gcf()
fig.set_size_inches(5,5)
plt.show()


# This graph could be interpreted in two ways, or most of the criminal activities are not serious enough to trigger an arrest and it is simply a fine without more penalty, or that the author of the crime has not been located.

# In[ ]:


arrest = dfcrime.groupby('Arrest').resample('Y').count()
arrest = arrest[['Arrest']]
arrest.columns = ['Sum_Arrest']


# In[ ]:


ax = arrest.unstack(level=0).plot(kind = 'bar')

plt.legend( bbox_to_anchor=(0.7, 0, .3, 1), loc=2,
           ncol=1, mode="expand", borderaxespad=0)

ax.set_xticklabels(b, rotation = 60)

plt.ylabel('Total')
plt.xlabel('Year')
plt.title('Arrests')
fig=plt.gcf()
fig.set_size_inches(10,5)
plt.show()


# In the bar chart you can see how the arrests, as well as the crimes that are committed have decreased in recent years.
# 
# <a id=16> </a>
# # 7.Type of Crimes #
# 
# Let's see what criminal activities are most common in the city of Chicago and the evolution of the most frequent during the years.

# In[ ]:


dfcrime.Primary_Type.value_counts().head(11)


# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(y = dfcrime['Primary_Type'])
plt.axvline(dfcrime['Primary_Type'].value_counts().mean(), color = 'black', alpha = 0.5) 

plt.ylabel('Crime')
plt.xlabel('Total')
plt.title('Total Crime')


# The analysis will be carried out with the 11 types of crimes that are considerably the most significant.

# In[ ]:


top_type = dfcrime.groupby('Primary_Type').resample('Y').count()
top_type  = top_type [['Primary_Type']]
top_type .columns = ['Sum_type']
top_type  = top_type.reset_index()
top_type = top_type.set_index(['date','Primary_Type'])
top_type .head(1)


# In[ ]:


top_type = top_type[top_type.index.get_level_values('Primary_Type').isin(['THEFT','BATTERY','CRIMINAL DAMAGE',
                                                        'NARCOTICS','OTHER OFFENSE','ASSAULT',
                                                        'BURGLARY','MOTOR VEHICLE THEFT','ROBBERY',
                                                        'DECEPTIVE PRACTICE','CRIMINAL TRESPASS'])]


# In[ ]:


top_type.head(1)


# In[ ]:


top_type.unstack(level=1).plot(kind='line')
fig=plt.gcf()
fig.set_size_inches(15,5)
plt.legend( loc = 'best')
plt.show()


# From this graph we would have to comment several things: </b>
# 
# - The first is, the most common crimes are maintained in the last 5 years, that is they do not increase and decrease (as it was shown in the annual analysis), and that for this case is so bad fot the city. This kind of rate it has to be reduced.
# - They should not be relax when it comes to robberies as it increases a lot during the last year.
# - In regard to narcotics, we must congratulate the city of Chicago, which year after year has been reducing this criminal activity in a very considerable way.
# 
# <a id=17 ></a>
# # 8.Correlacions #
# 
# Filtering the areas with more criminal activity, I will analyze which police districts have an action on the differents areas and also what  types of crimes that are the most common in these communities.
# 
# ## 8.1 Community Areas vs. Police Districts
# 

# In[ ]:


e = dfcrime[['District', 'Community_Area' ]]
e.index = e.Community_Area
del e['Community_Area']

h = e[e.index.get_level_values('Community_Area').isin([25, 43, 8, 23, 67, 
                                                        24, 71, 28, 29, 68, 49, 
                                                        66, 69, 32, 22])]
h = h.reset_index() 
h = pd.DataFrame (h.groupby(['District','Community_Area']).size())
h.columns = ['T_Distric']
h.head()


# In[ ]:


fig = plt.figure()

sns.heatmap(h.unstack(level=0), linewidths=.5, cmap="BuPu",vmin=-100000, vmax=333040)

plt.ylabel('Community Area')
plt.xlabel('District')
plt.title('Community Area vs Police Districts')

fig = plt.gcf()
fig.set_size_inches(15,5)


# 
# Let's remember which police districts have more activity crime and compare.

# In[ ]:


print('Top 10 police districts')
print(dfcrime.District.value_counts().head(10).index)


# In[ ]:


print('Top 10 community Areas')
print(dfcrime.Community_Area.value_counts().head(10).index)


# + All the districts with the most registered crimes have been worked in the communities with the most registered activity
# + Unsual, communities 25 and 8, which had the highest rate, are not controlled by any of these top10 police districts. The 25 is in charge by the district 15 and the 8 is in charge by the district 18.

# <a id=22 ></a>
# ## 8.2 Areas comunitarias vs Delitos

# In[ ]:


Type_vs_C_A = dfcrime[['Primary_Type', 'Community_Area']]

#Filter by community Areas
Type_vs_C_A.index = Type_vs_C_A.Community_Area
del Type_vs_C_A['Community_Area']
Type_vs_C_A = Type_vs_C_A[Type_vs_C_A.index.get_level_values('Community_Area').isin([25, 43, 8, 23, 67, 
                                                        24, 71, 28, 29, 68, 49, 
                                                        66, 69, 32, 22])]
Type_vs_C_A = Type_vs_C_A.reset_index() 

#Filter by Primary_Type
Type_vs_C_A.index = Type_vs_C_A.Primary_Type
del Type_vs_C_A['Primary_Type']
Type_vs_C_A = Type_vs_C_A[Type_vs_C_A.index.get_level_values('Primary_Type').isin(['THEFT','BATTERY','CRIMINAL DAMAGE',
                                                        'NARCOTICS','OTHER OFFENSE','ASSAULT',
                                                        'BURGLARY','MOTOR VEHICLE THEFT','ROBBERY',
                                                        'DECEPTIVE PRACTICE','CRIMINAL TRESPASS'])]
Type_vs_C_A = Type_vs_C_A.reset_index() 


Type_vs_C_A = pd.DataFrame (Type_vs_C_A.groupby(['Primary_Type','Community_Area']).size())
Type_vs_C_A.columns = [' ']

Type_vs_C_A.head()


# In[ ]:


grip = sns.heatmap(Type_vs_C_A.unstack(level=0), linewidths=.5, cmap="YlGnBu", annot=True)

plt.ylabel('Community_Area')
plt.xlabel('Primary Type')
plt.title('Community Area vs Primary Type')

fig = plt.gcf()
fig.set_size_inches(15,5)


#  This correlation shows several conclusions: </b>
#     
#  * Regarding types of crime, we find that robberies, assaults and  drug trafficking are present in almost all areas.
#  * Area 25, which was found to be the one with the most criminal activity, is characterized by its crimes in relation to drug trafficking. Although it is also where there are more aggressions .
#  * Area 8 is where there are more acitivity in relation with theft and this is much higher than the others.
#  
