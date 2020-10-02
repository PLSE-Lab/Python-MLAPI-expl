#!/usr/bin/env python
# coding: utf-8

# # Road Safety Data for the UK
# 
# #### The Data
# The [files](https://data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data) provide detailed road safety data about the circumstances of personal injury road accidents in GB, the types (including Make and Model) of vehicles involved and the consequential casualties. The statistics relate only to personal injury accidents on public roads that are reported to the police, and subsequently recorded, using the STATS19 accident reporting form. The files used here span 2013 to 2017.

# # Table of Contents
# <a id='Table of Contents'></a>
# 
# ### <a href='#1. Obtaining and Viewing the Data'>1. Obtaining and Viewing the Data</a>
# 
# ### <a href='#2. Preprocessing the Data'>2. Preprocessing the Data</a>
# 
# * <a href='#2.1. Converting Datetime Column'>2.1. Converting Datetime Column</a>
# * <a href='#2.2. Handling Missing Values'>2.2. Handling Missing Values</a>
# 
# ### <a href='#3. Exploratory Data Analysis (EDA)'>3. Exploratory Data Analysis (EDA)</a>
# 
# * <a href='#3.1. Main Characteristics of Accidents'>3.1. Main Characteristics of Accidents</a>
# * <a href='#3.2. Main Characteristics of Vehicles'>3.2. Main Characteristics of Vehicles</a>

# ### 1. Obtaining and Viewing the Data
# <a id='1. Obtaining and Viewing the Data'></a>

# In[1]:


# import the usual suspects ...
import pandas as pd
import numpy as np
import glob

import matplotlib.pyplot as plt
import seaborn as sns

# suppress all warnings
import warnings
warnings.filterwarnings("ignore")


# **Accidents**

# In[2]:


accidents = pd.read_csv('../input/Accident_Information.csv')
print('Records:', accidents.shape[0], '\nColumns:', accidents.shape[1])
accidents.head()


# In[3]:


#accidents.info()


# > **Vehicles**

# In[4]:


vehicles = pd.read_csv('../input/Vehicle_Information.csv', encoding='ISO-8859-1')
print('Records:', vehicles.shape[0], '\nColumns:', vehicles.shape[1])
vehicles.head()


# In[5]:


#vehicles.info()


# *Back to: <a href='#Table of Contents'> Table of Contents</a>*
# ### 2. Preprocessing the Data
# <a id='2. Preprocessing the Data'></a>

# #### 2.1. Converting Datetime Column
# <a id='2.1. Converting Datetime Column'></a>

# We had our `Date` column with values not properly stored in the correct format. Let's do this now:

# In[6]:


accidents['Date']= pd.to_datetime(accidents['Date'], format="%Y-%m-%d")


# In[7]:


# check
accidents.iloc[:, 5:13].info()


# - Morning Rush from 5am to 10am
# - Office Hours from 10am to 3pm (or: 10:00 - 15:00)
# - Afternoon Rush from 3pm to 7pm (or: 15:00 - 19:00)
# - Evening from 7pm to 11pm (or: 19:00 - 23:00)
# - Night from 11pm to 5am (or: 23:00 - 05:00)

# In[8]:


# slice first and second string from time column
accidents['Hour'] = accidents['Time'].str[0:2]

# convert new column to numeric datetype
accidents['Hour'] = pd.to_numeric(accidents['Hour'])

# drop null values in our new column
accidents = accidents.dropna(subset=['Hour'])

# cast to integer values
accidents['Hour'] = accidents['Hour'].astype('int')


# In[9]:


# define a function that turns the hours into daytime groups
def when_was_it(hour):
    if hour >= 5 and hour < 10:
        return "morning rush (5-10)"
    elif hour >= 10 and hour < 15:
        return "office hours (10-15)"
    elif hour >= 15 and hour < 19:
        return "afternoon rush (15-19)"
    elif hour >= 19 and hour < 23:
        return "evening (19-23)"
    else:
        return "night (23-5)"


# In[10]:


# apply thus function to our temporary hour column
accidents['Daytime'] = accidents['Hour'].apply(when_was_it)
accidents[['Time', 'Hour', 'Daytime']].head(8)


# In[11]:


# drop old time column and temporary hour column
#accidents = accidents.drop(columns=['Time', 'Hour'])


# #### 2.2. Handling Missing Values
# <a id='2.2. Handling Missing Values'></a>

# In[12]:


print('Proportion of Missing Values in Accidents Table:', 
      round(accidents.isna().sum().sum()/len(accidents),3), '%')


# In[14]:


#accidents.isna().sum()


# In[15]:


print('Proportion of Missing Values in Vehicles Table:', 
      round(vehicles.isna().sum().sum()/len(vehicles),3), '%')


# In[17]:


#vehicles.isna().sum()


# *Back to: <a href='#Table of Contents'> Table of Contents</a>*
# ### 3. Exploratory Data Analysis (EDA)
# <a id='3. Exploratory Data Analysis (EDA)'></a>

# #### 3.1. Main Characteristics of Accidents 
# <a id='3.1. Main Characteristics of Accidents'></a>

# ***Has the number of accidents increased or decreased over the last few years?***

# In[18]:


# prepare plot
sns.set_style('white')
fig, ax = plt.subplots(figsize=(15,6))

# plot
accidents.set_index('Date').resample('M').size().plot(label='Total per Month', color='grey', ax=ax)
accidents.set_index('Date').resample('M').size().rolling(window=10).mean()                           .plot(color='darkorange', linewidth=5, label='10-Months Moving Average', ax=ax)

ax.set_title('Accidents per Month', fontsize=14, fontweight='bold')
ax.set(ylabel='Total Count\n', xlabel='')
ax.legend(bbox_to_anchor=(1.1, 1.1), frameon=False)

# remove all spines
sns.despine(ax=ax, top=True, right=True, left=True, bottom=False);


# In[19]:


yearly_count = accidents['Date'].dt.year.value_counts().sort_index(ascending=False)

# prepare plot
sns.set_style('white')
fig, ax = plt.subplots(figsize=(12,5))

# plot
ax.bar(yearly_count.index, yearly_count.values, color='lightsteelblue')
ax.plot(yearly_count, linestyle=':', color='black')
ax.set_title('\nAccidents per Year\n', fontsize=14, fontweight='bold')
ax.set(ylabel='\nTotal Counts')

# remove all spines
sns.despine(ax=ax, top=True, right=True, left=True, bottom=True);


# ***On which weekdays are accidents most likely to be caused?***

# - Preparing dataframe that calculates average accidents per weekday:

# In[20]:


weekday_counts = pd.DataFrame(accidents.set_index('Date').resample('1d')['Accident_Index'].size().reset_index())
weekday_counts.columns = ['Date', 'Count']
#weekday_counts

weekday = weekday_counts['Date'].dt.weekday_name
#weekday

weekday_averages = pd.DataFrame(weekday_counts.groupby(weekday)['Count'].mean().reset_index())
weekday_averages.columns = ['Weekday', 'Average_Accidents']
weekday_averages.set_index('Weekday', inplace=True)
weekday_averages


# - Plotting this dataframe:

# In[21]:


# reorder the weekdays beginning with Monday (backwards because of printing behavior!)
days = ['Sunday', 'Saturday', 'Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday']

# prepare plot
sns.set_style('white')
fig, ax = plt.subplots(figsize=(10,5))
colors=['lightsteelblue', 'lightsteelblue', 'navy', 'lightsteelblue', 
        'lightsteelblue', 'lightsteelblue', 'lightsteelblue']

# plot
weekday_averages.reindex(days).plot(kind='barh', ax=ax, color=[colors])
ax.set_title('\nAverage Accidents per Weekday\n', fontsize=14, fontweight='bold')
ax.set(xlabel='\nAverage Number', ylabel='')
ax.legend('')

# remove all spines
sns.despine(ax=ax, top=True, right=True, left=True, bottom=True);


# - Preparing another dataframe by weekday and year:

# In[22]:


weekday = accidents['Date'].dt.weekday_name
year    = accidents['Date'].dt.year

accident_table = accidents.groupby([year, weekday]).size()
accident_table = accident_table.rename_axis(['Year', 'Weekday'])                               .unstack('Weekday')                               .reindex(columns=days)
accident_table


# - Plotting this second dataframe:

# In[23]:


plt.figure(figsize=(10,6))
sns.heatmap(accident_table, cmap='Reds')
plt.title('\nAccidents by Years and Weekdays\n', fontsize=14, fontweight='bold')
plt.xlabel('')
plt.ylabel('');


# ***How are accidents related to weather conditions?***

# In[24]:


accidents.Weather_Conditions.value_counts(normalize=True)


# *As most of the days the `Weather_Condition` is "fine" (=1), most accidents will likely to be happen then.*

# ***What percentage of each category of accident severity do we have?***

# In[26]:


accidents.Accident_Severity.value_counts()


# In[27]:


# assign the data
fatal   = accidents.Accident_Severity.value_counts()['Fatal']
serious = accidents.Accident_Severity.value_counts()['Serious']
slight  = accidents.Accident_Severity.value_counts()['Slight']

names = ['Fatal Accidents','Serious Accidents', 'Slight Accidents']
size  = [fatal, serious, slight]
#explode = (0.2, 0, 0)

# create a pie chart
plt.pie(x=size, labels=names, colors=['red', 'darkorange', 'silver'], 
        autopct='%1.2f%%', pctdistance=0.6, textprops=dict(fontweight='bold'),
        wedgeprops={'linewidth':7, 'edgecolor':'white'})

# create circle for the center of the plot to make the pie look like a donut
my_circle = plt.Circle((0,0), 0.6, color='white')

# plot the donut chart
fig = plt.gcf()
fig.set_size_inches(8,8)
fig.gca().add_artist(my_circle)
plt.title('\nAccident Severity: Share in % (2013-2017)', fontsize=14, fontweight='bold')
plt.show()


# ***How has the number of fatalities developed over the years?***

# In[29]:


# set the criterium to slice the fatalaties
criteria = accidents['Accident_Severity']=='Fatal'
# create a new dataframe
weekly_fatalities = accidents.loc[criteria].set_index('Date').sort_index().resample('W').size()

# prepare plot
sns.set_style('white')
fig, ax = plt.subplots(figsize=(14,6))

# plot
weekly_fatalities.plot(label='Total Fatalities per Month', color='grey', ax=ax)
plt.fill_between(x=weekly_fatalities.index, y1=weekly_fatalities.values, color='grey', alpha=0.3)
weekly_fatalities.rolling(window=10).mean()                           .plot(color='darkorange', linewidth=5, label='10-Months Moving Average', ax=ax)

ax.set_title('\nFatalities', fontsize=14, fontweight='bold')
ax.set(ylabel='\nTotal Count', xlabel='')
ax.legend(bbox_to_anchor=(1.2, 1.1), frameon=False)

# remove all spines
sns.despine(ax=ax, top=True, right=True, left=True, bottom=True);


# ***Is the share of fatal accidents increasing or decreasing?***

# In[30]:


sub_df = accidents[['Date', 'Accident_Index', 'Accident_Severity']]

# pull out the year
year = sub_df['Date'].dt.year
week = sub_df['Date'].dt.week

# groupby year and severities
count_of_fatalities = sub_df.set_index('Date').groupby([pd.Grouper(freq='W'), 'Accident_Severity']).size()

# build a nice table
fatalities_table = count_of_fatalities.rename_axis(['Week', 'Accident_Severity'])                                      .unstack('Accident_Severity')                                      .rename({1:'fatal', 2:'serious', 3:'slight'}, axis='columns')
fatalities_table.head()


# In[31]:


fatalities_table['sum'] = fatalities_table.sum(axis=1)
fatalities_table = fatalities_table.join(fatalities_table.div(fatalities_table['sum'], axis=0), rsuffix='_percentage')
fatalities_table.head()


# In[33]:


# prepare data
sub_df = fatalities_table[['Fatal_percentage', 'Serious_percentage', 'Slight_percentage']]

# prepare plot
sns.set_style('white')
fig, ax = plt.subplots(figsize=(14,6))
colors=['black', 'navy', 'lightsteelblue']

# plot
sub_df.plot(color=colors, ax=ax)
ax.set_title('\nProportion of Accidents Severity\n', fontsize=14, fontweight='bold')
ax.set(ylabel='Share on all Accidents\n', xlabel='')
ax.legend(labels=['Fatal Accidents', 'Serious Accidents', 'Slight Accidents'], 
          bbox_to_anchor=(1.3, 1.1), frameon=False)

# remove all spines
sns.despine(top=True, right=True, left=True, bottom=False);


# *The trend for fatal accidents seems to stagnate.*

# ***How are accidents distributed throughout the day?***

# - Distribution of Hours

# In[34]:


# prepare plot
sns.set_style('white')
fig, ax = plt.subplots(figsize=(10,6))

# plot
accidents.Hour.hist(bins=24, ax=ax, color='lightsteelblue')
ax.set_title('\nAccidents depending by Time\n', fontsize=14, fontweight='bold')
ax.set(xlabel='Hour of the Day', ylabel='Total Count of Accidents')

# remove all spines
sns.despine(top=True, right=True, left=True, bottom=True);


# - Counts of Accidents by Daytime

# In[35]:


# prepare dataframe
order = ['night (23-5)', 'evening (19-23)', 'afternoon rush (15-19)', 'office hours (10-15)', 'morning rush (5-10)']
df_sub = accidents.groupby('Daytime').size().reindex(order)

# prepare barplot
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['lightsteelblue', 'lightsteelblue', 'navy', 'lightsteelblue', 'lightsteelblue']

# plot
df_sub.plot(kind='barh', ax=ax, color=colors)
ax.set_title('\nAccidents depending by Daytime\n', fontsize=14, fontweight='bold')
ax.set(xlabel='\nTotal Count of Accidents', ylabel='')

# remove all spines
sns.despine(top=True, right=True, left=True, bottom=True);


# - Share of Accident Severity by Daytime

# In[40]:


# prepare dataframe with simple counts
counts = accidents.groupby(['Daytime', 'Accident_Severity']).size()

counts = counts.rename_axis(['Daytime', 'Accident_Severity'])                                .unstack('Accident_Severity')                                .rename({1:'fatal', 2:'serious', 3:'slight'}, axis='columns')
counts


# In[41]:


# prepare dataframe with shares
counts['sum'] = counts.sum(axis=1)
counts = counts.join(counts.div(counts['sum'], axis=0), rsuffix=' in %')
counts_share = counts.drop(columns=['Fatal', 'Serious', 'Slight', 'sum', 'sum in %'], axis=1)
counts_share


# In[42]:


# prepare barplot
fig, ax = plt.subplots(figsize=(10, 5))

# plot
counts_share.reindex(order).plot(kind='barh', ax=ax, stacked=True, cmap='cividis')
ax.set_title('\nAccident Severity by Daytime\n', fontsize=14, fontweight='bold')
ax.set(xlabel='Percentage', ylabel='')
ax.legend(bbox_to_anchor=(1.25, 0.98), frameon=False)

# remove all spines
sns.despine(top=True, right=True, left=True, bottom=True);


# *Back to: <a href='#Table of Contents'> Table of Contents</a>*
# #### 3.2. Main Characteristics of Vehicles 
# <a id='3.2. Main Characteristics of Vehicles'></a>

# In[43]:


#vehicles.describe().T


# ***What are the age and gender of the drivers who cause an accident?***

# In[44]:


vehicles.Sex_of_Driver.value_counts(normalize=True)


# *We'll have to keep in mind that two-thirds of the drivers are male --> imbalanced classes!*

# In[47]:


# create a new dataframe
drivers = vehicles.groupby(['Age_Band_of_Driver', 'Sex_of_Driver']).size().reset_index()

# drop the values that have no value
drivers.drop(drivers[(drivers['Age_Band_of_Driver'] == 'Data missing or out of range') |                      (drivers['Sex_of_Driver'] == 'Not known') |                      (drivers['Sex_of_Driver'] == 'Data missing or out of range')]                     .index, axis=0, inplace=True)
# rename the columns
drivers.columns = ['Age_Band_of_Driver', 'Sex_of_Driver', 'Count']
drivers


# In[48]:


# seaborn barplot
fig, ax = plt.subplots(figsize=(14, 7))
sns.barplot(y='Age_Band_of_Driver', x='Count', hue='Sex_of_Driver', data=drivers, palette='bone')
ax.set_title('\nAccidents Cars\' Drivers by Age and Sex\n', fontsize=14, fontweight='bold')
ax.set(xlabel='Count', ylabel='Age Band of Driver')
ax.legend(bbox_to_anchor=(1.1, 1.), borderaxespad=0., frameon=False)

# remove all spines
sns.despine(top=True, right=True, left=True, bottom=True);


# ***Which type of manoeuvre is often involved in accidents?***

# In[66]:


#vehicles.Vehicle_Manoeuvre.value_counts()


# In[63]:


# prepare dataframe
df_plot = vehicles.groupby('Vehicle_Manoeuvre').size()                                                .reset_index(name='counts')                                                    .sort_values(by='counts', ascending=False)
                                                        
df_plot = df_plot[df_plot.counts > 80000]
df_plot


# In[64]:


# library for plooting a tree map
import squarify


# In[67]:


# prepare plot
labels = df_plot.apply(lambda x: str(x[0]) + "\n (" + str(x[1]) + ")", axis=1)
sizes = df_plot['counts'].values.tolist()
colors = [plt.cm.Pastel1(i/float(len(labels))) for i in range(len(labels))]

# plot
plt.figure(figsize=(8,6), dpi= 80)
squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)

# Decorate
plt.title('\nTreemap of Vehicle Manoeuvre\n', fontsize=14, fontweight='bold')
plt.axis('off')
plt.show()


# In[ ]:




