#!/usr/bin/env python
# coding: utf-8

# Bakery is based on Edinburg old town.

# ### Importing all libraries

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from pylab import figure, show, legend, ylabel
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import seaborn as sns

import numpy as np
from scipy import stats

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,  iplot
init_notebook_mode(connected=True)


# ### The following are some of the questions I want to answer
# - Can I predict future sales?
# 
# - Could I explain the cause for unnormal periods where sales were really high or low?
# 
# - Can we predict repetition of those events based maybe on weather or local events?<br>
# In order to be able to answer the questions above I will also use **weather dataset** as well as **local events data**.

# ### Opening the Bakery dataset

# In[ ]:


bakery = pd.read_csv('../input/transactions-from-a-bakery/BreadBasket_DMS.csv')
bakery.head()


# In[ ]:


bakery.shape


# ### See total sells per item

# In[ ]:


#prints count of each item
print(bakery['Item'].value_counts().tail(10))


# In[ ]:


#prints count of each item
print(bakery['Item'].value_counts().head(10))


# - There is an item called 'NONE' there are in total 786 rows of it and one item called adjustment, but only one row of it, What are they? <br>This is what I see.

# In[ ]:


bakery['Item'].value_counts().head(6)


# In[ ]:


bakery[bakery['Transaction']==348]


# In[ ]:


bakery[bakery['Item']== 'NONE'].head(3)


# ### 'NONE' must be cancellation or errors, I am going to remove. I am going to remove the adjustments as well for the same reason.

# In[ ]:


bakery = bakery[bakery['Item'] != 'NONE']


# In[ ]:


bakery[bakery['Item'] == 'NONE']


# In[ ]:


bakery[bakery['Item'] == 'Adjustment']


# In[ ]:


bakery[bakery['Transaction'] == 938]


# In[ ]:


bakery = bakery[bakery['Item'] != 'Adjustment']


# ### Checking to see if there are missing values

# In[ ]:


#we do not have mising values
bakery.isnull().sum()


# ### The first and last date in bakery dataset

# In[ ]:


bakery['Date'].min()


# In[ ]:


bakery['Date'].max()


# In[ ]:


print(bakery.head())
bakery.dtypes
#Date is an object, need to chage to proper date


# ### Need to change date and time to datetime

# In[ ]:


#covert to datetime
bakery['Date_Time'] = pd.to_datetime(bakery['Date'].apply(str)+' '+bakery['Time'],format="%Y/%m/%d %H:%M:%S")

#todatetime 
print(bakery.dtypes)


# In[ ]:


bakery.head()


# ### Adding day of the week, to see which day is the most or least popular and similar for months

# In[ ]:


bakery['Day_of_Week'] = bakery['Date_Time'].dt.weekday_name


# ### Save clean dataset to excel As I like to explore in more detail the items list.

# In[ ]:


bakery.to_excel('cleanBakeryDF.xlsx', sheet_name='Sheet1')


# - Adding month

# In[ ]:


bakery['Month'] = bakery['Date_Time'].dt.month


# In[ ]:


#First month should be October
bakery['Month'].head(1)


# In[ ]:


#Last month should be April
bakery['Month'].tail(1)


# - This will add another column representing month in order

# In[ ]:


#Dictionary to map months in order
mo = {10 : 1, 11 : 2, 12 : 3 , 1 : 4 , 2 : 5 , 3 : 6 , 4 : 7}
 
m = bakery['Month']
bakery['Month_Order'] = m.map(mo)


# - Adding seasons

# In[ ]:


#adding season
##Dictionary to map month to season
x = {1 : 'Winter', 2 :'Winter', 3 :'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn',12:'Winter'}

y = bakery['Month']
bakery['Season'] = y.map(x)


# In[ ]:


bakery.head(2)


# - Exploring sales for the first day
# Coffee is the most sold item, followed by bread 

# In[ ]:


bakery.loc[(pd.to_datetime(bakery['Date_Time'].dt.date) == '2016-10-30')].groupby(['Item'])['Transaction'].count().plot.bar()


# There is one transaction at 1:21 in the morning. That currently is not included in the sessions. It was the sale of one bread.
# 
# **I may add it to afternoon session for completeness****

# In[ ]:


bakery[bakery['Time'] == '01:21:05']


# - Adding another column for hour, to see the most/least busy hours. This info can be use to determine number of staff that needs working.<br> Around midday is when it is most busy. Lunch time

# In[ ]:


bakery['Hour'] = bakery['Date_Time'].dt.hour


# In[ ]:


bakery['Hour'].value_counts()


# - Adding sessions as well.

# In[ ]:


#Dictionary to map session
t = {7 : 'Morning', 8 :'Morning', 9 :'Morning',10:'Morning',11:'Morning',12:'Morning',13:'Afternoon',14:'Afternoon',15:'Afternoon',16:'Afternoon',17:'Afternoon',18:'Afternoon',19:'Evening',20:'Evening',21:'Evening',22:'Evening',23:'Evening'}

h = bakery['Hour']
bakery['Session'] = h.map(t)


#  - I am creating a dictionary getting data from excel. Mapping items to category, which is food or drink.

# In[ ]:


#adding categories to items
from __future__ import print_function
from os.path import join, dirname, abspath
import xlrd

d = {}
wb = xlrd.open_workbook('../input/item-dic/items_dictionary.xlsx')
sh = wb.sheet_by_name('sheet1') 

for i in range(sh.nrows):
    cell_value_id = sh.cell_value(i,0)
    cell_value_class = sh.cell_value(i,1)
    d[cell_value_id] = cell_value_class
    


# In[ ]:


it = bakery['Item']
bakery['Category'] = it.map(d)


# In[ ]:


bakery.head(1)


# ### Overall the bakery sells more food than drinks, as expected

# In[ ]:


bakery.loc[(pd.to_datetime(bakery['Date_Time'].dt.date) == '2016-10-30')].groupby(['Category'])['Transaction'].count().plot.bar()


# ### Extracting a few more features from datetime to see if can get any more info.

# In[ ]:


bakery['Hourly'] = bakery['Date_Time'].dt.to_period('H')


# In[ ]:


bakery['Hourly'] = pd.to_datetime(bakery['Hourly'].apply(str),format="%Y/%m/%d %H:%M:%S")


# In[ ]:


bakery['Monthly'] = pd.to_datetime(bakery['Date_Time']).dt.to_period('M')


# In[ ]:


bakery['Weekly'] = pd.to_datetime(bakery['Date_Time']).dt.to_period('W')


# ### Loading temperature dataset.

# In[ ]:


temp_data = pd.read_csv('../input/temperature-data/temp_data.csv')


# ### Changing datatype of hourly in temperature dataset to datetime

# In[ ]:


temp_data['Hourly'] = pd.to_datetime(temp_data['Hourly'].apply(str), format = '%Y/%m/%d %H:%M:%S')


# In[ ]:


temp_data.head()


# In[ ]:


temp_data.isnull().sum()


# In[ ]:


dates_for_graph =bakery['Date'].value_counts().sort_index()


# ### Merging the two datasets.

# In[ ]:


bakery_temp = pd.merge(bakery, temp_data, on='Hourly', how='left')


# In[ ]:


bakery_temp.describe()


# As we have a few missing hours in temperature dataset, adding the last seen value.

# In[ ]:


bakery_temp.fillna(method='ffill', inplace=True)


# In[ ]:


bakery_temp.groupby('Date')['temperature'].min().head(10)


# - Interesting results below. it seems that often there is one busy period follow by periods which are not as busy. This behaviour keep repeating, maybe there is a day of the week that is most busy. Also I like to find out what date was the lowest item sold, it seems as almost zero. <br>Can I explain why?

# In[ ]:


trace1 = go.Scatter(
    x = bakery_temp.groupby('Date')['Item'].count().index,
    y = bakery_temp.groupby('Date')['Item'].count().values,
    mode = 'lines+markers',
    name = 'lines+markers')

data = [trace1]
layout = go.Layout(title = 'Daily Sales')
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# ### Sales and temperature graph, I am going to use the minimum temperatures

# In[ ]:


fig1 = figure(figsize=(15,6))
 
# and the first axes using subplot populated with data 
ax1 = fig1.add_subplot(1,1,1)
ax1.set_title("Sales and Temperature")
line1 = ax1.plot(bakery_temp.groupby('Date')['Item'].count(), 'o-')

ylabel("Total Sales per day")
 
# now, the second axes that shares the x-axis with the ax1
ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
line2 = ax2.plot(bakery_temp.groupby('Date')['temperature'].min(), 'xr-')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ylabel("Mininum temperature per day")


blue_line = mlines.Line2D([], [], color='blue', marker='o',
                          markersize=6, label='Sales')

red_line = mlines.Line2D([], [], color='red', marker='*',
                          markersize=6, label='Temperature')
plt.legend(handles=[blue_line,red_line])
#no working but leave it as it removes the axix labels
ax1.get_xaxis().set_major_locator(mdates.MonthLocator(interval=1))

show()


# - Temperature does not explain outliner. We see that some days with very low temperature had high sales and the other way around.
# - There is seasonality, as Saturday is the most busy day of the week
# - Can events explain outliners?
# 

# Can we see any relations if we only have sales of hot drinks and temperature

# In[ ]:



fig1 = figure(figsize=(15,6))
 
#the first axes using subplot populated with data 
ax1 = fig1.add_subplot(111)
ax1.set_title("Hot drinks sales and Temperature")
line1 = ax1.plot(bakery_temp[bakery_temp['Item'].isin(['Coffee', 'Tea', 'Hot Chocolate'])] .groupby('Date')['Item'].count(),'o-')
ylabel("Total Sales per day")
 
# now, the second axes that shares the x-axis with the ax1
ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
line2 = ax2.plot(bakery_temp.groupby('Date')['temperature'].min(), 'xr-')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ylabel("Mininum temperature per day")

#inverted Axis
ax2.invert_yaxis()

blue_line = mlines.Line2D([], [], color='blue', marker='o',
                          markersize=6, label='Sales')

red_line = mlines.Line2D([], [], color='red', marker='*',
                          markersize=6, label='Temperature')
plt.legend(handles=[blue_line,red_line])

ax1.get_xaxis().set_major_locator(mdates.MonthLocator(interval=1))

show()


# - It seems that when it is really cold, sometimes there are more hot drinks sales.
# - My guess is that people from Edinburg are used to very cold weather and it makes no differences to their day to day life. I could not do the same comparison with hot temperatures and cold drinks as we have limited data. But it will be very interesting to see.

# ### Formally try to see if there is a correlation for all items sold per hour and min temperature. No really.

# In[ ]:


item_hour = bakery_temp.groupby('Hourly')['Item'].count().values

temp_hour = bakery_temp.groupby('Hourly')['temperature'].min().values


# In[ ]:


hot_drink_df = bakery_temp[bakery_temp['Item'].isin(['Coffee', 'Tea', 'Hot Chocolate'])]


# In[ ]:


hot_drink_item_hour = hot_drink_df.groupby('Hourly')['Item'].count().values
hot_drink_temp_hour = hot_drink_df.groupby('Hourly')['temperature'].min().values


# In[ ]:


correlation, p_value = stats.pearsonr(temp_hour,item_hour)
print(correlation)
print(p_value)


# In[ ]:


plt.scatter(temp_hour, item_hour)


# In[ ]:


correlation, p_value = stats.pearsonr(hot_drink_temp_hour,hot_drink_item_hour)
print(correlation)
print(p_value)


# In[ ]:


plt.scatter(hot_drink_temp_hour, hot_drink_item_hour)


# ### Let's try to find out 
# - How many items were sold daily
# - days with most sales
# - days with least sales

# The date with only one sold item was the 1 Jan 2017

# In[ ]:


bakery_temp.groupby(['Date','Day_of_Week'])['Item'].count().sort_values()


# aha, mystery solved!!!!. After parting for the new years, one of the employees with a key to the bakery remember that they did not have bread for next day dinner party celebration so on the way home got in a buy one break. As it was new year's the bakery was close.

# In[ ]:


bakery_temp[bakery_temp['Date'] == '2017-01-01']


# ### Yes, Saturday is the most busy day.

# In[ ]:


bakery_temp.groupby('Day_of_Week')['Item'].count().plot.pie()


# In[ ]:


bakery_temp.groupby('Day_of_Week')['Item'].count().sort_values()


# ### We can see here the most busiest times of the day as well as the busiest day of the week.

# In[ ]:


fig, axes = plt.subplots(3, 2, figsize=(15,6), sharex=True, sharey=True, squeeze=False )

fig.suptitle('Total Sales by Hour and Day', fontsize=12)
fig.text(0.06, 0.5, 'Total Item Sold', ha='center', va='center', rotation='vertical')
#fig.text(0.5, 0.04, 'Hours', ha='center', va='center')
Saturday = bakery_temp[bakery_temp['Day_of_Week'] == 'Saturday'].groupby('Hour')['Item'].count()
Saturday.plot(ax=axes[0][0], grid=True, kind='area', title='Saturday', xticks=range(6,24,1), yticks=range(0, 1000,200))

#Removing the item sold at 1:20 in the morning
Sunday = bakery_temp[(bakery_temp['Date'] != '2017-01-01') & (bakery_temp['Day_of_Week'] == 'Sunday')].groupby('Hour')['Item'].count()
Sunday.plot(ax=axes[0][1], grid=True, kind='area', title='Sunday', xticks=range(6,24,1), yticks=range(0, 1000,200))

Monday = bakery_temp[bakery_temp['Day_of_Week'] == 'Monday'].groupby('Hour')['Item'].count()
Monday.plot(ax=axes[1][0], grid=True, kind='area', title='Monday', xticks=range(6,24,1), yticks=range(0, 1000,200))

Tuesday = bakery_temp[bakery_temp['Day_of_Week'] == 'Tuesday'].groupby('Hour')['Item'].count()
Tuesday.plot(ax=axes[1][1], grid=True, kind='area', title='Tuesday', xticks=range(6,24,1), yticks=range(0, 1000,200))

Thursday = bakery_temp[bakery_temp['Day_of_Week'] == 'Thursday'].groupby('Hour')['Item'].count()
Thursday.plot(ax=axes[2][0], grid=True, kind='area', title='Thursday', xticks=range(6,24,1), yticks=range(0, 1000,200))

Friday = bakery_temp[bakery_temp['Day_of_Week'] == 'Friday'].groupby('Hour')['Item'].count()
Friday.plot(ax=axes[2][1], grid=True, kind='area', title='Friday', xticks=range(6,24,1), yticks=range(0, 1000,200))


# - Not all Saturdays are open until 22:00, Maybe worth checking if the days with longer opening hours where due to local festivities
# - For all days, the volume of sales after six o'clock is very small. is it worth keeping the Bakery open after 6?

# - The graph below confirms the findings, the outliner at the bottom refer to the event on Jan 1st. 
# - The outliner at the top refer to some Saturday sales where there were local events.
# Local events data was taken from http://www.edinburghguide.com/events/
# Actual dates '2016-11-05','2016-11-12','2017-01-28','2017-02-04','2017-02-18','2017-03-04'

# In[ ]:


bakery_temp.groupby('Date')['Item'].count().plot.box()


# In[ ]:


bakery_temp[(bakery_temp['Time'] > '20:00:00') & (bakery_temp['Day_of_Week'] == 'Saturday')].groupby(['Date','Day_of_Week','Time'])['Item'].count()


# In[ ]:


bakery_temp[(bakery_temp['Time'] > '18:00:00') & (bakery_temp['Day_of_Week'] == 'Friday')].groupby(['Date','Day_of_Week'])['Item'].count()


# In[ ]:


bakery_temp[(bakery_temp['Time'] > '17:00:00') & (bakery_temp['Day_of_Week'] == 'Friday')].groupby(['Date','Day_of_Week'])['Item'].count()


# In[ ]:


bakery_temp.shape


# Checking what sold the most, food or drink

# In[ ]:


#unstack, will put the categories in columns
bakery_temp.groupby(['Month_Order','Category'])['Category'].count().unstack()


# In[ ]:



fig, ax = plt.subplots()
bakery_temp.groupby(['Month_Order','Category'])['Category'].count().unstack().plot(kind='bar', figsize=(15,6), ax=ax)
ax.set_title('Monthly Sales', fontsize=21, y=1.01)
ax.legend(loc="upper right")
ax.set_ylabel('Sales', fontsize=16)
ax.set_xlabel('Category', fontsize=16)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.show()


# This will display the total sales per month. Can we see why the last and the first have the least sales? Also, we can see that December and January are the months with less sales. Maybe December and Jan have lots of bank holidays? 

# In[ ]:


bakery_temp.groupby('Month')['Date'].nunique()


# ### Can see what are the top item per day?

# In[ ]:


#this will give me a list with unique names of item
item_Name = bakery_temp['Item'].value_counts().index
#this will give me the values of the unique item name
item_Value = bakery_temp['Item'].value_counts().values


# In[ ]:


#this will give me a list with unique names of item
bakery_saturday = bakery_temp[bakery_temp['Day_of_Week'] == 'Saturday']
item_Name_Saturday = bakery_saturday['Item'].value_counts().index
#this will give me the values of the unique item name
item_Value_Saturday = bakery_saturday['Item'].value_counts().values


# In[ ]:


#this will give me a list with unique names of item
bakery_monday = bakery_temp[bakery_temp['Day_of_Week'] == 'Monday']
item_Name_monday = bakery_monday['Item'].value_counts().index
#this will give me the values of the unique item name
item_Value_monday = bakery_monday['Item'].value_counts().values


# In[ ]:


item_Value_Saturday[10:].sum()


# In[ ]:


item_Value_monday[:10]


# In[ ]:


item_Name_monday[:10]


# In[ ]:


item_Value_Saturday[:10]


# In[ ]:


item_Name_Saturday[:10]


# In[ ]:


#Top 10 items plus aggregating the rest as others


# In[ ]:


item_Saturday_Value = [1103,  760,  288,  246,  166,  161,  146,  146,  143,  118, 1328]


# In[ ]:


item_Saturday_Name = ['Coffee', 'Bread', 'Tea', 'Cake', 'Pastry', 'Sandwich', 'Hot chocolate',
       'Scone', 'Medialuna', 'Scandinavian', 'Other']


# In[ ]:


plt.figure(figsize=(12,4))
plt.ylabel('Values', fontsize='medium')
plt.xlabel('Items', fontsize='medium')
plt.title('10 Most sold itme')
plt.bar(item_Name[:10],item_Value[:10], width = 0.7, color="blue",linewidth=0.4)

plt.xticks(rotation=45)
plt.show()


# In[ ]:


init_notebook_mode(connected=True)

labels = item_Name_Saturday[:10]
values = item_Value_Saturday[:10]

trace = go.Pie(labels=labels, values=values)

data= [trace]
layout = go.Layout(title = 'Top 10 item sold on Saturday')
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# We can use the Saturday percentage of sales with Saturday sale prediction as a guidance for the bakery stock.

# In[ ]:



labels = item_Saturday_Name
values = item_Saturday_Value

trace = go.Pie(labels=labels, values=values)

data= [trace]
layout = go.Layout(title = 'All Items sold on Saturday')
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# In[ ]:


bakery_temp['Session'].value_counts()


# Saving dataframe keeping data type

# In[ ]:


bakery_temp.to_pickle('bakery_temp_dataframe.pkl')


# ### Creating a data frame with only the features that are needed and then saving to pickle file

# In[ ]:


#Extract dates as we want then to be the index
dates = pd.DatetimeIndex(bakery_temp['Date_Time'])


# ### Will leave temperature just in case

# In[ ]:


bakery_temp_sum = bakery_temp[['Date_Time','Date','Item','Day_of_Week','temperature']].copy()


# In[ ]:


bakery_temp_sum.head(10)


# In[ ]:


bakery_temp_sum.to_pickle('bakery_temp_sum_dataframe.pkl')


# ### Some of the added features such as season, session, month, etc where not really used as this dataset has only data for four full months. 

# ### Below are the most important business facts discovered
# - Data only from the 30th October 2016 to the 9th of March 2017
# - The bakery sales 96 different items in total, 85 are foods and 6 drinks as well as others for cancellations and adjustment
# - The bakery opens 7 days a week
# - It was closed only 4 days during the 6 month period. Those days were the 25th and 26th of December 2016 and the 1st and 2nd of January 2017
# - Coffee is the most sold item followed by bread
# - There are a few item that have been sold once or twice only, maybe there are worth stop selling those, or at least item that have a short life date
# - There is not enough data to make compare between seasons or months
# - Saturday is the busiest day of the week. Where there are more volume of sales around lunch and tea time. It could be that people prefer to eat lunch out with family or friends while doing shopping or other outdoor activities.
# - The busiest hours are from 10:00 to  14:00 all days. 

# In[ ]:




