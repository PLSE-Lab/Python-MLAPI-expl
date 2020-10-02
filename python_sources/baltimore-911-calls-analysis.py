#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for charts
import matplotlib.dates as md
import matplotlib.patches as mpatches
import statsmodels.api as sm
from pylab import rcParams
import IPython
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/911_calls_for_service.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
df.head()


# <h1><center>Data Cleaning</center></h1>

# In[ ]:


# no missing data for 'description'
df.isnull().sum()


# In[ ]:


# convert all description to upper case
df['description']=df['description'].str.upper()


# In[ ]:


# check ramdom record at some index
df.loc[[140242]]


# In[ ]:


#strip out all wired characters
df.description = df.description.str.strip()


# In[ ]:


df.description[df.description==''].count()


# In[ ]:


df.loc[df.description=='','description'] = 'NO DESC'


# In[ ]:


# top 1000 records group by description
grouped = df.groupby('description').count().reset_index()
grouped.sort_values('priority', ascending=False,inplace=True)
grouped[grouped.priority>1000].count()


# In[ ]:


# top description in data
grouped[grouped.priority>1000].head(n=10)


# <h1><center>Description Cleaning</center></h1>

# In[ ]:


#clean top description spelling mostly
df.description.replace({ #FINDING AND REPLACING THE DESCRIPTION
    'INVESTIFATE': 'INVESTIGATE',
    'BAIL-OUT': 'BAILOUT',
    'BALOUT': 'BAILOUT',
    ' BAIL OUT':'BAILOUT',
    'BAILED OUT': 'BAILOUT',
    ' ASSAULT': 'ASSAULT',
    ' ASSAULTED': 'ASSAULT',
    'Y': 'YELLING FOR HELP',
    'YELLING HELP': 'YELLING FOR HELP',
    'YP':'YELLING FOR HELP',
    'ECORT':'ESCORT',
    'WROMG NUMBER': 'WRONG NUMBER',
    'WRONG UNMBER': 'WRONG NUMBER',
    'Wanted on Warr': 'Warrant Service',
    'YELLING FOR HELP':'YELL FOR HELP',
    'YELL 4 HELP': 'YELL FOR HELP',
    'WRECKLESS DRVR': 'WRECKLESSDRIVER',''
    ' BAIL-0UT': 'BAILOUT',
    ' BAILOUT': 'BAILOUT',
    ' BAIL-OUT': 'BAILOUT',
    'WRECKLESS DRVING': 'WRECKLESSDRIVER',
    'WRECKLESS DRV': 'WRECKLESSDRIVER',
    'WRECKLESS DRIVR': 'WRECKLESSDRIVER',
    'WRECKLESS DRIVIN': 'WRECKLESSDRIVER',
    'WRECKLESS DRIVER':  'WRECKLESSDRIVER',
    'WRECKLESS DRIVE': 'WRECKLESSDRIVER',
    'WRECFKLESS DRIVE':'WRECKLESSDRIVER',
    'WRCKLSS DRIVER': 'WRECKLESSDRIVER',
    'WRCKLESS DRIVER': 'WRECKLESSDRIVER',
    ' WRECKLESS DRIVE':'WRECKLESSDRIVER',
    'WRECKLESS': 'WRECKLESSDRIVER',
    'WRECKLES DRIVING': 'WRECKLESSDRIVER',
    'WRECKLES DRIVERS': 'WRECKLESSDRIVER',
    'WRECKLES DRIVER': 'WRECKLESSDRIVER',
    '`INVEST': 'INVESTIGATE',
    '`INVESTIGATE': 'INVESTIGATE',
    'iNVESTIGATE': 'INVESTIGATE',
    '*#TRAFIC ARREST': '*#TRAFFIC ARREST',
    '&75': '& 75',
    '911/No Voice': '911/NO  VOICE',
    '911HANGUP': '911/HANGUP',
    '911 HANGUP': '911/HANGUP',
    'FAMILY DISTURBAN': 'FAMILY DISTURB',
    'FAMILY DISTURBP': 'FAMILY DISTURB',
    'AUTO ACC/INJURY': 'AUTO ACCIDENT',
    '*INVESTIGATE' : 'INVESTIGATE',
    '*TRAFFIC CONTROL'    : 'TRAFFIC CONTROL',
    'CHECK WELLBEING': 'CHECK WELL BEING'}, inplace = True)


# In[ ]:


# check some replacement
df.loc[df.description=='CHECK WELL BEING'].head()


# In[ ]:


# convert to time series
df['callDateTime']=  pd.to_datetime(df['callDateTime'])
df.index = df['callDateTime']
del df['callDateTime']


# <h1><center>Priority Cleaning</center></h1>

# In[ ]:


# let's check null records...
df.priority.isnull().value_counts()


# In[ ]:


#let's group by with desceription and priority
grouped = df.groupby(['description','priority']).count()[['callNumber']]


# In[ ]:


#check idxmax function to get max value description with priority.
grouped.groupby(level=0).idxmax()['callNumber'].values


# In[ ]:


# create a key
key=grouped.loc[grouped.groupby(level=0).idxmax()['callNumber'].values]


# In[ ]:


#how key looks like
key.head()


# In[ ]:


# check one description has multiple priority
grouped.loc[['BURGLARY']]


# In[ ]:


#check same description
key.loc[['BURGLARY']]


# In[ ]:


# check key index is true
key.index.is_unique


# In[ ]:


#key needs to be a Series
key = key.reset_index().drop('callNumber',axis=1).set_index('description')['priority'] 


# In[ ]:


key.head()


# In[ ]:


key.loc['BURGLARY']


# In[ ]:


key.index.is_unique


# In[ ]:


#create mask for priority is null
mask = df.priority.isnull()


# In[ ]:


key.loc[df.loc[mask,'description']].count()


# In[ ]:


key.loc[df.loc[mask,'description']].values


# In[ ]:


# update priority missing data with key data
df.loc[mask,'priority'] = key.loc[df.loc[mask,'description']].values


# In[ ]:


# check for still null
df.priority.isnull().value_counts()


# In[ ]:


# these records are not updated yet
df[df.priority.isnull()]


# unique descriptions with no priority data.
# use similar descriptions to fill missing priorities.

# In[ ]:


df[df.description.str.contains('CHILD ALONE')]


# In[ ]:


df.loc[(df.description == 'CHILDREN ALONE') & (df.priority.isnull()),'priority'] = 'Medium'


# In[ ]:


df[df.priority.isnull()]


# In[ ]:


df[(df.description.str.contains('CHECK')) & (df.description.str.contains('ADVISE'))].groupby('priority').count()


# In[ ]:


df.loc[(df.description == 'CK & ADVISE') & (df.priority.isnull()),'priority'] = 'Low'


# In[ ]:


df[df.priority.isnull()]


# In[ ]:


df.loc[(df.description.str.contains('COURT')) & (df.description.str.contains('VIO'))].groupby('priority').count()


# In[ ]:


df.loc[(df.description == 'COURT ORDER VIOL') & (df.priority.isnull()),'priority'] = 'Low'


# In[ ]:


df[df.priority.isnull()]


# In[ ]:


# no data available
df[df.description.str.contains('MEAR')]


# In[ ]:


#make it low
df.loc[(df.description == 'MMEARING') & (df.priority.isnull()),'priority'] = 'Low'


# In[ ]:


df[df.priority.isnull()]


# In[ ]:


df.loc[(df.description.str.contains('BAIL'))].groupby('priority').count()


# In[ ]:


# fix the last one
df.loc[(df.description == 'W BAILOUT') & (df.priority.isnull()),'priority'] = 'Low'


# In[ ]:


#make sure no null
df.priority.isnull().value_counts()


# In[ ]:


df.head()


# <h1><center>High Level Analysis</center></h1>

# In[ ]:


# function to set the label of subplots
def label(ax, string):
    ax.annotate(string, (1, 1), xytext=(-8, -4), ha='right', va='top', size=14, 
                xycoords='axes fraction', textcoords='offset points') 

# function to create autocorrelation chart
def DrawAutoCorrelation(data, title, firsttitle, secondtitle, thirdtitle, dateformat = 'hours'):
    
    fig, axes = plt.subplots(nrows=3, figsize=(15, 12))
    
    # using tigh layout so there is no space between title and sub plots
    fig.tight_layout()
    
    # set the tilte for graph
    fig.suptitle(title, fontsize=20)
    
    label(axes[0], firsttitle)
    axes[0].plot(data)

    if (dateformat == 'days'):
        xfmt = md.DateFormatter('%e')
    elif (dateformat == 'month'):
        xfmt = md.DateFormatter('%b, %Y')
    else:
        xfmt = md.DateFormatter('%H:%M')
        
    axes[0].xaxis.set_major_formatter(xfmt)

    # create autocorrelation chart from metplotlib
    axes[1].acorr(data, maxlags=data.size-1)
    label(axes[1], secondtitle)
    
    # create autocorrelation char from panda through autocorrelation_plot method
    label(axes[2], thirdtitle)
    pd.plotting.autocorrelation_plot(data, ax=axes[2])    
    plt.show()
    
# Function to set explode value for pie slice
def isExplode(x):
    if x == True:
        return 0.1
    else:
        return 0
    
# Function to draw pie chart and explode the slice with max size
def DrawPieChart(data, title, savefile = np.nan):
    #convert to dataframe
    data = pd.Series.to_frame(data);
    
    data['Size'] = data / data.sum() * 100
    
    # Create new attribute to set the explode slice
    data['Explode'] = data['Size'].max() == data['Size']    
    data['Explode'] = data['Explode'].apply(isExplode)
    
    fig1, ax1 = plt.subplots(figsize=(15,8))
    ax1.pie(data['Size'], explode = data['Explode'], autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.legend(data.index, loc="upper right")
    plt.title(title, fontsize=20)
    plt.tight_layout()
    if (savefile != np.nan):
        plt.savefig(savefile, bbox_inches='tight')
    plt.show()
    
    print(data)
    
def SetChartProperties(ax, xlabel, ylabel, labeltext, title=''):
    ax.grid('on', which='minor', axis='x' )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    label(ax, labeltext)


# In[ ]:


# get all calls by month and create line chart
ax = df.resample('M').callNumber.count().plot(figsize=(15,10), title="Total Monthly Calls")
SetChartProperties(ax, 'Period', 'No of calls', 'All Calls', 'All Calls by Month')
plt.savefig('1.png', bbox_inches='tight')


# In[ ]:


# average call per month
df.resample('M').priority.count().mean()


# In[ ]:


# get bar chart for all priorities
ax = df.groupby('priority').callNumber.count()[:-1].plot(kind='bar', figsize=(15,8))
SetChartProperties(ax, 'Priority', 'No of calls', 'All Priorities', 'All Calls by Priority')
plt.savefig('2.png', bbox_inches='tight')


# In[ ]:


# get pie chart for all priorities
DrawPieChart(df.groupby('priority').callNumber.count(), 'All Priorities', '3.png')


# In[ ]:


# get group by description for all calls, and get top records only
grouped = df.groupby('description').count()
grouped.sort_values('priority', ascending=False, inplace=True)
DrawPieChart(grouped[grouped.priority>50000][['priority']]             .reset_index()             .set_index('description')['priority'],'Top Descriptions', '4.png')


# In[ ]:


# Draw chart for all districts
DrawPieChart(df.groupby('district').callNumber.count(), 'All Districts', '5.png')


# In[ ]:


#let's create daily data
daily_data = df.resample('D').callNumber.count()
#convert to dataframe
daily_data = pd.DataFrame(daily_data)
# add weekdays column
daily_data['Weekdays'] = daily_data.index.weekday_name


# In[ ]:


# group by weekdays and check how many calls each day
ax =daily_data.groupby('Weekdays').sum().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])    .plot(kind='bar', figsize=(15,10))
SetChartProperties(ax, 'Weekdays', 'No of calls', '', 'All Calls By Weekday')
plt.savefig('6.png', bbox_inches='tight')


# In[ ]:


# let's check which day was highest calls on July 2015, seems like Wed, Thru, and Friday are highest
ax = daily_data['2015-07'].groupby('Weekdays').sum().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])    .plot(kind='bar', figsize=(15,10))
SetChartProperties(ax, 'Weekdays', 'No of calls', '', 'July 2015 By Weekday')
plt.savefig('7.png', bbox_inches='tight')


# <center><b>STL: A Seasonal-Trend Decomposition Procedure Based on Loess</b></center>

# In[ ]:


rcParams['figure.figsize'] = 15, 12

decomposition = sm.tsa.seasonal_decompose(df.resample('M').callNumber.count(), model='additive')
fig = decomposition.plot()
plt.savefig('8.png', bbox_inches='tight')
plt.show()


# In[ ]:


# top 10 days pf calls.
df.resample('D').callNumber.count().sort_values(ascending=False).head(10)


# <h1><center>April 2015 Analysis</center></h1>

# In[ ]:


# April 2015 seems like a larger data, let's create a chart on this month.
ax =df['2015-04'].resample('D').callNumber.count().plot(figsize=(15,10), title="April 2015 Calls")
SetChartProperties(ax, 'Period', 'No of calls', 'Total Calls')
plt.savefig('9.png', bbox_inches='tight')


# <b>Facts for April 2015</b>
# 
# http://www.cnn.com/2015/04/27/us/baltimore-riots-timeline/index.html
# 
# https://en.wikipedia.org/wiki/Death_of_Freddie_Gray
# 
# http://www.cnn.com/2015/04/27/us/baltimore-unrest/index.html
# 
# https://www.nytimes.com/2015/04/26/us/baltimore-crowd-swells-in-protest-of-freddie-grays-death.html

# In[ ]:


# April 27-28 calls
ax =df['2015-04-27': '2015-04-28'].resample('H').callNumber.count().plot(figsize=(15,10),                                                     title="April 27-28, 2015 Calls")
SetChartProperties(ax, 'Period', 'No of calls', '48 Hours Calls', 'April 2015, 27/28 Calls')
plt.savefig('10.png', bbox_inches='tight')


# In[ ]:


# April 2015 day of weeks chart
ax = daily_data['2015-04'].groupby('Weekdays').sum().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])    .plot(kind='bar', figsize=(15,10))
SetChartProperties(ax, 'Weekdays', 'No of calls', '', 'April 2015 By Weekday')
plt.savefig('11.png', bbox_inches='tight')


# <h1><center>Correlation Two Years</center></h1>

# In[ ]:


calls15 = df['2015'].resample('D').count()
calls15['dayofyear'] = calls15.index.dayofyear
calls15.set_index('dayofyear',inplace=True)
calls16 = df['2016'].resample('D').count()
calls16['dayofyear'] = calls16.index.dayofyear
calls16.set_index('dayofyear',inplace=True)


# In[ ]:


ax = calls15.priority.plot(x='dayofyear',figsize=(15,10), title="2015 vs 2016 Calls",label='2015',alpha=0.5)
c15_rMean = calls15.priority.rolling(window=30,center=True).mean() # Rolling Mean
ax.plot(c15_rMean, color='red',label='2015 Mean')

ax = calls16.priority.plot(figsize=(15,10), title="2015 vs 2016 Calls",label='2016',alpha=0.5)
c16_rMean = calls16.priority.rolling(window=30,center=True).mean() # Rolling Mean
ax.plot(c16_rMean, color='darkgreen',label='2016 Mean')
tick = round(366/4)
ax.xaxis.set_ticks([tick,2*tick,3*tick])
ax.grid('on',  axis='x', which = 'major',color='black')
ax.grid('on',  axis='x', which = 'minor',linestyle='--')
ax.minorticks_on()
ax.legend()
plt.savefig('12.png', bbox_inches='tight')


# In[ ]:


plt.figure(figsize=(15,10))
plt.scatter(calls15.priority,calls16.priority[:-1],c=['red','blue'])
plt.title('2015 vs 2016 Calls - Scatter Plot')

red_patch = mpatches.Patch(color='red', label='2015 Calls')
blue_patch = mpatches.Patch(color='blue', label='2016 Calls')

plt.legend(handles=[red_patch,blue_patch])
plt.savefig('13.png', bbox_inches='tight')


# <h1><center>Story 1: Assault Analysis</center></h1>

# In[ ]:


# load assaults calls
assault_calls = df.loc[df.description.str.contains('ASSAULT')==True]
assault_calls.head()


# In[ ]:


# total assaults calls
assault_calls.callNumber.count()


# In[ ]:


#percentage assaults calls.
assault_calls.callNumber.count() / df.callNumber.count() * 100


# In[ ]:


# draw chart for prority for assaults.
DrawPieChart(assault_calls.groupby('priority').callNumber.count(), 'Priority for Assaults', '14.png')


# In[ ]:


# draw chart for  district in assaults.
DrawPieChart(assault_calls.groupby('district').callNumber.count(), 'Top Districts for Assaults',             '14.png')


# In[ ]:


# monthly chart for assaults
ax = assault_calls.resample('M').callNumber.count().plot(figsize=(15,10), title="Total Monthly Calls for Assaults")
SetChartProperties(ax, 'Period', 'No of calls', 'Assaults Calls', 'Assault Calls by Month')
plt.savefig('15.png', bbox_inches='tight')


# <center><b>May 2015 is highest</b></center>

# <b>Related news May 2015</b><br>
# http://www.baltimoresun.com/news/maryland/crime/bal-may-2015-baltimores-deadliest-month-in-15-years-sg-storygallery.html<br>
# 
# http://www.cnn.com/2015/05/26/us/baltimore-deadliest-month-violence-since-1999/index.html<br>
# 
# http://www.cnn.com/2015/05/02/us/freddie-gray-baltimore-death/index.html
# 
# According to wikipedia
# 
# Increase in violence and decrease in policing[edit]
# Baltimore recorded 43 homicides in the month of May, the second deadliest month on record and the worst since December 1971 when 44 homicides were recorded. There have also been more than 100 non-fatal shootings in May 2015.[151] Police commissioner Anthony Batts blames looted drugs, stolen from 27 pharmacies and two methadone clinics, as well as street distribution and turf wars for the spike in crime.
# 
# https://en.wikipedia.org/wiki/2015_Baltimore_protests

# In[ ]:


# monthly bar chart for assaults
monthly_assault = pd.Series.to_frame(assault_calls.resample('M').callNumber.count())
fig, ax = plt.subplots(figsize=(15, 8))
ax.xaxis.set_major_formatter(md.DateFormatter('%b, %y'))
ax.bar(monthly_assault.index, monthly_assault['callNumber'], width=25, align='center')
SetChartProperties(ax, 'Period', 'No of calls', 'Total Calls', 'Assault Calls by Month')
plt.savefig('16.png', bbox_inches='tight')


# In[ ]:


# May 2015 calls chart for assaults
ax = assault_calls['2015-05'].resample('D').callNumber.count().plot(figsize=(15,10),     title="Total Calls in May 2015 for Assaults")
SetChartProperties(ax, 'Days', 'No of calls', 'May 2015 Calls', 'Assault Calls on May 2015')
plt.savefig('17.png', bbox_inches='tight')


# <center><b>May 10 is highest</b></center>

# In[ ]:


# May 10, 2015 calls chart for assaults
ax = assault_calls['2015-05-10'].resample('H').callNumber.count().plot(figsize=(15,10), title="Hourly Calls in May 10 2015 for Assaults")
SetChartProperties(ax, 'Days', 'No of calls', 'May 10, 2015 Calls', 'Assault Calls on May 10, 2015')
plt.savefig('18.png', bbox_inches='tight')


# <center><b>Between 5-6 pm was peak time</b></center>

# In[ ]:


# create chart for two year compariason
assaults2015 = assault_calls['2015'].resample('M').callNumber.count().reset_index()
assaults2015['month'] = assaults2015['callDateTime']
assaults2015['month'] = assaults2015['month'].apply(lambda x: x.month)
assaults2015.drop('callDateTime', axis=1, inplace=True)
assaults2015.columns = ['2015', 'month']

assaults2016 = assault_calls['2016'].resample('M').callNumber.count().reset_index()
assaults2016['month'] = assaults2016['callDateTime']
assaults2016['month'] = assaults2016['month'].apply(lambda x: x.month)
assaults2016.drop('callDateTime', axis=1, inplace=True)
assaults2016.columns = ['2016', 'month']

mergeassaults = pd.merge(assaults2015, assaults2016)
mergeassaults.set_index('month', inplace=True)


# In[ ]:


# draw a chart
ax = mergeassaults.plot(figsize=(15,10), title="Assault Calls Yearly Comparison")
SetChartProperties(ax, 'Months', 'No of calls', '', 'Assault Calls 2015-2016 Comparison')
plt.savefig('19.png', bbox_inches='tight')


# <center><b>STL: A Seasonal-Trend Decomposition Procedure Based on Loess</b></center>

# In[ ]:


rcParams['figure.figsize'] = 15, 12

decomposition = sm.tsa.seasonal_decompose(assault_calls.resample('M').callNumber.count(), model='additive')
fig = decomposition.plot()
plt.show()


# In[ ]:


DrawAutoCorrelation(assault_calls.resample('M').callNumber.count().astype('float64'),                     'Total Assault Calls', 'All Months',                     'Autocorrelation assaults calls by Matplotlib',                     'Autocorrelation assaults calls by Pandas', dateformat='month')


# <center>The plot above clearly shows the upwards trend of total assaults calls, along with its yearly seasonality.</center>

# <h1><center>Story 2: Narcotics Analysis</center></h1>

# In[ ]:


#load narcotics calls
narcotics_calls = df.loc[df.description.str.contains('NARCOTICS')==True]
narcotics_calls.head()


# In[ ]:


# total narcotics calls
narcotics_calls.callNumber.count()


# In[ ]:


#percentage narcotics calls.
narcotics_calls.callNumber.count() / df.callNumber.count() * 100


# In[ ]:


# draw priority for Narcotics
DrawPieChart(narcotics_calls.groupby('priority').callNumber.count(), 'Priority for Narcotics', 'narcotics=priority.png')


# In[ ]:


# draw distict for Narcotics
DrawPieChart(narcotics_calls.groupby('district').callNumber.count(),              'Top Districts for Narcotics', '20.png')


# In[ ]:


# monthly chart line chart for narcotics
ax = narcotics_calls.resample('M').callNumber.count().plot(figsize=(15,10), title="Total Monthly Calls for Narcotics")
SetChartProperties(ax, 'Period', 'No of calls', 'Narcotics Calls', 'Narcotics Calls by Month')
plt.savefig('21.png', bbox_inches='tight')


# <center><b>May 2017 is highest</b></center>

# <b>Related news around peak time May to July 2017</b><br>
# http://baltimore.cbslocal.com/2017/06/02/9-million-heroin-seized/<br>
# 
# http://baltimore.cbslocal.com/2017/05/25/md-state-police-makes-drug-bust-spanning-5-counties-including-delaware-130000-worth-of-drugs/<br>
# 
# http://www.baltimoresun.com/news/maryland/crime/bs-md-doctors-indicted-20170810-story.html

# In[ ]:


# monthly bar chart for narcotics
monthly_narcotics = pd.Series.to_frame(narcotics_calls.resample('M').callNumber.count())
fig, ax = plt.subplots(figsize=(15, 8))
ax.xaxis.set_major_formatter(md.DateFormatter('%b, %y'))
ax.bar(monthly_narcotics.index, monthly_narcotics['callNumber'], width=25, align='center')
SetChartProperties(ax, 'Period', 'No of calls', 'Total Calls', 'Narcotics Calls by Month')
plt.savefig('22.png', bbox_inches='tight')


# In[ ]:


# May 2015 calls chart for assaults
ax = narcotics_calls['2017-05'].resample('D').callNumber.count().plot(figsize=(15,10), title="Total Calls in May 2017 for Narcotics")
SetChartProperties(ax, 'Days', 'No of calls', 'May 2017 Calls', 'Narcotics Calls on May 2017')
plt.savefig('23.png', bbox_inches='tight')


# <center><b>May 02 is highest</b></center>

# In[ ]:


# May 10, 2015 calls chart for assaults
ax = narcotics_calls['2017-05-02'].resample('H').callNumber.count().plot(figsize=(15,10), title="Hourly Calls in May 02 2017 for Narcotics")
SetChartProperties(ax, 'Days', 'No of calls', 'May 02, 2017 Calls', 'Narcotics Calls on May 02, 2017')
plt.savefig('24.png', bbox_inches='tight')


# In[ ]:


# create chart for two year compariason
narcotics2015 = narcotics_calls['2015'].resample('M').callNumber.count().reset_index()
narcotics2015['month'] = narcotics2015['callDateTime']
narcotics2015['month'] = narcotics2015['month'].apply(lambda x: x.month)
narcotics2015.drop('callDateTime', axis=1, inplace=True)
narcotics2015.columns = ['2015', 'month']

narcotics2016 = narcotics_calls['2016'].resample('M').callNumber.count().reset_index()
narcotics2016['month'] = narcotics2016['callDateTime']
narcotics2016['month'] = narcotics2016['month'].apply(lambda x: x.month)
narcotics2016.drop('callDateTime', axis=1, inplace=True)
narcotics2016.columns = ['2016', 'month']

mergenarcotics = pd.merge(narcotics2015, narcotics2016)
mergenarcotics.set_index('month', inplace=True)


# In[ ]:


# draw a chart
ax = mergenarcotics.plot(figsize=(15,10), title="Narcotics Calls Comparison")
SetChartProperties(ax, 'Months', 'No of calls', '', 'Narcotics Calls 2015-2016 Comparison')
plt.savefig('25.png', bbox_inches='tight')


# <center><b>STL: A Seasonal-Trend Decomposition Procedure Based on Loess</b></center>

# In[ ]:


rcParams['figure.figsize'] = 15, 12

decomposition = sm.tsa.seasonal_decompose(narcotics_calls.resample('M').callNumber.count(), model='additive')
fig = decomposition.plot()
plt.show()


# In[ ]:


DrawAutoCorrelation(narcotics_calls.resample('M').callNumber.count().astype('float64'),                     'Total Narcotics Calls', 'All Months',                     'Autocorrelation narcotics calls by Matplotlib',                     'Autocorrelation narcotics calls by Pandas', dateformat='month')


# <center>The plot above clearly shows the upwards trend of total narcotics calls, along with its yearly seasonality.</center>

# <h1><center>Story 3: Accidents Analysis</center></h1>

# In[ ]:


accident_calls = df.loc[df.description.str.contains('ACCIDENT')==True]


# In[ ]:


accident_calls.head()


# In[ ]:


DrawPieChart(accident_calls.groupby('priority').callNumber.count(), 'Priority for Accidents', '26.png')


# In[ ]:


DrawPieChart(accident_calls.groupby('district').callNumber.count(), 'Top Districts for Accidents', '27.png')


# In[ ]:


# monthly chart line chart for accidents
ax = accident_calls.resample('M').callNumber.count().plot(figsize=(15,10), title="Total Monthly Calls for Accidents")
SetChartProperties(ax, 'Period', 'No of calls', 'Accidents Calls', 'Accidents Calls by Month')
plt.savefig('28.png', bbox_inches='tight')


# <center><b>May 2016 is highest</b></center>

# <b>Related news May 2016 bus accident</b><br>
# http://baltimore.cbslocal.com/2016/05/26/several-students-injured-in-baltimore-school-bus-crash/
# 
# <b>Road accidents death increase from 2015 to 2016</b><br>
# http://www.mdot.maryland.gov/News/Releases2017/2016_April_26_MDOT_Releases_2016_Roadway_Deaths

# In[ ]:


# monthly bar chart for accidents
monthly_accidents = pd.Series.to_frame(accident_calls.resample('M').callNumber.count())
fig, ax = plt.subplots(figsize=(15, 8))
ax.xaxis.set_major_formatter(md.DateFormatter('%b, %y'))
ax.bar(monthly_accidents.index, monthly_accidents['callNumber'], width=25, align='center')
SetChartProperties(ax, 'Period', 'No of calls', 'Total Calls', 'Accidents Calls by Month')
plt.savefig('29.png', bbox_inches='tight')


# In[ ]:


# create chart for two year compariason
accidents2015 = accident_calls['2015'].resample('M').callNumber.count().reset_index()
accidents2015['month'] = accidents2015['callDateTime']
accidents2015['month'] = accidents2015['month'].apply(lambda x: x.month)
accidents2015.drop('callDateTime', axis=1, inplace=True)
accidents2015.columns = ['2015', 'month']

accidents2016 = accident_calls['2016'].resample('M').callNumber.count().reset_index()
accidents2016['month'] = accidents2016['callDateTime']
accidents2016['month'] = accidents2016['month'].apply(lambda x: x.month)
accidents2016.drop('callDateTime', axis=1, inplace=True)
accidents2016.columns = ['2016', 'month']

mergeaccidents = pd.merge(accidents2015, accidents2016)
mergeaccidents.set_index('month', inplace=True)


# In[ ]:


# draw a chart
ax = mergeaccidents.plot(figsize=(15,10), title="Accidents Calls Comparison")
SetChartProperties(ax, 'Months', 'No of calls', '', 'Accidents Calls 2015-2016 Comparison')
plt.savefig('30.png', bbox_inches='tight')


# <center><b>STL: A Seasonal-Trend Decomposition Procedure Based on Loess</b></center>

# In[ ]:


rcParams['figure.figsize'] = 15, 12

decomposition = sm.tsa.seasonal_decompose(accident_calls.resample('M').callNumber.count(), model='additive')
fig = decomposition.plot()
plt.show()


# <center>The plot above clearly shows the upwards trend of total accidents calls, along with its yearly seasonality.</center>

# In[ ]:


DrawAutoCorrelation(accident_calls.resample('M').callNumber.count().astype('float64'),                     'Total Accidents Calls', 'All Months',                     'Autocorrelation accidents calls by Matplotlib',                     'Autocorrelation accidents calls by Pandas', dateformat='month')


# In[ ]:




