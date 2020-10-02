#!/usr/bin/env python
# coding: utf-8

# # Library and Data loading

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
register_matplotlib_converters()


# In[ ]:


sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv', parse_dates=['Date'])
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv', parse_dates=['Date'])


# # Introduction and Initial Look at the data 
# 
# The main contribution of this notebook are the graphing tools. I introduce compair_curves which attempts to line up the curves from two different regions. From what I can tell Italy and Spain are lined up pretty close, California is progressing slower than Italy, but New York is showing a faster rate of infection that Italy. I know that there are many factors that go into these 'curves' and will be updating this kernal with any comments/sugestions provided. 
# 
# I also included my EDA which did not get any votes from my last kernal, but shows very clearly the effectivness of China's lockdown.
# 
# - from previous kernal
# 
# The more I look at the data the more it becomes abundantly clear that "locking down" is essential to slow the spread of the desease. After an early lockdown all of China's provences other than Hubei (Wuhan) kept their confirmed cases under 1,500. Wuhan did not lock down sooner as it was the epicenter of the spread, so this is the best model of what is to come. However, China locked down Wuhan with significantly less confirmed cases than most areas in Europe, Iran, and some of the US States. I am here in CA and I know that after two days of lock down there is significantly less movement, however I am pretty sure it is not near how it was in China. My wife is from Mexico and she told me they are already starting the "quarentera" which is a 40 day lock down. In addition many countries in South America are doing the same. What is happening in Europe and New York is unprecidented and will be hard to predict, the authorities in those areas let the desease spread more than it did in Wuhan before they locked down. In this notebook I will be developing a prediction model using the avaialable data and visulizing the 'curves' as I go. I am an Air Quality/Weather Engineer by trade, so at some point I will try to incorperate weather data into my analysis. 
# 
# I looked at the data using pandas and pyplot. From this initial look I can tell that there is training data for 284 Country/Area combinations from 1/22/20 through 3/18/20 and the compition will be judged using data from these same Country/Area combinations from 3/12 through 4/23. I used "f-strings" to output the various numbers/dates I was looking for in the dataset to make everything look better. 

# In[ ]:


train['Province/State'] = train['Province/State'].fillna('No Sub') # I will fill the NAs for the areas with no sub-region
train['area_full'] = train['Country/Region'] + '_' + train['Province/State'] # make a unique identifier for each sub-region


# In[ ]:


most_cases = train.groupby('area_full').max().sort_values('ConfirmedCases', ascending=False).iloc[:30]
_, ax = plt.subplots(figsize=(10,8))
sns.barplot(most_cases['ConfirmedCases'], most_cases.index, ax=ax)
ax.set_title('Countries with the Most Cases')


# In[ ]:


train.shape


# In[ ]:


print(f"There are { len(train['area_full'].unique())} unique areas within {len(train['Country/Region'].unique())} Country/Regions. The training data set starts on "      f"{train.Date.min().strftime('%m-%d-%Y')} and goes through {train.Date.max().strftime('%m-%d-%Y')}. The test set starts on {test.Date.min().strftime('%m-%d-%Y')}"      f" and goes through {test.Date.max().strftime('%m-%d-%Y')}.")


# In[ ]:


train[train.ConfirmedCases == train.ConfirmedCases.max()]['area_full'].iloc[0] # find Wuhan's sub-area


# The graph cases function will be how I visulize the Cases and Fatalities and compare various areas. I will updated it as I go and possibly make other versions. 

# # Compare Curves Testing

# In[ ]:


def compare_curves(loc1, loc2, df=train):
    """Compare one curve to another with offset.
       loc1: location with no offset
       loc2: location with offset
       df: train dataframe
       offset: # of days"""
    myFmt = mdates.DateFormatter('%m/%d/%Y')
    days = mdates.DayLocator(interval=4)

    df1 = train[train['area_full'] == loc1]
    df2 = train[train['area_full'] == loc2]
    fig, ax = plt.subplots(figsize=(16,8))
    
    cases_min = 25
    if df1.ConfirmedCases.min() > cases_min: cases_min = df1.ConfirmedCases.min()
    
    offset = df1.loc[df1['ConfirmedCases'] > cases_min, 'Date'].iloc[0] - df2.loc[df2['ConfirmedCases'] > cases_min, 'Date'].iloc[0]
    df2['Date'] = df2['Date'] + offset
    
    ax.plot(df1['Date'], df1['ConfirmedCases'], label=loc1.replace('_No Sub', ''))
    ax.plot(df2['Date'], df2['ConfirmedCases'], label=loc2.replace('_No Sub', ''))
    
    start = df1.loc[df1['ConfirmedCases'] > 0, 'Date'].iloc[0]
    
    ax.set_xlim(start, train['Date'].max()+pd.DateOffset(2))
    ax.set_ylim(0, df1['ConfirmedCases'].max()*1.3)
    ax.grid()
    ax.xaxis.set_major_locator(days)
    ax.xaxis.set_major_formatter(myFmt)
    
    title_string = f"Comparison between Current {loc1.replace('_No Sub', '')} and {loc2.replace('_No Sub', '')} {offset.days} Days Ago"
    
    ax.set_title(title_string, size=20)
    ax.legend()
    fig.autofmt_xdate()

compare_curves('Spain_No Sub', 'Italy_No Sub', train)


# In[ ]:


compare_curves('Italy_No Sub', 'China_Hubei')


# In[ ]:


compare_curves('US_New York', 'China_Hubei')


# In[ ]:


compare_curves('US_Florida', 'US_New York')


# In[ ]:


compare_curves('US_California', 'US_Washington')


# In[ ]:


compare_curves('US_New York', 'Italy_No Sub')


# In[ ]:


def graph_cases(ax=False, fig=None, location=None, df=None, cases=True, deaths=True):
    df = df[df['area_full'] == location]
    location = location.replace('_No Sub', '') # get rid of No Sub if there is no subregion
    myFmt = mdates.DateFormatter('%m/%d/%Y')
    days = mdates.DayLocator(interval=4)

    if ax == False: 
        fig, ax = plt.subplots(figsize=(16,8))
    if cases: ax.plot(df.Date, df.ConfirmedCases, label=f'{location} Cases')
    if deaths: ax.plot(df.Date, df.Fatalities, label=f'{location} Fatalities')
    ax.xaxis.set_major_locator(days)
    ax.xaxis.set_major_formatter(myFmt)
    ax.grid()
    ax.legend()
    ax.set_title(f'COVID-19 Cases', size=26)
    fig.autofmt_xdate()
    return ax, fig


# ## Wuhan vs. Italy
# Since these two areas have been ahead of other regions, I will take a look at them first. I searched news articles to find when the lockdowns happened which are linked here:
# - https://www.theguardian.com/world/2020/mar/19/chinas-coronavirus-lockdown-strategy-brutal-but-effective
# - https://www.nbcnews.com/news/world/coronavirus-has-italy-lockdown-what-rest-us-have-look-forward-n1155396

# In[ ]:


ax, fig = graph_cases(location='China_Hubei', df=train)
ax, fig = graph_cases(ax=ax, fig=fig, location='Italy_No Sub', df=train)
ax.plot([pd.datetime(2020,1,23), pd.datetime(2020,1,23)], [0,70000], 'g--', label='Hubei Lockdown Starts') ## turn info a function
ax.plot([pd.datetime(2020,3,10), pd.datetime(2020,3,10)], [0,70000], 'r--', label='Italy Locks down Country')
plt.grid()
ax.legend(loc='upper left')
plt.show()


# In[ ]:





# In[ ]:


ax, fig = graph_cases(location='Italy_No Sub', df=train)
# ax, fig = graph_cases(ax=ax, fig=fig, location='US_Washington', df=train)
ax, fig = graph_cases(ax=ax, fig=fig, location='France_France', df=train)
ax, fig = graph_cases(ax=ax, fig=fig, location='Iran_No Sub', df=train)
# ax.grid()
# ax.grid()
ax.set_xlim(pd.datetime(2020,2,15))


# In[ ]:


print(f"The number of cases in Hubei (Wuhan) when lockdown procedures were initiated: "       f"{ train[ (train['area_full'] == 'China_Hubei') & (train['Date'] == pd.datetime(2020,1,25))]['ConfirmedCases'].iloc[0]}")


# ## China Regional Analysis
# Now I will take a look at all of the other regions in China. It is pretty clear that China's lockdown worked. I will see what kind of conection I can make using the number of cases once lockdown starts to the total number of current cases, which for China is when the daily new cases stopped for the most part. Clearly there are many factors which are hidden in the ConfirmedCases variable which include "tests avaiable", "showing symptoms", and "percent age group in region" to name a few. From what I can tell on the news China locked down the contry pretty uniformly on 1/23/2020. I used 1/25 as the starting point to calculate the growth factor for the various regions in China as the numbers on the 23rd were pretty small and likely not very representative of the overall infection rate. The "lockdown" initial count can be changed using the START_DATE constant. 
# 
# I made histograms to veiw the distribution of the confirmed cases during the start and end of the analysis along with the calculated 'Growth Rates'. It is pretty clear that Wuhan is an outlier in all three of these categories. The number of cases was around 90 times when the outbreak subsided in Wuhan than the modeled 'start date' when the lockdowns began. The rest of the regions went up between 10 - 40 times the initial value before the numbers started to flatten out. There are many possible reasons for this. The amount of people infected / the amount of people tested is the first one that comes to mind. 

# In[ ]:


START_DATE = pd.datetime(2020,1,25)

c_regions = china_regions = train[train['Country/Region'] == 'China']['area_full'].unique()

ax, fig = graph_cases(location='China_Anhui', df=train, deaths=False)
for reg in c_regions:
    if reg not in ['China_Anhui', 'China_Hubei']:
        ax, fig = graph_cases(ax=ax, fig=fig, location=reg, df=train, deaths=False)
        
ax.plot([START_DATE, START_DATE], [0,1400], 'g--', label='Start Date') ## turn info a function
ax.legend(bbox_to_anchor=(1.25,1)) # move the legend so you can see cases
ax.grid()


# In[ ]:


china = train[train['Country/Region'] == 'China']
num_shutdown = china[china['Date'] == START_DATE] # This number can be changed 
num_latest = china[china['Date'] == china['Date'].max()] # find the latest variables
sd = num_shutdown[['Province/State', 'ConfirmedCases']].merge(num_latest[['Province/State', 'ConfirmedCases']], on='Province/State', suffixes=('_sd', '_ct'))
sd['PercentGrowth'] = sd['ConfirmedCases_ct'] / sd['ConfirmedCases_sd']
sd = sd[sd['ConfirmedCases_sd'] > 0] # get rid of the infinity cases


# In[ ]:


_, axs = plt.subplots(ncols=3, figsize=(24,8))
axs[0].hist(sd['ConfirmedCases_sd'], bins=30)
axs[0].plot([sd['ConfirmedCases_sd'].mean(), sd['ConfirmedCases_sd'].mean()], [0,30], 'r--', 3, label='Mean')
axs[0].set_title(f"COVID-19 Confirmed Cases on {START_DATE.strftime('%Y-%m-%d')}", size=18)
axs[0].legend() # Mean is mentioned twice on the legend
axs[0].grid()
axs[1].hist(sd['ConfirmedCases_ct'], bins=30)
axs[1].plot([sd['ConfirmedCases_ct'].mean(), sd['ConfirmedCases_ct'].mean()], [0,30], 'r--', 3, label='Mean')
axs[1].set_title(f"COVID-19 Confirmed Cases on {china.Date.max().strftime('%Y-%m-%d')}", size=18)
axs[1].legend() # Mean is mentioned twice on the legend
axs[1].grid()
axs[2].hist(sd['PercentGrowth'], bins=10)
axs[2].plot([sd['PercentGrowth'].mean(), sd['PercentGrowth'].mean()], [0,11], 'r--', 3, label='Mean')
axs[2].set_title('COVID-19 Groth Rate Distribution for Chinese Provenses', size=18)
axs[2].legend() # Mean is mentioned twice on the legend
axs[2].grid()
plt.show()


# In[ ]:


sd[sd['PercentGrowth'] > 40]


# In[ ]:


sd[sd['ConfirmedCases_sd'] > 50]


# ## US Regional Analysis
# 
# Next I will take a look at the U.S. and use what I learned from China to project what the numbers might look like over the coming weeks. I like in CA and was just locked down (3/18/2020) and I know that New York was locked down as well around the same time. I afm assuming that as the number of cases grow, the rest of the states will likely do the same. However, the lock down(s) in the US will likely not be as strict as it is in China. I made another function here to compare curves using a date offset. This will likely be usefull for looking at the countries for the rest of word as well.

# In[ ]:


us = train[train['Country/Region'] == 'US']
max_cases = us.groupby('area_full').max()
over_50 = max_cases[max_cases['ConfirmedCases'] > 100].index
# us.groupby('Date').sum()


# In[ ]:


compare_curves('US_New York', 'Iran_No Sub', )


# In[ ]:


train.area_full.unique()


# In[ ]:




