#!/usr/bin/env python
# coding: utf-8
Are Formula 1 cars getting faster? Let's try and answer that question with a step by step python analysis. I want to provide beginners with an example of how you can use a python notebook to analyse data, from start to finish. As it is often the case, a lot of the code below is about cleaning the data. You can jump to section 7 if you are not interested in that.
# ### 1. Classic libraries

# In[ ]:


import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
import re


# ### 2. Loading the data

# #### 2.1. Accessing the files on Kaggle

# In[ ]:


files = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        files.append(str(os.path.join(dirname, filename)))
files


# In[ ]:


names = [re.findall('\w*.csv', x)[0].split('.')[0] for x in files]
names


# #### 2.2. Reading the CSVs in a dictionary of dataframes

# In[ ]:


data_dict = {}

for i, file in enumerate(files):
    name = names[i]
    data_dict[name] = pd.read_csv(file, encoding = 'latin-1') # https://stackoverflow.com/questions/5552555/unicodedecodeerror-invalid-continuation-byte


# ### 3. Getting familiar with the data

# In[ ]:


data_dict['circuits'].sample(3)


# In[ ]:


data_dict['constructorResults'].sample(3)


# In[ ]:


data_dict['constructors'].sample(3)


# In[ ]:


data_dict['constructorStandings'].sample(3)


# In[ ]:


data_dict['drivers'].sample(3)


# In[ ]:


data_dict['driverStandings'].sample(3)


# In[ ]:


data_dict['lapTimes'].sample(3)


# In[ ]:


data_dict['pitStops'].sample(3)


# In[ ]:


data_dict['qualifying'].sample(3)


# In[ ]:


data_dict['races'].sample(3)


# In[ ]:


data_dict['races'][data_dict['races']['year'] == 2017]


# In[ ]:


data_dict['results'].sample(3)


# In[ ]:


data_dict['seasons'].sample(3)


# In[ ]:


data_dict['status'].sample(3)


# In[ ]:


data_dict['results'][data_dict['results']['raceId'] == 988].head()


# In[ ]:


data_dict['status'][data_dict['status']['statusId'].isin([1, 11, 36, 9])]


# In[ ]:


data_dict['results']['positionText'].value_counts()


# In[ ]:


data_dict['results'][data_dict['results']['positionText'].isin(['R', 'F', 'W', 'N', 'E'])].isna().sum()


# ### 4. Cleaning and prepping

# #### 4.1. Building a single data frame for our use case

# Remove and rename columns

# In[ ]:


data_dict['results'].drop(columns = ['rank', 'positionOrder'], inplace = True)

data_dict['constructors'].drop(columns = ['nationality', 'url', 'Unnamed: 5'], inplace = True)
data_dict['constructors'].rename(columns = {'name': 'constructorName'}, inplace = True)

data_dict['races'].drop(columns = ['time', 'url'], inplace = True)
data_dict['races'].rename(columns = {'name': 'raceName', 'date': 'raceDate'}, inplace = True)

data_dict['circuits'].drop(columns = ['lat', 'lng', 'alt', 'url'], inplace = True)
data_dict['circuits'].rename(columns = {'name': 'circuitName'}, inplace = True)

data_dict['drivers'].drop(columns = ['number', 'code', 'url'], inplace = True)


# Join the results data set with the other 5 above + the status dataframe

# In[ ]:


df = data_dict['results'].copy()
df = df.merge(data_dict['constructors'], how = 'left')
df = df.merge(data_dict['races'], how = 'left')
df = df.merge(data_dict['circuits'], how = 'left')
df = df.merge(data_dict['drivers'], how = 'left')
df = df.merge(data_dict['status'], how = 'left')


# Same length pre and post join, good

# In[ ]:


len(df), len(data_dict['results'])


# What time frame are we covering BTW?

# In[ ]:


df['year'].min(), df['year'].max()


# #### 4. 2. Adding new columns and fixing issues

# Adding new columns and converting some data types - starting with a lap distance column (speed = distance / time ==> distance = speed * time)

# In[ ]:


df['fastestLapTimeSec'] = (df['fastestLapTime'].str.split(':', expand = True)[0].astype('float64') * 60) + df['fastestLapTime'].str.split(':', expand = True)[1].astype('float64')
df['fastestLapTimeSec'].tail(5)


# In[ ]:


df['fastestLapSpeed'].head(5)


# In[ ]:


df['fastestLapSpeed'].astype('float64')


# There is were the error is coming from

# In[ ]:


df[df['fastestLapSpeed'] == '01:42.6'][['country', 'year', 'driverRef', 'fastestLapSpeed', 'fastestLapTime']]


# In[ ]:


df[(df['year'] == 2017) & (df['country'] == 'UAE')][['driverRef', 'fastestLapTime', 'fastestLapTimeSec', 'fastestLapSpeed']]


# Let's correct the wrong speed value for Ocon using the Ricciardo observation

# In[ ]:


t = df.loc[23766, ['fastestLapTimeSec']] 
s = df.loc[23766, ['fastestLapSpeed']].astype('float64')
r = s[0] / t[0]
r


# In[ ]:


newSpeed = r * df.loc[23764, ['fastestLapTimeSec']][0]
df.loc[23764, ['fastestLapSpeed']] = str(newSpeed)
df[(df['year'] == 2017) & (df['country'] == 'UAE')][['driverRef', 'fastestLapTime', 'fastestLapTimeSec', 'fastestLapSpeed']]


# We can now build our new speed column in meters per second

# In[ ]:


df['fastestLapSpeed'] = df['fastestLapSpeed'].astype('float64')
df['fastestLapSpeedMetersPerSec'] = 1000 * df['fastestLapSpeed'] / 3600


# We can now calculte the approx length in meter of the track using the fastest lap

# In[ ]:


df['fastestLapLength'] = df['fastestLapTimeSec'] * df['fastestLapSpeedMetersPerSec']
df['fastestLapLength'].tail(5)


# ### 5. Sense checking

# #### 5.1. Our circuit length calculation

# https://www.google.com/search?q=monza+circuit+length&rlz=1C1GCEA_enGB804GB804&oq=monza+circuit+length&aqs=chrome..69i57.5270j1j4&sourceid=chrome&ie=UTF-8

# In[ ]:


df[(df['year'] == 2017) & (df['country'] == 'Italy')][['driverRef', 'circuitName', 'fastestLapLength']]


# #### 5.2. The number of points of each driver got at the end of the 2017 championship

# https://en.wikipedia.org/wiki/2017_Formula_One_World_Championship

# In[ ]:


df[df['year'] == 2017].groupby(['driverRef']).agg({'points': 'sum'}).sort_values(by = 'points', ascending = False).plot.bar()


# #### 5.3. Ricciardo's fastest lap time at Monza in 2017

# https://gpracingstats.com/circuits/monza/

# In[ ]:


df[(df['year'] == 2017) & (df['surname'] == 'Ricciardo') & (df['country'] == 'Italy')][['fastestLapTime', 'fastestLapSpeed']]


# ### 6. So... Are F1 cars faster?

# Let's create a data frame for that question

# In[ ]:


df_speed = df.copy()


# Let's also record the length of that df

# In[ ]:


orig_df_speed_len = len(df_speed)


# 77% of our observations are missing the speed info

# In[ ]:


len(df_speed[df_speed['fastestLapSpeed'].isna()]) / len(df_speed)


# #### 6.1. Dealing with missing speed values pre 2004

# Looks like we have a lot less missing speeds from 2004 onwards

# In[ ]:


df_speed[df_speed['fastestLapSpeed'].isna()].groupby(['year']).agg({'resultId': 'count'}).plot()


# In[ ]:


df_speed[df_speed['fastestLapSpeed'].isna()].groupby(['year']).agg({'resultId': 'count'}).loc[2000:]


# So let's remove all the data before that season from our speed data frame

# In[ ]:


df_speed = df_speed[df_speed['year'] >= 2004]


# A few speed nulls remain

# In[ ]:


df_speed.isna().sum()


# In[ ]:


df_speed['fastestLapSpeed'].isna().sum() / len(df_speed)


# #### 6.2. Dealing with missing speed values for the Shanghai GP 2011

# Let's look into the 5% of 2004 - 2017 data that has no fastestLapSpeed

# In[ ]:


df_speed[df_speed['fastestLapSpeed'].isna()].groupby(['year', 'country'], as_index = False).agg({'resultId': 'count'}).sort_values(by = 'resultId', ascending = False)


# Looks like no speed data is available for any driver in the 2011 Shanghai GP

# In[ ]:


df_speed[(df_speed['country'] == 'China') & (df_speed['year'] == 2011)]


# Let's drop this GP

# In[ ]:


df_speed = df_speed[df_speed['raceId'] != 843]


# #### 6.3. Dealing Kvyat's missing speed values for the Russian GP 2015

# Only one guy finished despite having no fastest lap speed, who dat?

# In[ ]:


df_speed[df_speed['fastestLapSpeed'].isna()]['status'].value_counts()


# That's Daniil Kvyat

# In[ ]:


df_speed[(df_speed['fastestLapSpeed'].isna()) & (df_speed['status'] == 'Finished')]


# In[ ]:


df_speed.loc[22832]


# Let's find Kvyat's speed on his fastest lap in the lapTimes dataframe (raceId 941, driverId 826); first we need to find his fastest lap

# In[ ]:


df_kvy = data_dict['lapTimes'][(data_dict['lapTimes']['raceId'] == 941) & (data_dict['lapTimes']['driverId'] == 826)]
df_kvy.sort_values(by = 'milliseconds').head(5)


# We also need the length of the track

# In[ ]:


df_speed[(df_speed['country'] == 'Russia') & (df_speed['year'] == 2015)]['fastestLapLength'].median()


# We can now calculate the speed

# In[ ]:


t = df_kvy.loc[106791, ['milliseconds']][0] / 1000
d = df_speed[(df_speed['country'] == 'Russia') & (df_speed['year'] == 2015)]['fastestLapLength'].median()
s = d / t
s = 3.6 * s
s


# And replace the missing value in df_speed

# In[ ]:


df_speed.loc[22832, ['fastestLapSpeed']] = s
df_speed[(df_speed['country'] == 'Russia') & (df_speed['year'] == 2015)][['driverRef', 'position', 'fastestLapSpeed']]


# #### 6.4. Dealing the remaining missing speed values: race issues

# All the remaining speed nulls seem to belong to drivers with race issues

# In[ ]:


df_speed[df_speed['fastestLapSpeed'].isna()]['status'].value_counts()


# These nulls represent 4.7% of the remaining records

# In[ ]:


df_speed['fastestLapSpeed'].isna().sum() / len(df_speed)


# So let's define finishers (i.e. drivers without race issues)

# In[ ]:


finisher = df_speed[(df_speed['status'].str.startswith('+')) | (df_speed['status'] == 'Finished')]['status'].drop_duplicates()
finisher = list(finisher)
finisher


# 77% of the remaining records have a status falling under the finisher statuses defined above

# In[ ]:


len(df_speed[(df_speed['status'].isin(finisher))]) / len(df_speed)


# Let's drop the non finisher records

# In[ ]:


df_speed = df_speed[(df_speed['status'].isin(finisher))]


# No null left in the fastestLapSpeed column

# In[ ]:


df_speed.isna().sum()


# Overall we got rid of 82% of the data in df_speed

# In[ ]:


1 - (len(df_speed) / orig_df_speed_len)


# #### 6.5. Aggregating the data by year

# In[ ]:


df_speed = df_speed.groupby(['year', 'country'], as_index = False).agg({'fastestLapSpeed': np.median, 'fastestLapLength': np.mean})
df_speed.rename(columns = {'fastestLapSpeed': 'medianFastestLapSpeed', 'fastestLapLength': 'approxLapLength'}, inplace = True)

countries = list(df_speed['country'].drop_duplicates())


# #### 6.6. Visualising speed vs track length

# Track length as an indicator of whether the track has changed or not
# 

# In[ ]:


fig, ax = plt.subplots(figsize = (5, 5))

country = 'Australia'

df_temp = df_speed[df_speed['country'] == country]

ax.scatter(df_temp['year'], df_temp['medianFastestLapSpeed'], color = '#4B878BFF')
ax.set(xlabel = 'year'
       , ylabel = 'median fasted lap speed (km/h)'
       , title = country
       , ylim = (0, 260)
       , xlim = (2003, 2018))

ax2 = ax.twinx()
ax2.plot(df_temp['year'], df_temp['approxLapLength'] / 1000, color = '#D01C1FFF')
ax2.set(xlabel = 'year'
        , ylabel = 'approx lap lenghth (km)'
        , ylim = (0, 6)
        , xlim = (2003, 2018))

ax2.spines['left'].set_color('#4B878BFF')
ax2.spines['right'].set_color('#D01C1FFF')

plt.tight_layout()


# Using a loop to display all GPs

# In[ ]:


countries = list(df_speed['country'].drop_duplicates())
len(countries)


# In[ ]:


fig, ax = plt.subplots(nrows = 5, ncols = 5, figsize = (25, 20))

i = 0 
for r in range(0, 5):
    for c in range(0, 5):
        if i < len(countries):
            ax1 = ax[r, c]
            df_temp = df_speed[df_speed['country'] == countries[i]]
            # primary vertical axis
            ax1.scatter(df_temp['year'], df_temp['medianFastestLapSpeed'], color = '#4B878BFF')
            ax1.set(xlabel = 'year'
                    , ylabel = 'median fasted lap speed (km/h)'
                    , title = '{} GP: median fastest lap by season'.format(countries[i])
                    , ylim = (0, 260)
                    , xlim = (2003, 2018))
            # secondary vertical axis
            ax2 = ax1.twinx()
            ax2.plot(df_temp['year'], df_temp['approxLapLength'] / 1000, color = '#D01C1FFF')
            ax2.set(xlabel = 'year'
                    , ylabel = 'approx lap lenghth (km)'
                    , ylim = (0, 8)
                    , xlim = (2003, 2018))
            # colouring the axis
            ax2.spines['left'].set_color('#4B878BFF')
            ax2.spines['right'].set_color('#D01C1FFF')
            # colouring the ticks
            ax1.tick_params(axis='y', labelcolor = '#4B878BFF')
            ax2.tick_params(axis='y', labelcolor = '#D01C1FFF')
            # colouring the axis labels
            ax1.yaxis.label.set_color('#4B878BFF')
            ax2.yaxis.label.set_color('#D01C1FFF')
            # counter
            i = i + 1
        else:
            break
fig.tight_layout()
plt.show()


# Let's list the GPs for which the track hasn't changed  and all races occured between 2004 and 2017

# In[ ]:


focus_countries = ['Australia', 'Brazil', 'Hungary', 'Malaysia', 'Monaco']
len(focus_countries)


# ### 7. Packaging things nicely for communication

# Are F1 cars getting faster? Looking at the median fastest lap year after year, it seems that they've actually been getting gradually slower up until 2004 and have then gotten faster since.
# 
# Source: "Formula 1 Race Data" by Chris G available on Kaggle (https://www.kaggle.com/cjgdev/formula-1-race-data-19502017, accessed 20/04/2020)
# - 2004-2017 data
# - displaying data for only 5 tracks (the 5 that haven't changed and have been used every year over the period)
# - data from drivers who haven't finished the race is excluded

# In[ ]:


for country in focus_countries:
    fig, ax = plt.subplots(figsize = (7, 7))
    df_temp = df_speed[df_speed['country'] == country]
    # scatter plot
    ax.scatter(df_temp['year'], df_temp['medianFastestLapSpeed'], color = '#4B878BFF')
    ax.set(xlabel = 'year'
            , ylabel = 'median fasted lap speed (km/h)'
            , title = '{} GP: speed by season (labels for 2004, 2014, 2017)'.format(country)
            , ylim = (0, 240)
            , xlim = (2002, 2018))
    # adding vertical line for 2014
    ax.axvline(x = 2014, color = '#a3a3a3', linewidth = 1, linestyle = 'dashed')
    # adding labels for the 3 key years
    plt.text(x = 2004
             , y = df_temp.iloc[0, 2] + 5
             , s = '{} km/h'.format(round(df_temp.iloc[0, 2], 1))
             , size = 8
             , color = '#4B878BFF'
             , ha = 'center')
    plt.text(x = 2014
             , y = df_temp.iloc[10, 2] - 15
             , s = '{} km/h'.format(round(df_temp.iloc[10, 2], 1))
             , size = 8
             , color = '#4B878BFF'
             , ha = 'center')
    plt.text(x = 2017
             , y = df_temp.iloc[13, 2] - 15
             , s = '{} km/h'.format(round(df_temp.iloc[13, 2], 1))
             , size = 9
             , color = '#4B878BFF'
             , ha = 'center')
plt.show()


# ### 8. Plenty of other things we could have looked into

# #### 8.1. Finland: population 5 million but so many drivers

# In[ ]:


# data stuff
df_fin = df.groupby(['nationality'], as_index = False).agg({'points': 'sum'}).sort_values('points', ascending = False).copy()
df_fin['points_pct'] = df_fin['points'] / df_fin['points'].sum()
df_fin = df_fin.iloc[0:10]

# viz stuff
import matplotlib.ticker as mtick
fig, ax = plt.subplots(figsize = (14, 5))
ax.bar(df_fin['nationality'], 100 * df_fin['points_pct'], color = '#4B878BFF')
ax.set(xlabel = 'nationality'
       , ylabel = '% of all points earned'
       , title = 'Top 10 nationalities in terms of percentage of championship points earned')
ax.get_children()[3].set_color('#D01C1FFF') 
plt.text(x = 3 - 0.15
         , y = 100 * df_fin['points_pct'].iloc[3] + 0.5
         , s = '{}%'.format(round(100 * df_fin['points_pct'].iloc[3], 2))
         , size = 10
         , color = '#D01C1FFF'
         , weight = 'bold')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()


# #### 8.2.  How much does the grid position matters?

# In[ ]:


# data stuff
df_rel = df[['grid', 'position']].copy()
df_rel = df_rel.dropna()

# viz stuff
fig, ax = plt.subplots(figsize = (6, 6))
ax.scatter(df_rel['grid'], df_rel['position'], alpha = 0.05, color = '#4B878BFF')
ax.set(xlabel = 'grid position'
       , ylabel = 'final position'
       , title = 'Grid position vs final position')
plt.show()


# #### 8.3.  Long term constructor performance

# In[ ]:


# data stuff
df_cns = data_dict['constructorStandings'][['raceId', 'constructorId', 'points', 'position']].copy()
df_cns = df_cns.merge(data_dict['constructors'], how = 'left')
df_cns = df_cns.merge(data_dict['races'][['raceId', 'year', 'raceName', 'round']], how = 'left')
df_cns = df_cns[['constructorName', 'position', 'year', 'round']]
df_fnl = df_cns.groupby(['year'], as_index = False).agg({'round': np.max})
df_fnl.rename(columns = {'round': 'finalRound'}, inplace = True)
df_cns = df_cns.merge(df_fnl, how = 'left')
df_cns = df_cns[df_cns['round'] == df_cns['finalRound']]
df_cns = df_cns[['constructorName', 'position', 'year']]
focus_constructors = ['Mercedes', 'Ferrari', 'Red Bull', 'BRM', 'Tyrrell', 'Force India', 'Jordan']
focus_palette = {'Mercedes': '#707070'
                 , 'Ferrari': '#de0000'
                 , 'Red Bull': '#001496'
                 , 'BRM': '#228243'
                 , 'Tyrrell': '#4dc3eb'
                 , 'Force India': '#ff85ef'
                 , 'Jordan': '#fcd700'}
df_cns = df_cns[df_cns['constructorName'].isin(focus_constructors)]

# viz stuff
fig, ax = plt.subplots(figsize = (15, 5))
ax = sns.lineplot(data = df_cns
                  , x = 'year'
                  , y = 'position'
                  , hue = 'constructorName'
                  , palette = focus_palette
                  , hue_order = focus_constructors)
ax.set(xlabel = 'round'
       , ylabel = 'rank'
       , title = 'Final season ranking for few constructors since 1958')
ax.legend(loc = 'upper left')
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer = True))
ax.yaxis.set_major_locator(MaxNLocator(integer = True))
plt.show()

