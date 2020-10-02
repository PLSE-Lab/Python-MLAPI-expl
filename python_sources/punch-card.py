# # How long does Hillary work
# 
# My main goal with the analysis is to use these released emails to estimate her work hours. I'm 
# doing this by looking at the Sent times for emails that originanted from Hilary Clinton. Due to
# gaps in the release I've limited this look to dates between May 2009 to July 2010.
from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sbn
sbn.set_context("notebook", font_scale=1.5)
sbn.set_style('whitegrid')
# ## Data Prep
# 
# We'll need to load in the data and use some `pandas` magic to convert things to the datetime 
# objects. 
em_df = pd.read_csv('../input/Emails.csv')
for col in em_df.columns:
    if 'Date' in col:
        em_df[col] = pd.to_datetime(em_df[col], errors='coerce').copy()
        
def safe_len(txt):
    try:
        return len(txt)
    except:
        return np.nan

em_df['TextLength'] = em_df['RawText'].map(safe_len)
per_df = pd.read_csv('../input/Persons.csv')
per_df.ix[per_df['Name'].str.startswith('H')]
hil_df = em_df.query('SenderPersonId == 80')[['ExtractedDateSent', 'SenderPersonId', 'TextLength']].dropna() 
hil_df.head()
import matplotlib.dates as dt

def hours_from_midnight(tm):
    
    mid = pd.to_datetime(tm.date())
    delta = tm-mid
    return delta/np.timedelta64(60*60, 's')

hil_df['Hour'] = hil_df['ExtractedDateSent'].map(lambda x: x.hour)
hil_df['Time'] = hil_df['ExtractedDateSent'].map(lambda x: x.time())
hil_df['Date'] = hil_df['ExtractedDateSent'].map(lambda x: x.date())
hil_df.head()
# ## Results
rdf = hil_df[['Date', 'TimeNum']].dropna().groupby('Date')['TimeNum'].agg(['max', 'min'])
rdf['delta'] = rdf['max'] - rdf['min']
rdf['Last Email'] = pd.rolling_mean(rdf['max'], 7)
rdf['First Email'] = pd.rolling_mean(rdf['min'], 7)
fig, time_ax = plt.subplots(1, 1, figsize=(10, 5), sharex=True)

rdf.plot(y = ['Last Email', 'First Email'], ax=time_ax)
time_ax.set_xlim('5/1/2009', '7/1/2010')
time_ax.set_ylim(0, 24)
time_ax.set_yticks(range(0, 25, 4))
time_ax.set_ylabel('Time of day')
# Legend: The time of day of the first and last email (green and blue respectively)
# was averaged using a 1 week rolling mean.
# 
# From this figure we can see a few things. There are some shifting of schedules, notably around 
# October 2009. This could represent either shifts in work hours or changes in time-zones. 
# Using a simple Google News search it seems like she was in Ireland around Oct 16th. We can 
# also see that the width between the two lines fluctutes.
rdf['Active Time'] = rdf['Last Email'] - rdf['First Email']
fig, (tm_ax, hist_ax) = plt.subplots(1, 2, figsize=(12, 5))

rdf.plot(y='Active Time', ax=tm_ax, legend=False)
tm_ax.set_xlim('5/1/2009', '7/1/2010')
tm_ax.set_ylabel('Hours')
loc = dt.AutoDateLocator(minticks=4, maxticks=5)
tm_ax.xaxis.set_major_locator(loc)
tm_ax.set_ylim(0, 18)

sbn.distplot(rdf['Active Time'].dropna(), ax=hist_ax, vertical=True)
hist_ax.set_xlabel('Density')
hist_ax.set_xlim(0, 0.2)
hist_ax.invert_xaxis()
hist_ax.yaxis.set_label_position("right")
hist_ax.yaxis.tick_right()
hist_ax.set_ylim(0, 18)

fig.tight_layout()
# Legend: (Left) The time between the first and last emails of the by day. 
# (Right) A normalized histogram and Kernal Density estimation of the times shown on the left.
# 
# Here we see that Hillary traditionally puts in a 7-9 hour day (at least according to email times).
hil_df['DateNum'] = hil_df['ExtractedDateSent'].map(lambda x: dt.date2num(x))
hil_df['TimeNum'] = hil_df['ExtractedDateSent'].map(hours_from_midnight)

fig, ax = plt.subplots(1,1, figsize=(10, 10))
sbn.kdeplot(data=hil_df['DateNum'], data2=hil_df['TimeNum'], shade=True)
ax.set_xlim(733500, 734000)
ax.set_ylim(0, 24)
loc = dt.AutoDateLocator(minticks=5, maxticks=6)
ax.set_yticks(range(0, 25, 4))
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(dt.AutoDateFormatter(loc))
ax.set_ylabel('Time of day')
ax.set_xlabel('Date')
ax.set_title('Punch Card by Email Sent Time')
# Legend: A bivariate KDE was fit using the date and time of day of emails sent by Hillary Clinton 
# with darker values indicating a higher density of emails.
# 
# This lets us see that the majority of emails are sent around 8AM regardless of the date.
# The constant string of early emails is supplemented by emails around 4PM (in Sept 2009 and 
# Feb 2010) as well as 10PM (in Feb 2010). Some Google News searching reveals that Feb 2010 when 
# talks with Iran were a main focus.