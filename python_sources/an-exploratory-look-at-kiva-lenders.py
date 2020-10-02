#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
from numpy import log10, ceil, ones
from numpy.linalg import inv 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # prettier graphs
import matplotlib.pyplot as plt # need dis too
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import HTML # for da youtube memes
import itertools # let's me iterate stuff
from datetime import datetime # to work with dates
import geopandas as gpd
from fuzzywuzzy import process
from shapely.geometry import Point, Polygon
import shapely.speedups
shapely.speedups.enable()
import fiona 
from time import gmtime, strftime
from shapely.ops import cascaded_union
import gc

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

sns.set_style('darkgrid') # looks cool, man
import os

df_cntry_codes = pd.read_csv("../input/countries-iso-codes/wikipedia-iso-country-codes.csv")
df_cntry_codes = df_cntry_codes.rename(index=str, columns={'Alpha-2 code': 'country_code', 'English short name lower case' : 'country'})


# # An Exploratory Look at Kiva *Lenders*
# At the moment, I've been unable to do any work on my original Kernel [Kiva Exploration by a Kiva Lender and Python Newb
# ](https://www.kaggle.com/doyouevendata/kiva-exploration-by-a-kiva-lender-and-python-newb)  for days.  I am hoping Kaggle can recover it soon.  I had been doing all work on the original base provided data, although I've added Beluga's [great dataset](https://www.kaggle.com/gaborfodor/additional-kiva-snapshot/data) to this Kernel you are viewing now to poke around at in the mean time, because it had some more interesting information around lender data.  I am also working on learning python at the moment, so wanted something to kee practicing on.  In this Kernel I plan on simply poking around at the Kiva lenders, rather than Kiva borrowers.  It's probably not prize track worthy stuff, but it is neat and interesting.  Update: created third poverty targeting methodolgy kernel: [Kiva Povery Targeting](https://www.kaggle.com/doyouevendata/kiva-poverty-targeting/notebook).
# 
# * [1. The Data](#1)
# * [2. Kiva Registered User Locations](#2)
# * [3. Kiva Active User Locations](#3)
# * [4. Loans Per User Per Country](#4)
#   * [4.1 Top Loans Per User Per Country](#4_1)
#   * [4.2 Top Loans Per User Per Country, >= 50 Registered Users](#4_2)
#     * [4.2.1 Oman](#4_2_1)
#     * [4.2.2 Kazakhstan](#4_2_2)
#     * [4.2.3 Thailand](#4_2_3)
#     * [4.2.4 Switzerland](#4_2_4)
#     * [4.2.5 Iran](#4_2_5)
#   * [4.3 Top Loans Per User Per Country, >= 1,000 Registered Users](#4_3)
#   * [4.4 Top Loans Per User Per Country, >= 10,000 Registered Users](#4_4)
# * [5. US Users By State](#5)
#   * [5.1 Registered Users By State](#5_1)
#   * [5.2 Registered Users Per State, Population Weighted](#5_2)
#   * [5.3 Loans Per User Per State](#5_3)
# * [6. Kiva Whales](#6)
#   * [6.1 Big Lenders](#6_1)
#   * [6.2 Big Inviters](#6_2)
#   * [6.3 Big Amount Lenders](#6_3)
# * [7. Loans Per User - Minus Outliers](#7)
#   * [7.1 Top Loans Per User Per Country, >= 1,000 Registered Users, Minus Outliers](#7_1)
#   * [7.2 Top Loans Per User Per Country, >= 10,000 Registered Users, Minus Outliers](#7_2)
#   * [7.3 Loans Per User Per State, Minus Outliers](#7_3)
# * [8. Lender Occupations](#8)
# * [9. Trends](#9)
#   * [9.1 Monthly Active Users](#9_1)
#   * [9.2 Average and Meidan Days Since Last Visit](#9_2)
#   * [9.3 Unfunded Loan Gap](#9_3)
# * [10. Recommendations](#10)
# 
# <a id=1></a>
# # 1. The Data
# What info do we have for lenders?  Unfortunately plenty of nulls.  I was able to find myself in here too!  I think this may actually be a complete dump of all Kiva data.  Lender data:

# In[2]:


df_lenders = pd.read_csv("../input/additional-kiva-snapshot/lenders.csv")
df_lenders.head()


# In[3]:


df_lenders[df_lenders['permanent_name'] == 'mikedev10']


# The loan-lenders mapping.

# In[4]:


df_loans_lenders = pd.read_csv("../input/additional-kiva-snapshot/loans_lenders.csv")
df_loans_lenders.head()


# The loans - a particularly large file!

# In[5]:


df_loans = pd.read_csv("../input/additional-kiva-snapshot/loans.csv")
df_loans['posted_time'] = pd.to_datetime(df_loans['posted_time'], format='%Y-%m-%d %H:%M:%S').dt.round(freq='D')
df_loans['raised_time'] = pd.to_datetime(df_loans['posted_time'], format='%Y-%m-%d %H:%M:%S').dt.round(freq='D')
df_loans['posted_month'] = df_loans['posted_time'].dt.to_period('M').dt.to_timestamp()
df_loans.head()


# There is a loan_purchase_num in the lenders file telling us how many loans a lender has made.  I am curious if this is due to database design and if Kiva uses a NoSQL database and the model makes this an attribute of a lender?  I come from the data warehousing world and really like the idea of counting things from the atomic layer up.  In my world I am of course running reports in this scenario, whereas a NoSQL database is more about running an application.  In any case, we can get more interesting data from the atomic loan level - namely the time of a loan.
# 
# So here's what we'll do - split out lenders from loans to have multiple rows; then join them to populated lender country info to map countries, then to the loans to get times.  This will probably take some time to chooch through and give us a pretty tall dataset of atomic date-user-loan-country values.  We can roll this up to date-country-count and take a look at aggregates as well as trends over time.

# In[6]:


# thanks to
# https://stackoverflow.com/questions/38651008/splitting-multiple-columns-into-rows-in-pandas-dataframe

# STEP 1 - explode lenders out of of column entry into multiple rows
def explode(df, columns):
    idx = np.repeat(df.index, df[columns[0]].str.len())
    a = df.T.reindex_axis(columns).values
    concat = np.concatenate([np.concatenate(a[i]) for i in range(a.shape[0])])
    p = pd.DataFrame(concat.reshape(a.shape[0], -1).T, idx, columns)
    return pd.concat([df.drop(columns, axis=1), p], axis=1).reset_index(drop=True)

# THE BELOW WAS REPLACED WITH OUTPUT AND READ OF A FILE TO GET AROUND CONSTANT NOTEBOOK CRASHING AND ELIMINATE DUPLICATES
#df_exp = df_loans_lenders
#df_exp['lenders'] = df_exp['lenders'].str.split(',')
#df_exp = explode(df_exp, ['lenders'])
#df_exp = df_exp.rename(index=str, columns={'lenders': 'permanent_name'})
#df_exp['permanent_name'] = df_exp['permanent_name'].str.strip()
#df_exp = df_exp.drop_duplicates()
df_exp = pd.read_csv("../input/kiva-lender-helper/df_exp.csv")
#dupe check
#df_exp[df_exp['loan_id'] == 885412]


# In[8]:


# STEP 2 - map users to countries
df_lender_cntry = df_exp.merge(df_lenders[['permanent_name', 'country_code']], on='permanent_name')
df_lender_cntry.dropna(axis=0, how='any', inplace=True)
#df_lender_cntry

# STEP 3 - merge users to loans and aggregate count by country and day
df_cntry_cnts = df_lender_cntry.merge(df_loans[['loan_id', 'posted_time']], on='loan_id')[['country_code', 'posted_time']].groupby(['country_code', 'posted_time']).size().reset_index(name='counts')
#df_cntry_cnts.head()

# STEP 4 - let's make life easier with these country codes...
df_cntry_cnts = df_cntry_cnts.merge(df_cntry_codes[['country_code', 'country']], on='country_code', how='left')
df_cntry_cnts['country'] = np.where(df_cntry_cnts['country_code'] == 'SS', 'South Sudan', df_cntry_cnts['country'])  
df_cntry_cnts['country'] = np.where(df_cntry_cnts['country_code'] == 'XK', 'Kosovo', df_cntry_cnts['country'])  


df_cntry_cnts.head()


# <a id=2></a>
# # 2. Kiva Registered User Locations
# Where are people lending from?

# In[ ]:


plt.figure(figsize=(15,8))
plotSeries = df_lenders['country_code'].value_counts()
ax = sns.barplot(plotSeries.head(30).values, plotSeries.head(30).index, color='c')
ax.set_title('Top 30 Lender Locations', fontsize=15)
ax.set(ylabel='Country (ISO-2 Abbrev)', xlabel='Lender Count')
plt.show()


# The US dwarfs everyone else.  Kiva is, indeed, a US non-profit.  It has nearly 600,000 registered users, with Canada encroaching on 70,000 as the next most popular place for contributors to live.
# 
# What's this breakdown look like without the US?

# In[ ]:


plt.figure(figsize=(15,8))
plotSeries = df_lenders[df_lenders['country_code'] != 'US']['country_code'].value_counts()
ax = sns.barplot(plotSeries.head(29).values, plotSeries.head(29).index, color='b')
ax.set_title('Top 30 Lender Locations, Minus the US', fontsize=15)
ax.set(ylabel='Country (ISO-2 Abbrev)', xlabel='Lender Count')
plt.show()


# <a id=3></a>
# # 3. Kiva Active User Locations
# Ok - now instead of a count of *user locations* - let's take a look at the count of *active user contributions* by country.  We'll show with and without the US again.

# In[ ]:


df_cntry_sum = df_cntry_cnts.groupby(['country', 'country_code']).sum()
df_cntry_sum.reset_index(level=1, inplace=True)
df_cntry_sum.reset_index(level=0, inplace=True)
df_display = df_cntry_sum.sort_values('counts', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='counts', y='country', data=df_display, color='c')

ax.set_title('Top 30 Locations by Lender Contribution Count', fontsize=15)
ax.set(ylabel='Country', xlabel='Number of Loan Contributions')
plt.show()


# This graph looks more similar than I thought, at least in regards to the amount.  The US has well over 600k *loan contributions* now, although, I suspected it would have more.  Note that plenty of users may simply try the system once, but never actually get that into the platform.

# In[ ]:


df_display = df_cntry_sum[df_cntry_sum['country_code'] != 'US'].sort_values('counts', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='counts', y='country', data=df_display, color='b')

ax.set_title('Top 30 Locations by Lender Contribution Count, Minus the US', fontsize=15)
ax.set(ylabel='Country', xlabel='Number of Loan Contributions')
plt.show()


# We do see some jostling above - GB was above AU for registered users; although relatively close.  Here, however, we see AU users take 2nd position and have in fact made many more contributions than GB!  Contributions are made in incremental amounts of $25 USD - and we don't know of what size each lender is contributing, but AU users have certainly participated in funding more loans on a per person basis.  Which would of course be an interesting metric to rank by; loans per country / users per country.
# <a id=4></a>
# # 4. Loans Per User Per Country
# <a id=4_1></a>
# ## 4.1 Top Loans Per User Per Country

# In[ ]:


df_lender_sum = df_lenders.groupby(['country_code']).size().reset_index(name='counts_reg')
df_lender_sum = df_lender_sum.merge(df_cntry_sum, on='country_code')
df_lender_sum['loans_per_lender'] = df_lender_sum['counts'] / df_lender_sum['counts_reg']
df_lender_sum.head()


# <a id=4_2></a>
# ## 4.2 Top Loans Per User Per Country, >= 50 Registered Users

# In[ ]:


df_display = df_lender_sum.sort_values('loans_per_lender', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_lender', y='country', data=df_display, color='r')

ax.set_title('Top 30 Locations by Lender Contribution Count Per Registered User', fontsize=15)
ax.set(ylabel='Country', xlabel='Number of Loan Contributions')
plt.show()


# Wow - the US is now gone!  In some ways, this makes sense; the US would certainly have the largest amount of people who try the site, or might get a gift card, but then don't really get into it.  Whereas many outside of the US who want to try Kiva, would be much more motivated users; to simply even start, they need to fund it via something like Paypal with an international currency conversion.  This is probably someone who is motivated by Kiva's mission and likely to be a more active user.  The raw data is below, and we can see that counts_reg can really skew the chart.  BV is the Norwegian dependent territory of Bouvet Island.  On which we have 3 pretty active Kiva contributors.  The top 10 are below.  Let's try plotting this but requiring at least 50 registered users.

# In[ ]:


df_display.head(10)


# In[ ]:


df_display = df_lender_sum[df_lender_sum['counts_reg'] >= 50].sort_values('loans_per_lender', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_lender', y='country', data=df_display, color='orange')

ax.set_title('Top 30 Locations by Lender Contribution Count Per Registered User, >= 50 Users', fontsize=15)
ax.set(ylabel='Country', xlabel='Number of Loan Contributions Per User')
plt.show()


# This is really interesting!  How many of these people are expats?  How many of these people are native born citizens of their countries?  How did these people come to find Kiva?  I'm not going to try and analzye this with python but will allow the reader to peruse some of their profile provided information for some of these top countries.
# <a id=4_2_1></a>
# ### 4.2.1 Oman

# In[ ]:


df_lenders[df_lenders['country_code'] == 'OM'][['country_code', 'occupation', 'loan_because']].dropna(axis=0, how='any')


# <a id=4_2_2></a>
# ### 4.2.2 Kazakhstan

# In[ ]:


df_lenders[df_lenders['country_code'] == 'KZ'][['occupation', 'loan_because']].dropna(axis=0, how='all')


# <a id=4_2_3></a>
# ### 4.2.3 Thailand

# In[ ]:


df_lenders[df_lenders['country_code'] == 'TH'][['occupation', 'loan_because']].dropna(axis=0, how='all')


# <a id=4_2_4></a>
# ### 4.2.4 Switzerland

# In[ ]:


df_lenders[df_lenders['country_code'] == 'CH'][['occupation', 'loan_because']].dropna(axis=0, how='all')


# <a id=4_2_5></a>
# ### 4.2.5 Iran

# In[ ]:


df_lenders[df_lenders['country_code'] == 'IR'][['occupation', 'loan_because']].dropna(axis=0, how='all')


# I will note some of these registrations appear to be in error - the loan_because field indicates someone was registering on the site to *ask for a loan* rather than be a loan contributor.  I believe all loan requests are done through field partners, except for special case direct Kiva loans, which are even still not done to just regular users.  The amazing part about this is that these people who signed up in error here are even skewing the loan contributions per registered user downward!  Without them, these top countries would have even greater loan contributions per user.  Amazing!
# <a id=4_3></a>
# ## 4.3 Top Loans Per User Per Country, >= 1,000 Registered Users

# In[ ]:


df_display = df_lender_sum[df_lender_sum['counts_reg'] >= 1000].sort_values('loans_per_lender', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_lender', y='country', data=df_display, color='purple')

ax.set_title('Top 30 Locations by Lender Contribution Count Per Registered User, >= 1,000 Users', fontsize=15)
ax.set(ylabel='Country', xlabel='Number of Loan Contributions Per User')
plt.show()


# Now we see the US on the board.
# <a id=4_4></a>
# ## 4.4 Top Loans Per User Per Country, >= 10,000 Registered Users

# In[ ]:


df_display = df_lender_sum[df_lender_sum['counts_reg'] >= 10000].sort_values('loans_per_lender', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_lender', y='country', data=df_display, color='green')

ax.set_title('Top Locations by Lender Contribution Count Per Registered User, >= 10,000 Users', fontsize=15)
ax.set(ylabel='Country', xlabel='Number of Loan Contributions Per User')
plt.show()


# In[ ]:


# Youtube
HTML('Just look at the Netherlands go!  They are running a great campaign!<br><iframe width="560" height="315" src="https://www.youtube.com/embed/ELD2AwFN9Nc?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# 

# <a id=5></a>
# # 5. US Users By State
# <a id=5_1></a>
# ## 5.1 Registered Users By State

# In[ ]:


# clean up some data first
df_lenders['state'] = np.where(df_lenders['state'].str.len() <= 3, df_lenders['state'].str.upper().str.strip('.').str.strip(), df_lenders['state'].str.title())
df_lenders['state'] = np.where(df_lenders['state'].str.contains('California'), 'CA', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Texas'), 'TX', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('New York'), 'NY', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Florida'), 'FL', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Washington'), 'WA', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Illinois'), 'IL', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Colorado'), 'CO', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Pennsylvania'), 'PA', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Oregon'), 'OR', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Ohio'), 'OH', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Michigan'), 'MI', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Georgia'), 'GA', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Massachusetts'), 'MA', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Indiana'), 'IN', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Missouri'), 'MO', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Virginia'), 'VA', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Minnesota'), 'MN', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('North Carolina'), 'NC', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Arizona'), 'AZ', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Wisconsin'), 'WI', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Maryland'), 'MD', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('New Jersey'), 'NJ', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Kentucky'), 'KY', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Kansas'), 'KS', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Oklahoma'), 'OK', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Utah'), 'UT', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Tennessee'), 'TN', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('District of Columbia'), 'DC', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Iowa'), 'IA', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Connecticut'), 'CT', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Alabama'), 'AL', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Louisiana'), 'LA', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Idaho'), 'ID', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('South Carolina'), 'SC', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Maine'), 'ME', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Arkansas'), 'AR', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('New Mexico'), 'NM', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Hawaii'), 'HI', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Alaska'), 'AK', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('New Hampshire'), 'NH', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Nebraska'), 'NE', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Vermont'), 'VT', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Nevada'), 'NV', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Montana'), 'MT', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Rhode Island'), 'RI', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('West Virginia'), 'WV', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Mississippi'), 'MS', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Delaware'), 'DE', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('North Dakota'), 'ND', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('Wyoming'), 'WY', df_lenders['state'])
df_lenders['state'] = np.where(df_lenders['state'].str.contains('South Dakota'), 'SD', df_lenders['state'])
df_lenders['state'] = np.where((df_lenders['state'].str.len() > 2) & (df_lenders['state'].str.len() <= 5), 
                               df_lenders['state'].str.upper().str.replace('.', '').str.strip(), df_lenders['state'])


# In[ ]:


plt.figure(figsize=(15,8))
plotSeries = df_lenders[df_lenders['country_code'] == 'US']['state'].value_counts()
ax = sns.barplot(plotSeries.head(30).values, plotSeries.head(30).index, color='c')
ax.set_title('Top 30 Lender State Locations', fontsize=15)
ax.set(ylabel='US State',
       xlabel='Lender Count')
plt.show()


# <a id=5_2></a>
# ## 5.2 Registered Users Per State, Population Weighted

# In[ ]:


# STEP 2 - map users to US states
df_lender_state = df_exp.merge(df_lenders[df_lenders['country_code'] == 'US'][['permanent_name', 'state']], on='permanent_name')
df_lender_state.dropna(axis=0, how='any', inplace=True)
#df_lender_cntry

# STEP 3 - merge users to loans and aggregate count by country and day
df_state_cnts = df_lender_state.merge(df_loans[['loan_id', 'posted_time']], on='loan_id')[['state', 'posted_time']].groupby(['state', 'posted_time']).size().reset_index(name='counts')


# In[ ]:


df_state_sum = df_state_cnts.groupby(['state']).sum()
df_state_sum.reset_index(level=0, inplace=True)

df_lender_sum_state = df_lenders[df_lenders['country_code'] == 'US'].groupby(['state']).size().reset_index(name='counts_reg')
df_lender_sum_state = df_lender_sum_state.merge(df_state_sum, on='state')
df_lender_sum_state['loans_per_lender'] = df_lender_sum_state['counts'] / df_lender_sum_state['counts_reg']

# let's merge in state populations here too
df_state_pop = pd.read_csv("../input/population-by-state/population.csv")
df_state_pop = df_state_pop.rename(index=str, columns={'State': 'state'})
df_lender_sum_state = df_lender_sum_state.merge(df_state_pop, on='state')
df_lender_sum_state['loans_per_capita'] = df_lender_sum_state['counts_reg'] / df_lender_sum_state['Population']

df_lender_sum_state.sort_values('counts_reg', ascending=False).head()


# In[ ]:


df_display = df_lender_sum_state.sort_values('loans_per_capita', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_capita', y='state', data=df_display, color='lightblue')

ax.set_title('Top 30 Lender State Locations, Loans Per Capita', fontsize=15)
ax.set(ylabel='US State', xlabel='Lender Count Per Capita')
plt.show()


# Maybe not a surprise to see DC up there since Kiva is quite politically related.  California, where Kiva is also located, still coming out on top.  Which state actually has the most active users?
# <a id=5_3></a>
# ## 5.3 Loans Per User Per State

# In[ ]:


df_display = df_lender_sum_state.sort_values('loans_per_lender', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_lender', y='state', data=df_display, color='salmon')

ax.set_title('Top 30 Lender State Locations, Loans Per User', fontsize=15)
ax.set(ylabel='US State', xlabel='Number of Loans Per User')
plt.show()


# Wow very interesting!  Incredibly, California has dropped off the chart of the top 30 states!  Note this also may in part be due to...
# <a id=6></a>
# # 6 Kiva Whales
# <a id=6_1></a>
# ## 6.1 Big Lenders
# I know I have a pretty decent amount of loans on Kiva, but wow are there some people with a simply *incredible* amount.  So much so that I feel the top guy maybe even hired someone to make all these loans, that are he just suuuper loves Kiva.  Let's take a look at the distribution.

# In[ ]:


plt.figure(figsize=(12,6))
sns.distplot(df_lenders['loan_purchase_num'].fillna(0), bins=30)
plt.show()


# Yes, that's right...  someone with a single account has thus far made 85,190 loan contributions on Kiva.  The bottom isn't flat on this chart - there are plenty of other heavy hitters in here too.  Here's the top 20 lenders by contributions.

# In[ ]:


df_lenders.sort_values('loan_purchase_num', ascending=False).head(20)[['permanent_name', 'city', 'state', 'country_code', 'occupation', 'loan_because', 'loan_purchase_num', 'num_invited']]


# Very interesting.  from a general browsing standpoint, we can see:
# 1. We've got some solid international friends at Kiva!
# 2. It seems Christians and retirees are represented well (also cats and dogs)
# 3. Some did not count in our current ratings as they do not have a country code.
# 4. These top 20 lenders have a combined 650,322 loan contributions *alone* - nearly the volume or Kiva's originally provided dataset.  At a minimum contribution, this group is responsible for lending out *at least* $16,258,050.  Summing up loan_purchase_num, I got 25,071,662; which means these **top 20 lenders did nearly 2.6% of all contributions** themselves.
# 5. Good Dogg seems like he's probably averaging 20+ loans *a day* - at this point he surely has so many repayments coming back that quite a bit of that is simply coming in and going back out the door.  His daily lending habits seem vastly superior to my daily (not so much) gym habits.  A very good dogg indeed.
# 
# These people are even exceptions to the rule at the top, as can be seen below.

# In[ ]:


for x in range(0,10):
    print('99.' + str(x) + 'th percentile loan_purchase_num is: ' + str(df_lenders['loan_purchase_num'].quantile(0.99 + x/1000)))


# We don't even break 1000 until we are at the very top percentiles.  Although we'd have to get much more finely grained to look at the Kiva super lenders.  How many Kiva lenders have contributed to at least 1000 loans?

# In[ ]:


df_lenders[df_lenders['loan_purchase_num'] >= 1000]['loan_purchase_num'].agg(['count', 'sum'])


# We have 1,810 of the 1000+ loan contributors, making up about 20.5% of the number of contributions.  That's a really incredible amount.  At a minimum of 25 USD each, that means these 1,810 people have lent at least $128.36 million, which is incredible.
# 
# Let's look at the lower scale.  

# In[ ]:


for x in range(1,10):
    print(str(x * 10) + 'th percentile loan_purchase_num is: ' + str(df_lenders['loan_purchase_num'].quantile(x/10)))
for x in range(1,10):
    print('9' + str(x) + 'th percentile loan_purchase_num is: ' + str(df_lenders['loan_purchase_num'].quantile(.9 + x/100)))


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(12,6))
sns.distplot(df_lenders[df_lenders['loan_purchase_num'] <= 50]['loan_purchase_num'].fillna(0), bins=30)

ax.set_title('Distribution - Number of Loan Contributions by User, <= 50 Loans', fontsize=15)
ax.set(xlabel='Loan Contributions by User')
plt.show()


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(12,6))
sns.distplot(df_lenders[(df_lenders['loan_purchase_num'] <= 50) & (df_lenders['country_code'] == 'US')]['loan_purchase_num'].fillna(0), bins=30)

ax.set_title('Distribution - Number of Loan Contributions by User, <= 50 Loans, USA Only', fontsize=15)
ax.set(xlabel='Loan Contributions by User')
plt.show()


# The big question for Kiva is, how do we retain sticky users, who will keep coming back and re-lending out on the site?  Not everyone is going to make 1000 loans, but how many 1s or 2s can we turn into 10s or 20s?
# <a id=6_2></a>
# ## 6.2 Big Inviters
# num_invited is the number of successfully invited Kiva lenders someone has brought to the platform - whether via their invite link or by sending them a gift card.  It did appear that a couple of my invitees never made a loan.  We could account for that but I consider that more trouble than it's worth.  Who has had the most invites, where people at least made an account?  An interesting thing of note is some of these have many more invites than loan contributions.  One seems like it might have been a GoDaddy company effort??  Note "kivastaff" itself is second on the list.

# In[ ]:


df_lenders.sort_values('num_invited', ascending=False).head(20)[['permanent_name', 'city', 'state', 'country_code', 'occupation', 'loan_because', 'loan_purchase_num', 'num_invited']]


# <a id=6_3></a>
# ## 6.3 Big Amount Lenders
# Whether being very sympathetic to these particularly borrowers, or these lenders are short on time, or for some other reason - here's a list of the top 20 loans sorted by funded amount / numbers of lenders - and they are all quite big doozy contributions!  Incredibly, each loan before (and more) were funded by a single Kiva user.  Note there are different Kiva users that show up doing the funding, although it's a small sample of heavy hitters (5 lenders + anonymous make up this set of 20.)

# In[9]:


df_loans['avg_funded'] = df_loans['funded_amount'] / df_loans['num_lenders_total']
df_loans.sort_values('avg_funded', ascending=False).head(20)[['loan_id', 'funded_amount', 'num_lenders_total', 'activity_name', 'description_translated']]


# <a id=7></a>
# # 7 Loans Per User - Minus Outliers
# Some of the outliers weren't counted, for example the 85k lender had state as CA but has a null country value instead of the US.  The Netherlands is small enough that it's big lender could certainly skew its user activity ranking.  These people are great!  However, if we want to look at the standard person, I think we should omit them.  I myself am just shy from 1000 loans, although I'm not sure I should count myself entirely as the 99.8th percentile range hits 707, and my own numbers are not purely from my own political beliefs or altruism; my numbers are goosed, as are likely many a flying buddy's - by a phenomenon in the US called manufactured spending, used to churn money through credit cards for benefits.  It's a somewhat arbitrary line in the sand, but I'm going to draw that line at 707 and omit everything above it.  The below metrics are looking at all the users "only" up to the 99.7th percentile.  They have done 75.84% of the funding.  
# <a id=7_1></a>
# ## 7.1 Top Loans Per User Per Country, >= 1,000 Registered Users, Minus Outliers

# In[ ]:


#df_whale = df_lenders.sort_values('loan_purchase_num', ascending=False).head(20)[['permanent_name']]
#remove outliers
outlier = 707
df_whale = df_lenders[df_lenders['loan_purchase_num'] >= outlier][['permanent_name']]
df_whale['whale'] = 'Y'
df_lenders = df_lenders.merge(df_whale, how='left', on='permanent_name')

# STEP 2 - map users to countries - exclude whales
df_lender_cntry = df_exp.merge(df_lenders[df_lenders['whale'].isnull()][['permanent_name', 'country_code']], on='permanent_name')
df_lender_cntry.dropna(axis=0, how='any', inplace=True)
#df_lender_cntry

# STEP 3 - merge users to loans and aggregate count by country and day
df_cntry_cnts = df_lender_cntry.merge(df_loans[['loan_id', 'posted_time']], on='loan_id')[['country_code', 'posted_time']].groupby(['country_code', 'posted_time']).size().reset_index(name='counts')
#df_cntry_cnts.head()

# STEP 4 - let's make life easier with these country codes...
df_cntry_cnts = df_cntry_cnts.merge(df_cntry_codes[['country_code', 'country']], on='country_code', how='left')
df_cntry_cnts['country'] = np.where(df_cntry_cnts['country_code'] == 'SS', 'South Sudan', df_cntry_cnts['country'])  
df_cntry_cnts['country'] = np.where(df_cntry_cnts['country_code'] == 'XK', 'Kosovo', df_cntry_cnts['country'])  

# sum up to country level
df_cntry_sum = df_cntry_cnts.groupby(['country', 'country_code']).sum()
df_cntry_sum.reset_index(level=1, inplace=True)
df_cntry_sum.reset_index(level=0, inplace=True)

# country divided by registered users...  slightly unfairly, counting whales as 0s now...
df_lender_sum = df_lenders.groupby(['country_code']).size().reset_index(name='counts_reg')
df_lender_sum = df_lender_sum.merge(df_cntry_sum, on='country_code')
df_lender_sum['loans_per_lender'] = df_lender_sum['counts'] / df_lender_sum['counts_reg']


# In[ ]:


df_display = df_lender_sum[df_lender_sum['counts_reg'] >= 1000].sort_values('loans_per_lender', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_lender', y='country', data=df_display, color='purple')

ax.set_title('Top 30 Locations by Lender Contribution Count Per Registered User, >= 1,000 Users, Minus Outliers', fontsize=15)
ax.set(ylabel='Country', xlabel='Number of Loan Contributions Per User')
plt.show()


# Some jostling has occurred - with the Swiss still being very big fans!
# <a id=7_2></a>
# ## 7.2. Top Loans Per User Per Country, >= 10,000 Registered Users, Minus Outliers

# In[ ]:


df_display = df_lender_sum[df_lender_sum['counts_reg'] >= 10000].sort_values('loans_per_lender', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_lender', y='country', data=df_display, color='green')

ax.set_title('Top Locations by Lender Contribution Count Per Registered User, >= 10,000 Users, Minus Outliers', fontsize=15)
ax.set(ylabel='Country', xlabel='Number of Loan Contributions Per User')
plt.show()


# The Dutchies are still rocking it - barely edging out Germany.
# <a id=7_3></a>
# ## 7.3 Loans Per User Per State, Minus Outliers

# In[ ]:


# STEP 2 - map users to US states
df_lender_state = df_exp.merge(df_lenders[(df_lenders['country_code'] == 'US') & (df_lenders['whale'].isnull())][['permanent_name', 'state']], on='permanent_name')
df_lender_state.dropna(axis=0, how='any', inplace=True)
#df_lender_cntry

# STEP 3 - merge users to loans and aggregate count by country and day
df_state_cnts = df_lender_state.merge(df_loans[['loan_id', 'posted_time']], on='loan_id')[['state', 'posted_time']].groupby(['state', 'posted_time']).size().reset_index(name='counts')

df_state_sum = df_state_cnts.groupby(['state']).sum()
df_state_sum.reset_index(level=0, inplace=True)

df_lender_sum_state = df_lenders[df_lenders['country_code'] == 'US'].groupby(['state']).size().reset_index(name='counts_reg')
df_lender_sum_state = df_lender_sum_state.merge(df_state_sum, on='state')
df_lender_sum_state['loans_per_lender'] = df_lender_sum_state['counts'] / df_lender_sum_state['counts_reg']

# let's merge in state populations here too
df_state_pop = pd.read_csv("../input/population-by-state/population.csv")
df_state_pop = df_state_pop.rename(index=str, columns={'State': 'state'})
df_lender_sum_state = df_lender_sum_state.merge(df_state_pop, on='state')
df_lender_sum_state['loans_per_capita'] = df_lender_sum_state['counts_reg'] / df_lender_sum_state['Population']


# In[ ]:


# there's one guy in TT... adding in this registered user count to make that outlier disappear though.
df_display = df_lender_sum_state[df_lender_sum_state['counts_reg'] > 10].sort_values('loans_per_lender', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_lender', y='state', data=df_display, color='salmon')

ax.set_title('Top 30 Lender State Locations, Loans Per User', fontsize=15)
ax.set(ylabel='US State', xlabel='Number of Loans Per User')
plt.show()


# Top ranking is similar, but lower.  The outliers are definitely affecting the final numbers, although at the threshold I chose, they are not really swinging rankings too much.  Let's take a look at this intensity on a map.

# In[ ]:


scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_lender_sum_state['state'],
        z = df_lender_sum_state['loans_per_lender'],
        locationmode = 'USA-states',
        #text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "loans per registered user")
        ) ]

layout = dict(
        title = 'Loan Contribution Count Per Registered User',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )


# For comparison and of potential interest, one of many blue/red/purple state images.  Image taken from http://www.electoral-vote.com/evp2012/Pres/Phone/Aug07.html
# ![](http://www.electoral-vote.com/evp2012/Images/map-for-ties.jpg)

# <a id=8></a>
# # 8 Lender Occupations

# In[ ]:


df_lenders['occupation'] = df_lenders['occupation'].str.title()
df_display = df_lenders['occupation'].value_counts().head(30).to_frame()

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='occupation', y=df_display.index, data=df_display, color='aqua')

ax.set_title('Top 30 Occupations - Registered Users', fontsize=15)
ax.set(ylabel='Occupation', xlabel='Number of Registered Lenders')
plt.show()


# In[ ]:



df_display = df_lenders.groupby('occupation')['loan_purchase_num'].sum().to_frame().sort_values('loan_purchase_num', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loan_purchase_num', y=df_display.index, data=df_display, color='darkcyan')

ax.set_title('Top 30 Occupations - Most Loan Contributions', fontsize=15)
ax.set(ylabel='Occupation', xlabel='Number of Loan Contributions')
plt.show()


# It's amusing that the user with the most contributions himself shows up here!  He is a lawyer.  Unsurprising is that retirees come out on top.  Surprising is teachers and students show up so high here - Kiva has probably had a good run as a teaching tool.  What about loans per user for an occupation?
# 

# In[ ]:


df_occ_cnts = df_lenders['occupation'].value_counts().to_frame()
df_occ_cnts.reset_index(level=0, inplace=True)
df_occ_cnts = df_occ_cnts.rename(index=str, columns={'occupation': 'count_reg', 'index' : 'occupation'})

df_occ_loans = df_lenders.groupby('occupation')['loan_purchase_num'].sum().to_frame()
df_occ_loans.reset_index(level=0, inplace=True)
df_occ_loans = df_occ_loans.rename(index=str, columns={'loan_purchase_num': 'count_loans'})

df_occ_loans = df_occ_loans.merge(df_occ_cnts, on='occupation')
df_occ_loans['loans_per_occ'] = df_occ_loans['count_loans'] / df_occ_loans['count_reg']


# In[ ]:


gt_than = 3
df_display = df_occ_loans[df_occ_loans['count_reg'] > gt_than].sort_values('loans_per_occ', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_occ', y='occupation', data=df_display, color='chocolate')

ax.set_title('Top 30 Occupations - Most Loan Contributions Per Occupation User; >' + str(gt_than) + ' Users With Occupation', fontsize=15)
ax.set(ylabel='Occupation', xlabel='Number of Loan Contributions')
plt.show()


# In[ ]:


gt_than = 50
df_display = df_occ_loans[df_occ_loans['count_reg'] > gt_than].sort_values('loans_per_occ', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_occ', y='occupation', data=df_display, color='limegreen')

ax.set_title('Top 30 Occupations - Most Loan Contributions Per Occupation User; >' + str(gt_than) + ' Users With Occupation', fontsize=15)
ax.set(ylabel='Occupation', xlabel='Number of Loan Contributions')
plt.show()


# In[ ]:


gt_than = 500
df_display = df_occ_loans[df_occ_loans['count_reg'] > gt_than].sort_values('loans_per_occ', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_occ', y='occupation', data=df_display, color='cornflowerblue')

ax.set_title('Top 30 Occupations - Most Loan Contributions Per Occupation User; >' + str(gt_than) + ' Users With Occupation', fontsize=15)
ax.set(ylabel='Occupation', xlabel='Number of Loan Contributions')
plt.show()


# In[ ]:


gt_than = 1500
df_display = df_occ_loans[df_occ_loans['count_reg'] > gt_than].sort_values('loans_per_occ', ascending=False).head(30)

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

sns.barplot(x='loans_per_occ', y='occupation', data=df_display, color='mediumaquamarine')

ax.set_title('Top 30 Occupations - Most Loan Contributions Per Occupation User; >' + str(gt_than) + ' Users With Occupation', fontsize=15)
ax.set(ylabel='Occupation', xlabel='Number of Loan Contributions')
plt.show()


# Kiva could use these charts to directly target users in those occupations that appear to be stickier and really enjoy using their platform.
# 
# <a id=9></a>
# # 9 Trends
# <a id=9_1></a>
# ## 9.1 Monthly Active Users
# I don't have web info, or date of contribution info, but as a proxy to roughly make this metric we can count distinct users who made a loan contribution within the month that a loan listing began.

# In[ ]:


df_display = df_exp.merge(df_loans[['loan_id', 'posted_month']], on='loan_id')[['permanent_name', 'posted_month']].groupby(['permanent_name', 'posted_month']).size().reset_index(name='counts').groupby(['posted_month']).size().reset_index(name='counts')

fig, ax = plt.subplots(1, 1, figsize=(20, 8), sharex=True)

plt.plot(df_display['posted_month'], df_display['counts'])
#plt.legend(['LBP', 'USD'], loc='upper left')
ax.set_title('Monthly Active Users (at least one loan contribution within month)', fontsize=15)
plt.show()


# Kiva had some really solid linear growth, blasting all through the US recession and economic output gap.  Unfortunately that growth seems to have plateaued a bit.  What happened?
# <a id=9_2></a>
# ## 9.2 Average and Median Days Since Last Visit
# We are again approximating this - a distinct list of users and loan start dates is created.  Time between the last loan start date is used.  A user who logged in once January 10th who funded a loan that began January 1st, another than began January 5th, and a third that began January 12th would thus show up like this:
# * username,5-Jan-2018,1-Jan-2018,4 days
# * username,12-Jan-2018,5-Jan-2018,7 days
# 
# We can probably do a few things here, but I'm simply going to plot a point for a post_date of a loan a user funded, and the average number of days since the last time the same user funded a loan.

# In[ ]:


#need memory
del df_cntry_sum
del df_lender_sum
del df_lender_cntry
del df_cntry_cnts
del df_lender_state
del df_state_cnts
del df_state_sum
del df_lender_sum_state
gc.collect()


# In[ ]:


# merge exploaded users to loans to get posted times
df_display = df_exp.merge(df_loans[['loan_id', 'posted_time']], on='loan_id')[['permanent_name', 'posted_time']].drop_duplicates().sort_values(['permanent_name', 'posted_time'])

# get a distinct list of names and loan start dates
#df_last_visit = df_loans_users[(df_loans_users['permanent_name'] == 'sam4326') | (df_loans_users['permanent_name'] == 'rebecca3499')][['permanent_name', 'posted_time']].drop_duplicates().sort_values(['permanent_name', 'posted_time'])
#df_display = df_loans_users[['permanent_name', 'posted_time']].drop_duplicates() #.sort_values(['permanent_name', 'posted_time'])

# get the prior loan date for user
df_display['prev_loan_dt'] = df_display.groupby('permanent_name')['posted_time'].shift()

df_display.dropna(axis=0, how='any', inplace=True)

# calc days different
df_display['date_diff'] = (df_display['posted_time'] - df_display['prev_loan_dt']).dt.days


# In[ ]:


df_disp = df_display.groupby('posted_time')['date_diff'].mean().to_frame()
df_disp.reset_index(level=0, inplace=True)

df_disp = df_disp[df_disp['posted_time'] <= '2017-12-25']

fig, ax = plt.subplots(1, 1, figsize=(20, 8), sharex=True)

plt.plot(df_disp['posted_time'], df_disp['date_diff'])

ax.set_title('Average Days Since Last Loan Visit', fontsize=15)
plt.show()


# In[ ]:


df_disp = df_display.groupby('posted_time')['date_diff'].median().to_frame()
df_disp.reset_index(level=0, inplace=True)

df_disp = df_disp[df_disp['posted_time'] <= '2017-12-25']

fig, ax = plt.subplots(1, 1, figsize=(20, 8), sharex=True)

plt.plot(df_disp['posted_time'], df_disp['date_diff'])

ax.set_title('Median Days Since Last Loan Visit', fontsize=15)
plt.show()


# There's probably a better way to do this methodology, but we can still look at the trend here; which looks reasonably stable, with the average user coming back to make a loan every 2 or 3 months.  Even nicer to see the median is pretty stable and much lower.
# <a id=9_3></a>
# ## 9.3 Unfunded Loan Gap

# In[ ]:


df_display = df_loans[df_loans['status'] != 'fundRaising'].groupby('posted_time')[['funded_amount', 'loan_amount']].sum()
df_display.reset_index(level=0, inplace=True)
df_display['posted_year'] = df_display['posted_time'].dt.year
df_display['gap_amount'] = df_display['loan_amount'] - df_display['funded_amount']
df_display['day_of_year'] = df_display['posted_time'].dt.dayofyear
df_display['month_of_year'] = df_display['posted_time'].dt.month


# In[ ]:


fig, lst = plt.subplots(4, 1, figsize=(20, 12), sharex=False)
j = 2014

for i in lst:

    i.plot(df_display[df_display['posted_year'] == j]['posted_time'], df_display[df_display['posted_year'] == j]['loan_amount'], color='#67c5cb', label='loan_amount')
    i.plot(df_display[df_display['posted_year'] == j]['posted_time'], df_display[df_display['posted_year'] == j]['funded_amount'], color='#cb6d67', label='funded_amount')
    i.plot(df_display[df_display['posted_year'] == j]['posted_time'], df_display[df_display['posted_year'] == j]['gap_amount'], color='salmon', label='gap_amount')
    j = j+1

lst[0].set_title('Funding Gap By Day; 2014-2017', fontsize=15)
lst[0].legend(loc='upper left', frameon=True)
    
plt.show()


# Above we have funded vs. requested amounts; and the subsequent gaps.  The amounts don't seem to include overall much of a seasonal or cyclical trend, although the gaps seem like they might.

# In[ ]:


df_disp = df_display.groupby('day_of_year')['gap_amount'].agg(['sum', 'count']).reset_index()
df_disp2 = df_display.groupby('month_of_year')['gap_amount'].agg(['sum', 'count']).reset_index()
df_disp2['gap_per_loan'] = df_disp2['sum'] / df_disp2['count']


# In[ ]:


fig, (ax1, ax2, ax5) = plt.subplots(3, 1, figsize=(20, 14), sharex=False)

ax1.plot(df_disp['day_of_year'], df_disp['sum'], color='salmon')
ax1.set_title('Funding Gap 2014-2017 by Day of Year', fontsize=15)
#ax3 = ax1.twinx()
#ax3.plot(df_disp['day_of_year'], df_disp['count'], color='green')

ax2.plot(df_disp2['month_of_year'], df_disp2['sum'], color='salmon', label='funding gap')
ax2.set_title('Funding Gap 2014-2017 by Month of Year vs. New Loan Requests', fontsize=15)
ax4 = ax2.twinx()
ax4.plot(df_disp2['month_of_year'], df_disp2['count'], color='darkblue', label='\nnew loan requests')
ax2.legend(loc='upper left', frameon=True)
leg = ax4.legend(loc='upper left', frameon=True)
leg.get_frame().set_alpha(0)

ax5.plot(df_disp2['month_of_year'], df_disp2['gap_per_loan'], color='c')
ax5.set_title('Funding Gap Per Loan, Aggregate 2014-2017 by Month of Year', fontsize=15)

plt.show()


# Our funding gap appears to be seasonal.  This does not appear to be a result of the amount requested for loans nor the number of loans requested; we clearly see a significant growth in April ownwards.  Given that the US is the primary loan contributor, I think what we're seeing here is in part simply due to the weather.  By June, no one is shoveling snow then keeping warm by their computer lending on Kiva; they're out at the beach.  Other countries are experiencing similar winter timeframes that come after the US as well.  As it gets cooler we see the funding gap shrinking; even though the loan requests have increased and Kiva may be additionally competing with holiday spending dollars.  Kiva might be able to smooth out this bump by attracting lenders in a part of the world that is experiencing its winter while the US is enjoying its summer.
# <a id=10></a>
# # 10. Recommendations
# 1. What is going on with the plateau reached by Kiva registered user growth hit in 2014?  It would be interesting to see what happened there, if Kiva has any indication of changes.
# 2. A lot of Kiva users register and make 0-1 loans.  In fact, simply making 2 loans already puts a user at the 40th percentile of lenders.  Kiva needs to figure out a good way to convert a registered user into a sticky user.  This might be increasing the frequency of contacts or perhaps the quality of them, to encourage users with credit to re-lend and get interested in adding additional risk-tolerant capital.
# 3. Kiva should market to those users it already knows tend to be stickier and like its platform and mission - those in the top occupations charts we saw above.  Additionally retirees and Christians are among its more loyal users.
# 4. Kiva's funding gap appears to grow the largest in the US summer months - it would be nice to fill it with some additional Southern Hemisphere lenders, or those around the Equator.  Australia already has more than 10,000 registered Kia users and a pretty strong economy; spreading the message and acquiring more users there might be perfect for reducing the gap.
# 5. It's a weird feature of a US "travel hacking" hobby, but some people meet credit card minimum spends for bonus by lending out to people on Kiva.  My own set of loans is in part due to that.  I don't think the effect can be denied, as by number of loans, I found at least the accounts ranked 45, 126, and 168 were all related to this, related to a travel hacking blog which I have read.  These 3 accounts alone have 24,546 loans to their name.  Kiva could market to travelers or more targeted, this niche of credit card and points travel hackers.  The [4th largest contributing Kiva group](https://www.kiva.org/team/insideflyer) is a travel and miles group.
# 6. Kiva should do something more with public leaderboards and gamifying the lending process.  Right now Kiva has lending groups, which are used in varying degrees to get like minded people to lend together.  These are based on religions, political views, companies,  hobbies (like the flyers mentioned above), etc.  When a loan is made, it can be credited towards a group.  So there is a fun informal gamification already going on, although I think Kiva could do more in providing public lists and leaderboards to gamify various demographics.  This could be most loans to country x; or for sector y.  Most loans to women, or groups; most loans from a state; etc etc.  Perhaps state isn't just individuals within a state, but a leaderboard for total state contributions or contributions per capita.  I think some friendly gamification will help encourage some stickiness in lenders, as well as result in some of those atop the leaderboards lending more, as they are able to see what it takes to climb up just a bit more...

# In[ ]:




