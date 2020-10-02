#!/usr/bin/env python
# coding: utf-8

# # Ontario COVID-19 Duration Descriptive Stats
# 
# This notebook consumes the Ontario COVID-19 case-level data release (https://data.ontario.ca/dataset/confirmed-positive-cases-of-covid-19-in-ontario/resource/455fd63b-603d-4608-8216-7d8647f43350) and describes the length of time it takes for various activities to occur. Ideally, all activities happen as rapidly as possible, leading to good visibility of COVID-19 cases and prompt medical care for those requiring it.
# 
# The "patient journey" is measured using four dates:
# * Active_Episode_Date -- the best estimate of when the onset of symptoms occured
# * Specimen_Date -- the date that a specimen was collected from the patient for testing
# * Test_Reported_Date -- the date on which the test result was reported by the laboratory
# * Case_Reported_Date -- the date that the confirmed-positive case was logged by the Public Health Unit into the iPHIS database
# 
# In some cases, steps may have occurred out-of-order, particularly reporting of tests and cases. It is not entirely clear why, but you will see this in the statistics.
# 
# We measure four durations, defined as the difference between the dates above:
# * Episode-to-Report -- the time from (estimated) onset of symptoms to reporting in iPHIS.  This is the end-to-end measure and the most relevant for measuring surveillance performance
# * Episode-to-Specimen -- how long did it take the patient to arrive at an assessment centre and have a specimen collected?  This measures a mix of patient behaviour and assessment centre availability.
# * Specimen-to-Result -- how long between the collection of the specimen and completed processing by the lab?  This measures the test-processing workflow.
# * Result-to-Report -- how long between a positive lab result and a report logged in iPHIS?  This would ideally be zero.

# In[ ]:


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[ ]:


DATE_FIELDS = ['Accurate_Episode_Date','Case_Reported_Date','Test_Reported_Date','Specimen_Date']


# In[ ]:


latest = pd.read_csv('https://data.ontario.ca/dataset/f4112442-bdc8-45d2-be3c-12efae72fb27/resource/455fd63b-603d-4608-8216-7d8647f43350/download/conposcovidloc.csv',
                    parse_dates=DATE_FIELDS,
                    )
latest.head()


# In[ ]:


percentiles = [50,80,90,95,100]
metrics = ['Episode_to_Report', 'Episode_to_Specimen', 'Specimen_to_Result', 'Result_to_Report']
combo_metrics = ['%s_%d' % (m, p) for m in metrics for p in percentiles]


# In[ ]:


latest['Episode_to_Report'] = (latest['Case_Reported_Date'] - latest['Accurate_Episode_Date']).dt.days
latest['Episode_to_Specimen'] = (latest['Specimen_Date'] - latest['Accurate_Episode_Date']).dt.days
latest['Specimen_to_Result'] = (latest['Test_Reported_Date'] - latest['Specimen_Date']).dt.days
latest['Result_to_Report'] = (latest['Case_Reported_Date'] - latest['Test_Reported_Date']).dt.days


# In[ ]:


latest[metrics].describe()


# ## Outliers
# 
# We definitely have some outliers.  Let's take a look at a few of the extremes.

# In[ ]:


latest[latest['Result_to_Report']==-83]


# Not clear what's happening with this one.  Is this a re-test?

# In[ ]:


latest[latest['Episode_to_Specimen']==-89]


# This looks like a data keying error.  5 and 2 are adjacent on the numeric pad, and the day part of the date would otherwise make sense.

# In[ ]:


latest[latest['Specimen_to_Result']==-30]


# Data entry error seems likely.  Test Reported = 03-31 would make sense.

# Overall the outliers are probably not worrisome, as we'll be looking at things by percentile.

# ## Missing Data?
# 
# How many missing dates do we have?

# In[ ]:


for m in metrics:
    print('There are %d cases of missing %s' % (len(latest[np.isnan(latest[m])]), m))
print('... out of %d total cases' % len(latest))


# # Computation and Display of Percentiles
# 
# Let's break down those metrics into percentiles, treating each day as its own cohort.  This will let use compare the trends from day to day.
# 
# We will look **starting March 1, 2020** as the handling of the pre-outbreak stage of the pandemic is not of much interest to us now.
# 
# NB: this calculates the percentiles out of the non-missing data.  We'll handle the computation taking into account the missing data below.

# In[ ]:


latest_date = latest[DATE_FIELDS].max().max()
latest_date


# In[ ]:


delay_df = pd.DataFrame(index=pd.date_range('2020-03-01', latest_date), columns=combo_metrics)

for crd, grp in latest[latest['Accurate_Episode_Date']>=pd.to_datetime('2020-03-01')].groupby('Case_Reported_Date'):
    for m in metrics:
        for p in percentiles:
            delay_df.loc[crd, '%s_%d' % (m, p)] = grp[m].quantile(p/100)
delay_df.tail()


# In[ ]:


fig, axarr = plt.subplots(4, figsize=(6, 12))
for i, m in enumerate(metrics):
    ax = axarr[i]
    for p in percentiles:
        delay_df.plot(y='%s_%d' % (m, p), label='%dth percentile' % p, ax=ax)
        ax.set_ylabel(m)
    ax.set_xlabel('Case_Reported_Date')


# The fact that the maximum-line is increasing linearly in the top plot indicates that cases from the beginning of the pandemic are still routinely being logged.  This is indicative of the poor data engineering at the start of the pandemic.

# # Break Down by Public Health Unit
# 
# Let's see how the PHUs compare.

# In[ ]:


latest[latest['Accurate_Episode_Date']>=pd.to_datetime('2020-03-01')].groupby('Reporting_PHU')[metrics].quantile([0.5, 0.9, 0.95, 1.0]).unstack()


# # Accounting for Missing Data
# 
# We will use a sentinel date far in the future to replace the missing data.  We will then calculate percentiles and the far-future date will force all missing data to show up at the "top" of the percentile rank-ordered list.  If we get a sentinel date back from a percentile calculation, we will then replace it with `NaN`, indicating that we do not yet known the value of that percentile.

# In[ ]:


sentinel_date = pd.Timestamp.max
sentinel_date


# If we're still dealing with COVID-19 in 242 years in Ontario, we deserve our fate.  Also, someone will need to update this code.

# In[ ]:


latest_sentinel = latest.copy()
for d_f in DATE_FIELDS:
    latest_sentinel[d_f] = latest_sentinel[d_f].fillna(sentinel_date)


# In[ ]:


latest_sentinel['Episode_to_Report'] = (latest_sentinel['Case_Reported_Date'] - latest_sentinel['Accurate_Episode_Date']).dt.days
latest_sentinel['Episode_to_Specimen'] = (latest_sentinel['Specimen_Date'] - latest_sentinel['Accurate_Episode_Date']).dt.days
latest_sentinel['Specimen_to_Result'] = (latest_sentinel['Test_Reported_Date'] - latest_sentinel['Specimen_Date']).dt.days
latest_sentinel['Result_to_Report'] = (latest_sentinel['Case_Reported_Date'] - latest_sentinel['Test_Reported_Date']).dt.days


# In[ ]:


sentinel_delay_df = pd.DataFrame(index=pd.date_range('2020-03-01', latest_date), columns=combo_metrics)

for crd, grp in latest_sentinel[latest_sentinel['Accurate_Episode_Date']>=pd.to_datetime('2020-03-01')].groupby('Case_Reported_Date'):
    for m in metrics:
        for p in percentiles:
            sentinel_delay_df.loc[crd, '%s_%d' % (m, p)] = grp[m].quantile(p/100)
sentinel_delay_df.tail()


# In[ ]:


fig, axarr = plt.subplots(4, figsize=(6, 12))
for i, m in enumerate(metrics):
    ax = axarr[i]
    for p in percentiles:
        sentinel_delay_df.plot(y='%s_%d' % (m, p), label='%dth percentile' % p, ax=ax)
        ax.set_ylabel(m)
    ax.set_xlabel('Case_Reported_Date')


# Good!  Our missing data is showing up!
# 
# Now let's correct for it.

# In[ ]:


def correct_sentinel(val_in_days):
    # if it's off by more than 2 years, it's due to missing data
    if (val_in_days > 730) or (val_in_days < -730):
        return np.nan
    else:
        return val_in_days


# In[ ]:


for cm in combo_metrics:
    sentinel_delay_df[cm] = sentinel_delay_df[cm].apply(correct_sentinel)


# In[ ]:


fig, axarr = plt.subplots(4, figsize=(6, 12))
for i, m in enumerate(metrics):
    ax = axarr[i]
    for p in percentiles:
        sentinel_delay_df.plot(y='%s_%d' % (m, p), label='%dth percentile' % p, ax=ax)
        ax.set_ylabel(m)
    ax.set_xlabel('Case_Reported_Date')


# In[ ]:


latest_sentinel[latest_sentinel['Accurate_Episode_Date']>=pd.to_datetime('2020-03-01')].groupby('Reporting_PHU')[metrics].quantile([0.5, 0.9, 0.95, 1.0]).unstack().applymap(correct_sentinel)


# This is interesting -- `NaN` indicates that some of the dates are missing, and you can clearly see where.

# ## Kaggle Boilerplate Below

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:




