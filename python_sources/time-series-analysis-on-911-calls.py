#!/usr/bin/env python
# coding: utf-8

# <font size='6'><b> Time Series Analysis on 911 Calls in Baltimore </b></font><br>
# <font size='4'>In the previous study, it was analyzed if police violence led to a decrease in calls to 911 in cities with black majority, such as Milwaukee. I ran a similar analysis on Baltimore city calls to observe if there was a decrease in number of calls.</font>
# 

# In[ ]:


from scipy.stats import ttest_ind as ttest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# We define some helper functions first, which will be used to process the data later on.

# In[ ]:


def _filter_priority(calls,prty):
    if prty == 'All':
        return calls
    return calls[calls['priority']==prty]


# In[ ]:


# This function counts the number of calls
# @param pandas Series of calls
# @param desired year
# @return pandas Series of call count per month
def count_it(calls, yr, prty, weekly=True):
    call_list = _filter_priority(calls,prty)
    date_series = pd.to_datetime(call_list['callDateTime'])

    if weekly:
        date_series = date_series[date_series.dt.year == yr]
        return date_series.dt.week.value_counts()

    if yr == 2017: mths = 7
    else: mths = 12

    counts = pd.Series(np.zeros(mths), index=range(1,mths+1))

    for mth in range(1,mths+1):
        counts.set_value(label = mth, value = date_series[(date_series.dt.year == yr) &
                                           (date_series.dt.month == mth)].count())

    return counts


# In[ ]:


# This function calculates the median number of calls per week
# @param pandas Series of calls
# @param desired priority or all calls
# @return pandas Series of median number of calls per week
def weekly_medians(calls,prty):
    call_list = _filter_priority(calls,prty)

    return pd.Series([count_it(call_list, 2015, prty).median(),
                      count_it(call_list, 2016, prty).median(),
                      count_it(call_list, 2017, prty).median()],
                     index=[2015,2016,2017],
                     name=prty)


# In[ ]:


# This function runs ttest on weekly number of calls
# @param pandas Series of calls
# @param desired year
# @return pandas Series of call count per month
def run_ttest(calls, prty):
    call_list = _filter_priority(calls,prty)

    return pd.Series([ttest(count_it(call_list,2015, prty),count_it(call_list,2016, prty))[1],
                      ttest(count_it(call_list, 2016, prty), count_it(call_list, 2017, prty))[1]],
                     index=[2016,2017],
                     name=prty)


# In[ ]:


# Load dataset into the memory
all_calls = pd.DataFrame(pd.read_csv('../input/911_calls_for_service.csv',index_col=0))


# In[ ]:


# Plot number of calls per month per priority group to see the seasonality of calls
call_types = pd.DataFrame([['All','Medium'],['High','Low']])

fig, axarr = plt.subplots(2,2)

for i in range(2):
    for j in range(2):
        for yr in [2015,2016,2017]:
            # print count_it(all_calls,yr)
            if yr == 2017:
                axarr[i,j].plot(pd.Series(range(1,8)),
                        count_it(all_calls,yr,prty=call_types[i][j],weekly=False),
                        label = yr,marker = '.')
            else:
                axarr[i,j].plot(pd.Series(range(1,13)),
                        count_it(all_calls,yr,prty=call_types[i][j],weekly=False),
                        label = yr,marker = '.')

            # if call_types[i][j] == 'Low': axarr[i,j].legend(loc='upper right')
            axarr[i,j].set_xlim(xmin=1, xmax=12)
            axarr[i,j].set_title(call_types[i][j])

axarr[1,1].legend(loc='upper right')
plt.setp([a.set_xticks(range(1,13,2)) for a in axarr[1, :]])
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
plt.show()


# 911 receives less calls at the beginning and end of the year. Second and third quarter of the year seems to be the busiest seasons for 911. Also the graph suggests that there might be misclassification of calls in the first two months of 2015, High priority calls being recorded as Low priority.

# In[ ]:


# Calculate weekly median number of calls and run t-test
df_weekly_medians = pd.DataFrame()
df_weekly_calls_ttest = pd.DataFrame()

for priority in ['All','High','Medium','Low']:
    df_weekly_medians.insert(0,priority,value = weekly_medians(all_calls,priority))
    df_weekly_calls_ttest.insert(0,priority,value = run_ttest(all_calls,priority))


# In[ ]:


# Plot weekly median calls per year
ax = plt.subplot()
df_weekly_medians.iloc[0].plot.bar(width=0.25, color='r',position=-0.5)
df_weekly_medians.iloc[1].plot.bar(width=0.25, color='b',position=0.5)
df_weekly_medians.iloc[2].plot.bar(width=0.25, color='g',position=1.5)
ax.autoscale(tight=True)
ax.legend(loc='upper left')
plt.xticks(rotation='horizontal')
ax.set_title('Weekly Median Number of Calls per Year',fontsize = 20)
ax.set_xlabel('Call Priority')
ax.set_ylabel('Median Number of Calls')
plt.show()


# When we look at the weekly median number of calls, we observe slight decreases in all categories but 'High' priority. There seems to be slight increases in High priority calls. However, when we check the p-values from t-test, we only see statistically significant difference in Low priority calls (p-value less than 0.05). 

# In[ ]:


# Check t-test results to see if the change in medians are statistically significant
df_weekly_calls_ttest


# According to US Census Bureau, population of Baltimore was estimated to drop to 614,664 in 2016 from the 2010 figure of 621,195 which is approximately 0.18% decrease annually. This figure is much smaller than observed decreases.

# Further analysis can be made on 'Description' of Low priority calls. More information on reasons for which people are calling 911 less can be discovered there. Coordinate-based clustering of calls can also reveal interesting insight.
