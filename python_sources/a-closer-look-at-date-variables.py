#!/usr/bin/env python
# coding: utf-8

# # Do points in time matter?
# 
# In this kernel we will take a closer look at all variables containing dates or timestamps and their relationship with the target variable. We will search for seasonal patterns, weekly patterns and more.
# 
# The historical data will not be used in this example. It's only of exploratory nature and does not contain any models whatsoever. It's only supposed to showcase some different possibilities for using these variables in your own models.

# In[ ]:


import warnings
import datetime
import calendar
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import time
from dateutil.relativedelta import relativedelta

# to ignore future warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

# set size of seaborn plots
sns.set(rc={'figure.figsize':(10, 7)})


# In[ ]:


## read data
train = pd.read_csv('../input/train.csv', sep = ',')
test = pd.read_csv('../input/test.csv', sep = ',')
merchants = pd.read_csv('../input/merchants.csv', sep = ',')
new_merchant = pd.read_csv('../input/new_merchant_transactions.csv', sep = ',')


# Before we can start we obviously have to merge the data.

# In[ ]:


## prepare data
# this is not a valid approach if you want to build models from the data
# drop some redundant columns
dropping = ['merchant_category_id', 'subsector_id', 'category_1', 'city_id', 'state_id',
            'category_2']
for var in dropping:
    merchants = merchants.drop(var, axis = 1)

# merge merchants with new_merchants
data = pd.merge(merchants, new_merchant, on = 'merchant_id')

# merge data with train data
data = pd.merge(data, train, on = 'card_id')


# The variables in question here are *first_active_month* and *purchase_time*. Let's take a brief look at their number of unique values and the first five values:

# In[ ]:


print(len(data['first_active_month'].unique()))
data['first_active_month'][:5]


# In[ ]:


print(len(data['purchase_date'].unique()))
data['purchase_date'][:5]


# This shows us that the *purchase_date* variable actually carrys more information than it's name makes it sound like. It's not only the date, but also the specific time. Let's recode this into two variables:

# In[ ]:


# recode purchase_date
data['purchase_time'] = data['purchase_date'].str.split(' ')
data['purchase_date'] = data['purchase_time'].str[0]
data['purchase_time'] = data['purchase_time'].str[1]


# Now there are two different strategies to use these time variables in further models.
# 
# 1. Recode them to a linear variable where the lowest number is the day furthest in the past and the highest number the most recent day
# 2. Recode them to ordered categorical variables
# 
# Let's start with number 1:
# 

# In[ ]:


def dates_to_numeric(series, kind = 'month'):
    # get all unique values
    months = list(series.unique())

    # sort them
    if kind == 'month':
        date_string = "%Y-%m"
    elif kind == 'day':
        date_string = "%Y-%m-%d"

    # make them a datetime object
    dates = [datetime.datetime.strptime(ts, date_string) for ts in months]
    dates.sort()
    sorteddates = [datetime.datetime.strftime(ts, date_string) for ts in dates]

    # generate all month stamps between first and last
    start_date = sorteddates[0]
    end_date = sorteddates[len(sorteddates) - 1]
    
    cur_date = start = datetime.datetime.strptime(start_date, date_string).date()
    end = datetime.datetime.strptime(end_date, date_string).date()

    months = []
    while cur_date < end:
        if kind == 'month':
            months.append(str(cur_date)[:-3])
            cur_date += relativedelta(months = 1)
        elif kind == 'day':
            months.append(str(cur_date))
            cur_date += relativedelta(days = 1)
    
    # create dict that maps new values to each month
    map_dict = {}
    keys = range(0, len(months))
    for i in keys:
        map_dict[i] = months[i]

    # reverse dict keys / values for mapping
    new_dict = {v: k for k, v in map_dict.items()}
    return new_dict

new_dict = dates_to_numeric(data['first_active_month'])
data['first_active_month_numeric'] = data['first_active_month'].apply(lambda x: new_dict.get(x))

new_dict = dates_to_numeric(data['purchase_date'], kind = 'day')
data['purchase_date_numeric'] = data['purchase_date'].apply(lambda x: new_dict.get(x))

# recode timestamp to number of seconds passed since 00:00:00
def timestamp_to_seconds(time):
    seconds = sum(x * int(t) for x, t in zip([3600, 60, 1], time.split(':'))) 
    return seconds

data['purchase_seconds'] = data['purchase_time'].apply(lambda x: timestamp_to_seconds(x))


# In[ ]:


ax = sns.regplot(x = data['first_active_month_numeric'], y = data['target'], marker = "+",
                 lowess = True, line_kws = {'color': 'black'})
ax.set_title('Relationship of the target variable and linear first active month')
ax.set_xlabel('first active month linear')


# In[ ]:


ax = sns.regplot(x = data['purchase_date_numeric'], y = data['target'], marker = "+",
                 lowess = True, line_kws = {'color': 'black'})
ax.set_title('Relationship of the target variable and linear purchase date')
ax.set_xlabel('purchase date linear')


# In[ ]:


#ax = sns.regplot(x = data['purchase_seconds'], y = data['target'], marker = "+",
#                 lowess = True, line_kws = {'color': 'black'})
#ax.set_xlabel('purchase seconds linear')
# This takes incredibly long and is therefore commented out. It however looks very similar to the plot above


# These plot show no relationship between the metric versions of the three date variables and the target variable. By using the lowess smoother instead of a linear regression we also made sure there is no nonlinear relationship between the two variables.
# 
# Now that this is out of the way, let's dive into the more reasonable data transformations. For the *purchase_date* variable we will create a new variable that contains the name of the corresponding weekday (e.g. monday, tuesday ..). For the *purchase_time* variable we will create a new categorical variable with 4 categories: Morning, Afternoon, Evening and Night. These correspond to:
# 
# **Morning**: 5am to 12pm (05:00 to 11:59)
# 
# **Afternoon**: 12pm to 5pm (12:00 to 16:59)
# 
# **Evening**: 5pm to 9pm (17:00 to 20:59)
# 
# **Night**: 9pm to 5am (21:00 to 04:59)
# 
# We will also create two variables containing the corresponding month (e.g. January, February ..) for *purchase_date* and *first_active_month*. For the latter we will create a *first_active_year* variable as well. **Let's start!
# 
# **EDIT:** I forgot that the time of the month itself (beginning, end etc.) may have an effect on the target variable as well. This will be explored too in this second version. We will look both at a categorical version and a numeric version just using the day.

# In[ ]:


def get_weekday(date_string):
    date = datetime.datetime.strptime(date_string, '%Y-%m-%d')
    return calendar.day_name[date.weekday()]

# get weekday for date variable
data['purchase_weekday'] = data['purchase_date'].apply(lambda x: get_weekday(x))

# for plotting recode to ordered categorical
day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
data['purchase_weekday'] = pd.Categorical(data['purchase_weekday'], categories = day_labels, 
                                          ordered = True)

def get_month(date_string, kind = 'month'):
    if kind == 'month':
        date = datetime.datetime.strptime(date_string, '%Y-%m')
    elif kind == 'day':
        date = datetime.datetime.strptime(date_string, '%Y-%m-%d')
    return date.strftime("%B")

data['purchase_month'] = data['purchase_date'].apply(lambda x: get_month(x, kind = 'day'))
data['first_active_month2'] = data['first_active_month'].apply(lambda x: get_month(x))
data['first_active_year'] = data['first_active_month'].str[:4]

month_labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                'September', 'October', 'November', 'December']
data['purchase_month'] = pd.Categorical(data['purchase_month'], categories = month_labels, 
                                          ordered = True)
data['first_active_month2'] = pd.Categorical(data['first_active_month2'], categories = month_labels, 
                                          ordered = True)

year_labels = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']
data['first_active_year'] = pd.Categorical(data['first_active_year'], categories = year_labels, 
                                          ordered = True)

# get time of the day
data['temp'] = data['purchase_time'].str.split(':')

def get_session(time_list):
    time_list[0] = int(time_list[0])
    if time_list[0] > 4 and time_list[0] < 12:
        return 'Morning'
    elif time_list[0] >= 12 and time_list[0] < 17:
        return 'Afternoon'
    elif time_list[0] >= 17 and time_list[0] < 21:
        return 'Evening'
    else:
        return 'Night'
    
data['purchase_session'] = data['temp'].apply(lambda x: get_session(x))

session_labels = ['Morning', 'Afternoon', 'Evening', 'Night']
data['purchase_session'] = pd.Categorical(data['purchase_session'], categories = session_labels, 
                                          ordered = True)


# In[ ]:


## time of month
# as categorical variable, thressholds are arbitrary and could be different
def get_time_of_month_cat(date):
    date_temp = date.split('-')
    if int(date_temp[2]) < 10:
        time_of_month = 'Beginning'
    elif int(date_temp[2]) >= 10 and int(date_temp[2]) < 20:
        time_of_month = 'Middle'
    else:
        time_of_month = 'End'
    return time_of_month

data['time_of_month_cat'] = data['purchase_date'].apply(lambda x: get_time_of_month_cat(x))

tof_labels = ['Beginning', 'Middle', 'End']
data['time_of_month_cat'] = pd.Categorical(data['time_of_month_cat'], categories = tof_labels, 
                                           ordered = True)

data['time_of_month_num'] = data['purchase_date'].str[8:].astype(int)


# In[ ]:


ax = sns.lineplot(x = "purchase_weekday", y = "target", 
                  markers = True, dashes = False, data = data)
plt.xticks(rotation = 45)
ax.set_title('Target Variable Changes over Purchase Week')
ax.set_xlabel('Purchase Weekday')


# There is a pattern here. The target variable follows a non-linear curve over the weekdays. The differences may be small, but they are statistically significant.

# In[ ]:


ax = sns.lineplot(x = "purchase_month", y = "target", 
                  markers = True, dashes = False, data = data)
plt.xticks(rotation = 45)
ax.set_title('Target Variable Changes over Purchase Month')
ax.set_xlabel('Purchase Month')


# Now this is more like it! There are pretty big differences in the mean of the target variable between each month of purchase. In January the mean is close to - 2 while in April it's at - 0,25. Adding the purchase month as dummy variables to your model looks promising. 

# In[ ]:


ax = sns.lineplot(x = "first_active_month2", y = "target", 
                  markers = True, dashes = False, data = data)
plt.xticks(rotation = 45)
ax.set_title('Target Variable Changes over the First Active Month')
ax.set_xlabel('First Active Month')


# There are some clear variations in this plot as well, which follow almost a step-wise linear pattern. The differences might not be nearly as strong as in the purchase month plot, but this still might help improve your model.

# In[ ]:


ax = sns.lineplot(x = "first_active_year", y = "target", 
                  markers = True, dashes = False, data = data)
plt.xticks(rotation = 45)
ax.set_title('Target Variable Changes over the First Active Year')
ax.set_xlabel('First Active Year')


# The first active year show some big differences! The target variable increases with each year. Definitly an interesting pattern.

# In[ ]:


ax = sns.lineplot(x = "purchase_session", y = "target", 
                  markers = True, dashes = False, data = data)
plt.xticks(rotation = 45)
ax.set_title('Target Variable Changes over Purchase Time of Day')
ax.set_xlabel('Purchase Time of Day')


# Differences over the day are rather small, but follow a clear pattern as well. Let's see if that pattern holds up if we look at it by different week days:

# In[ ]:


ax = sns.catplot(x = 'purchase_weekday', y = 'target', hue = 'purchase_session', data = data,
                kind = 'bar', height = 5, aspect = 2)
ax.despine(left = True)
plt.xticks(rotation = 45)
ax.set_ylabels("target")
ax.set_xlabels('Weekday')


# It seems like it does. The pattern seems to be nearly the same on all weekdays. There are some differences between saturday and tuesday though.
# 
# All in all there are some interesting relationships in the date variables of this data set. I hope this little eploratory insights helped some of you in some way. I am a beginner so I am very thankful for any advice / comments you have to offer.

# **NEW** Following are the new parts of the analysis:

# In[ ]:


ax = sns.regplot(x = data['time_of_month_num'], y = data['target'], marker = "+",
                 lowess = True, line_kws = {'color': 'black'})
ax.set_title('Relationship of the target variable and purchase time of month')
ax.set_xlabel('time of purchase inside month')


# Using just the day itself doesn't seem very useful. No direct relationship can be observed. Now let's look at the categorical version:

# In[ ]:


ax = sns.lineplot(x = "time_of_month_cat", y = "target", 
                  markers = True, dashes = False, data = data)
plt.xticks(rotation = 45)
ax.set_title('Target Variable Changes over Purchase Time of Month')
ax.set_xlabel('Purchase Time of Month')


# The pattern is really small with the biggest deviation being between -0.52 and -0.66, but it does exist. I am not sure if this will be particularly useful but it's definitly worth a try. Let's also look if this pattern is the same in each month:

# In[ ]:


ax = sns.catplot(x = 'purchase_month', y = 'target', hue = 'time_of_month_cat', data = data,
                kind = 'bar', height = 5, aspect = 2)
ax.despine(left = True)
plt.xticks(rotation = 45)
ax.set_ylabels("Target")
ax.set_xlabels('Purchase Time of Month')


# The pattern obviously isn't really stable at all. The only consistent finding is, that the end of the month seemingly always has a higher score on the target variable then other parts. If only the very end of the month has an effect on the target variable, we might still be able to utilize this. Let's try by creating a dummy variable that only looks at the very last days of the month:

# In[ ]:


def get_end_of_month(date):
    date_temp = date.split('-')
    if int(date_temp[2]) >= 25:
        end_of_month = 'Yes'
    else:
        end_of_month = 'No'
    return end_of_month

data['end_of_month'] = data['purchase_date'].apply(lambda x: get_end_of_month(x))

ax = sns.barplot(x = 'end_of_month', y = 'target', data = data)


# This shows bigger differences than before. Only trying it out will show if this is usefull in prediction or not though. Good luck!
