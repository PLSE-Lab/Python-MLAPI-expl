#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import json
import gc
from collections import defaultdict
from math import sin, radians
import os
import sys


# In[ ]:


sys.path.append(("../input/helper-scripts"))
from chart_helper import bar_chart, line_chart, value_counts_barchart, value_counts_linechart, merged_tail_barchart, merged_tail_linechart
import util_helper as utils


# In[ ]:


COUNT_THRESHOLD = 30


# In[ ]:


MERGE_THRESHOLD = 0.001


# In[ ]:


pd.set_option('display.max_columns', 100)


# ### Preprocess Data

# First we will unpack the JSON columns in the data. This will increase the number of columns significantly. 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Load and parse JSON\n# Since parsing takes > 1 min, we will keep a back up copy for reruns\n\n# Load backup copy if it exists \ntry:\n    train_data = train_back_up.copy()\n\n# If it does not\nexcept NameError:\n    # Load data from file\n    train_data = utils.load_data(path="../input/ga-customer-revenue-prediction/train.csv")\n    \n    # Parse JSON columns in data\n    train_data = utils.parse_data(data=train_data)\n    \n    # Create a back up copy, for re-run\n    train_back_up = train_data.copy()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Load and parse JSON\n# Since parsing takes > 1 min, we will keep a back up copy for reruns\n\n# Load backup copy if it exists \ntry:\n    test_data = test_back_up.copy()\n\n# If it does not\nexcept NameError:\n    # Load data from file\n    test_data = utils.load_data(path="../input/ga-customer-revenue-prediction/test.csv")\n    \n    # Parse JSON columns in data\n    test_data = utils.parse_data(data=test_data)\n    \n    # Create a back up copy, for re-run\n    test_back_up = test_data.copy()')


# In[ ]:


# Check shape
train_data.shape


# In[ ]:


test_data.shape


# Two columns missing in test data. One will be label. Which is the second?

# In[ ]:


[c for c in train_data.columns if c not in set(test_data.columns)]


# It is *trafficSource_campaignCode*. We have to remove it eventually.

# ---

# Compare *visitId* and *visitStartTime* 

# In[ ]:


train_data[["visitId","visitStartTime"]].head()


# Confirming that *visitId* and *visitStartTime* are same

# In[ ]:


(~train_data['visitId']==train_data["visitStartTime"]).sum()


# In[ ]:


(~test_data['visitId']==test_data["visitStartTime"]).sum()


# They are exactly same, but will keep both becuase we are going to convert to *visitStartTime* to date time format.

# In[ ]:


# Convert VisitStartTime to datetime object
train_data['visitStartTime'] = train_data['visitStartTime'].apply(pd.datetime.fromtimestamp)
train_data['date'] = pd.to_datetime(train_data['date'], format="%Y%m%d")


# In[ ]:


# Convert VisitStartTime to datetime object
test_data['visitStartTime'] = test_data['visitStartTime'].apply(pd.datetime.fromtimestamp)
test_data['date'] = pd.to_datetime(test_data['date'], format="%Y%m%d")


# In[ ]:


train_data[["visitId","visitStartTime"]].head()


# ---

# #### Converting columns to applicable data types. We use just three for simplicty
# 1. str for strings
# 2. bool for binary variable with prefix _is_
# 3. float64 for all other numerical type, including whole numbers

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# For every column\nfor col in train_data.columns:\n    # Convert ID to string\n    if "Id" in col:\n        train_data[col] = train_data[col].astype(\'str\')\n    \n    # Convert to boolean if applicable\n    elif \'_is\' in col and len(train_data[col].unique()) == 2:\n            train_data[col] = train_data[col].astype(\'bool\')\n    \n    # Convert to float if applicable\n    else:\n        try:\n            train_data[col] = train_data[col].astype(\'float64\')\n        except ValueError:\n            pass\n        except TypeError:\n            pass')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# For every column\nfor col in test_data.columns:\n    # Convert ID to string\n    if "Id" in col:\n        test_data[col] = test_data[col].astype(\'str\')\n    # Convert to boolean if applicable\n    elif \'_is\' in col and len(test_data[col].unique()) == 2:\n            test_data[col] = test_data[col].astype(\'bool\')\n    # Convert to float if applicable\n    else:\n        try:\n            test_data[col] = test_data[col].astype(\'float64\')\n        except ValueError:\n            pass\n        except TypeError:\n            pass')


# Some columns includes NaN as string. It needs to be replaced with np.nan

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Replace "nan" and "NaN" strings with np.NaN object\ntrain_data.replace(["nan", "NaN"], np.nan, inplace=True)    ')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Replace "nan" and "NaN" strings with np.NaN object\ntest_data.replace(["nan", "NaN"], np.nan, inplace=True)    ')


# Check the datastypes

# In[ ]:


train_data.dtypes


# ## Data Analysis

# Plotting function definition

# In[ ]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly import tools


# In[ ]:


init_notebook_mode(connected=True)


# ### Training Data

# Missing values in each column for training data

# In[ ]:


train_missing_values = defaultdict(list)
train_missing_percentage = defaultdict(list)
for col in train_data.columns:
    if train_data[col].isnull().sum() > 0:
        train_missing_values[col] = train_data[col].isnull().sum()
        train_missing_percentage[col] = train_data[col].isnull().sum()/train_data.shape[0]


# In[ ]:


figure = bar_chart(x_values=(train_missing_values.keys(),), 
                   y_values=(train_missing_values.values(),), 
                   title="Missing Values", 
                   orientation="h")
iplot(figure)


# Only a handful of columns contain missing data. Now let's check the unique values in those columns.

# In[ ]:


for i, c in enumerate(train_missing_values.keys()):
        print(f"{i+1}. {c}")
        try:
            print("\t", train_data[c].unique())
        except TypeError:
            print("\t", "Cannot parse")
        print("\n")


# It looks like the *totals_* columns has nan in place of zero. So let's fill nan with zeros

# In[ ]:


train_data["totals_transactionRevenue"].fillna(0, inplace=True)
train_data["totals_bounces"].fillna(0, inplace=True)
train_data["totals_newVisits"].fillna(0, inplace=True)
train_data["totals_pageviews"].fillna(0, inplace=True)


# Some other columns in trafficSource can also be filled *reasonably*

# In[ ]:


train_data["trafficSource_adwordsClickInfo_page"].fillna(0, inplace=True)


# ### Test Data

# Repeat same for test data

# In[ ]:


test_missing_values = defaultdict(list)
test_missing_percentage = defaultdict(list)
for col in test_data.columns:
    if test_data[col].isnull().sum() > 0:
        test_missing_values[col] = test_data[col].isnull().sum()
        test_missing_percentage[col] = test_data[col].isnull().sum()/test_data.shape[0]


# In[ ]:


figure = bar_chart(x_values=(test_missing_values.keys(), ), 
                   y_values=(test_missing_values.values(), ), 
                   title="Missing Values", 
                   orientation="h")
iplot(figure)


# Test data also has almost same distribution. Apart from the fact that number of nan is somewhat less

# In[ ]:


for i, c in enumerate(test_missing_values.keys()):
        print(f"{i+1}. {c}")
        try:
            print("\t", test_data[c].unique())
        except TypeError:
            print("\t", "Cannot parse")
        print("\n")


# In[ ]:


test_data["totals_bounces"].fillna(0, inplace=True)
test_data["totals_newVisits"].fillna(0, inplace=True)
test_data["totals_pageviews"].fillna(0, inplace=True)


# In[ ]:


test_data["trafficSource_adwordsClickInfo_page"].fillna(0, inplace=True)


# ### Cleanup Data 

# Now as we have filled whatever can be filled by inference, remove rest of the columns with null values

# In[ ]:


# Remove columns with more than p% missing values
p = 0.0
for col in train_missing_percentage:
    if train_data[col].isnull().sum() > int(p*train_data.shape[0]):
        try:
            print(col)
            del train_data[col]
            del test_data[col]
        except KeyError:
            pass


# ---

# In[ ]:


print("Columns in train data only are", set(train_data.columns) - set(test_data.columns))


# In[ ]:


print("Columns in test data only are", set(test_data.columns) - set(train_data.columns))


# As expected label 'totals_transactionRevenue' is only present in training data, while test data does not have any such columns.

# ### Unique values analysis

# The unique values present in every columns can be very useful.

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Find numnber of unique values for each column\nunique_values_train = {}\nfor col in train_data.columns:\n    unique_values_train[col] = len(train_data[col].unique())')


# In[ ]:


# Create dictionary with ID fields and their unique values
identifier_fields_train = {}
for k, v in unique_values_train.items():
    if "Id" in k and ("visit" in k.lower() or "session" in k.lower()) or k=="visitStartTime":
        identifier_fields_train[k] = v
for k in identifier_fields_train:
    try:
        del unique_values_train[k]
    except KeyError:
        pass


# In[ ]:


TOO_MANY = 1000


# We will split columns in to three groups.
# 1. With single unique values (useless for almost everything)
# 2. Moderate range (2 to 1000)
# 3. Too many (>1000)

# In[ ]:


# Create three seprate dictionary to hold number of unique values
moderate_values_train = {}
single_value_train = {}
too_many_values_train = {}
for k, v in unique_values_train.items():
    if v > 1 and v < TOO_MANY:
        moderate_values_train[k] = v
    elif v == 1:
        single_value_train[k] = v
    else:
        too_many_values_train[k] = v


# In[ ]:


# List columns with a single value
# This columns are useless for ML or analysis
pd.DataFrame({"Column": list(single_value_train.keys()), "Value": [train_data[k][0] for k in single_value_train.keys()]}) 


# As we can most of this are pseudofields, not actually present in dataset. Let's remove them from test data. Will also remove it from test data after having a look.

# In[ ]:


train_data = train_data[[c for c in train_data.columns if c not in set(single_value_train.keys())]]


# ---

# Now let's plot the moderate range columns

# In[ ]:


# Plot columns moderate number of unique values
figure = bar_chart(x_values=(moderate_values_train.keys(), ), 
                   y_values=(moderate_values_train.values(),), 
                   title="Unique Values (Moderate Range)", 
                   orientation="v")

iplot(figure)


# In[ ]:


too_many_values_train


# In[ ]:


figure = bar_chart(x_values=(too_many_values_train.keys(),), 
                   y_values=(too_many_values_train.values(),), 
                   title="Unique Values (High Range)",
                   orientation="h", 
                   height=300)

iplot(figure)


# "totals_transactionRevenue" is numeric and thus large number of unique values are expected. Other features are categorical and they are likely to be dropped, unless they provide much useful information. 

# This are just the different terms people have searched. May be it categorised as prcided or not provided.

# In[ ]:


list(train_data["geoNetwork_networkDomain"].unique())[0:20]


# Looks like it's related to ISP. Not sure if useful. Will probably delete.

# ---

# Let's check the ID fields in brief

# In[ ]:


identifier_fields_train


# In[ ]:


train_data[["fullVisitorId", "visitId", "sessionId", "visitStartTime"]].head()


# ### Test Data

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Find numnber of unique values for each column\nunique_values_test = {}\nfor col in test_data.columns:\n    unique_values_test[col] = len(test_data[col].unique())')


# In[ ]:


# Create dictionary with ID fields and their unique values
identifier_fields_test = {}
for k, v in unique_values_test.items():
    if "Id" in k and ("visit" in k.lower() or "session" in k.lower()) or k=="visitStartTime":
        identifier_fields_test[k] = v
for k in identifier_fields_test:
    try:
        del unique_values_test[k]
    except KeyError:
        pass


# In[ ]:


# Create three seprate dictionary to hold number of unique values
moderate_values_test = {}
single_value_test = {}
too_many_values_test = {}
for k, v in unique_values_test.items():
    if v > 1 and v < 1000:
        moderate_values_test[k] = v
    elif v == 1:
        single_value_test[k] = v
    else:
        too_many_values_test[k] = v


# In[ ]:


# List columns with a single value
# This columns are useless for ML or analysis
pd.DataFrame({"Column": list(single_value_test.keys()), "Value": [test_data[k][0] for k in single_value_test.keys()]}) 


# Let's check if single value columns are same for test and train data.

# In[ ]:


set(single_value_train.keys())-(set(single_value_test.keys()))


# In[ ]:


(set(single_value_test.keys()))-set(single_value_train.keys())


# They are same. Let's remove those columns from test data too.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntest_data = test_data[[c for c in test_data.columns if c not in set(single_value_test.keys())]]')


# ---

# Plot moderate values for test data

# In[ ]:


# Plot columns moderate number of unique values
figure = bar_chart(x_values=(moderate_values_test.keys(), ), 
                   y_values=(moderate_values_test.values(), ), 
                   title="Unique Values (Moderate Range) Test Data", 
                   orientation="v")

iplot(figure)


# Plot number of unique values in test and train for each column in moderate range

# In[ ]:


# Plot columns moderate number of unique values
figure = bar_chart(x_values=(moderate_values_train.keys(), moderate_values_test.keys()), 
                   y_values=(moderate_values_train.values(), moderate_values_test.values()), 
                   names = ("Train", "Test"),
                   title="Unique Values (Moderate Range)", 
                   orientation="v")

iplot(figure)


# As we can see number of unique values are different for test and train, when the the number is highers. We need to shrink them in a way the categories are same for test and train.

# In[ ]:


too_many_values_test


# In[ ]:


figure = bar_chart(x_values=(too_many_values_test.keys(), ), 
                   y_values=(too_many_values_test.values(), ), 
                   title="Unique Values (High Range) Test Data",
                   orientation="h", 
                   height=300)

iplot(figure)


# In[ ]:


identifier_fields_test


# In[ ]:


test_data[["fullVisitorId", "visitId", "sessionId", "visitStartTime"]].head()


# ## Columnwise Analysis

# In[ ]:


train_data.shape


# In[ ]:


column_type = {}
for k in train_data.columns:
    if 'Id' in k:
        column_type[k] = "Identifier"
    elif np.issubdtype(train_data[k].dtype, np.number):
        column_type[k] = "Numerical"
    elif train_data[k].dtype == 'object':
        column_type[k] = "Categorical"
    elif np.issubdtype(train_data[k].dtype, np.datetime64):
        column_type[k] = "DateTime"
    elif np.issubdtype(train_data[k].dtype, np.bool_):
        column_type[k] = "Binary"
    else:
        column_type[k] = "Unknown"


# In[ ]:


column_info = pd.DataFrame({"ColumnName":list(train_data.columns), 
              "ColumnsType": [column_type[k] for k in train_data.columns], 
              "DataType":list(train_data.dtypes)}, index=np.arange(1, len(list(train_data.columns))+1))


# In[ ]:


column_info


# ### Channel Grouping

# In[ ]:


figure, _ = value_counts_barchart(data=train_data, column="channelGrouping", title_suffix="Train Data" )
iplot(figure)


# #### Test Data

# In[ ]:


figure, _ = value_counts_barchart(data=test_data, column="channelGrouping", title_suffix="Test Data")
iplot(figure)


# ### Device Information

# In[ ]:


column_info.loc[column_info["ColumnName"].str.contains("device_"), :]


# #### Device Browser

# In[ ]:


figure, counts = value_counts_barchart(data=train_data, column="device_browser", orientation="v", title_suffix="Train Data")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=train_data, column="device_browser", title_suffix="Train Data")

iplot(figure)


# In[ ]:


figure, counts = value_counts_barchart(data=test_data, column="device_browser", orientation="v", title_suffix="Test Data")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=test_data, column="device_browser", title_suffix="Test Data")
iplot(figure)


# #### Device  IsMobile 

# In[ ]:


figure, _ = value_counts_barchart(data=train_data, column="device_isMobile", title_suffix="Train Data", orientation='h')

iplot(figure)


# In[ ]:


figure, _ = value_counts_barchart(data=test_data, column="device_isMobile", title_suffix="Test Data", orientation='h')

iplot(figure)


# #### Device Category

# In[ ]:


figure, _ = value_counts_barchart(data=train_data, column="device_deviceCategory", title_suffix="Train Data", orientation='h')

iplot(figure)


# In[ ]:


figure, _ = value_counts_barchart(data=test_data, column="device_deviceCategory", title_suffix="Test Data", orientation='h')

iplot(figure)


# #### Device Operating System

# In[ ]:


# Plot chennle grouping values relevant
figure,  _ = value_counts_barchart(data=train_data, column="device_operatingSystem", orientation='v', title_suffix="Train Data")

iplot(figure)


# In[ ]:


# Plot chennle grouping values relevant
figure,  _ = value_counts_barchart(data=test_data, column="device_operatingSystem", orientation='v', title_suffix="Test Data")

iplot(figure)


# ### GeoNetwork

# In[ ]:


column_info.loc[column_info["ColumnName"].str.contains("geoNetwork_"), :]


# #### GeoNetwork - City

# In[ ]:


# Plot chennle grouping values breakdown
figure, counts = value_counts_barchart(data=train_data, column="geoNetwork_city", orientation="v", title_suffix="Train Data")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=train_data, column="geoNetwork_city", orientation='v', title_suffix="Train Data")
iplot(figure)


# In[ ]:


# Plot chennle grouping values breakdown
figure, counts = value_counts_barchart(data=test_data, column="geoNetwork_city", orientation="v", title_suffix="Test Data")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=test_data, column="geoNetwork_city", orientation='v', title_suffix="Test Data")
iplot(figure)


# #### GeoNetwork - Metro

# In[ ]:


# Plot chennle grouping values breakdown
figure, counts = value_counts_barchart(data=train_data, column="geoNetwork_metro", orientation="v", title_suffix="Train Data")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=train_data, column="geoNetwork_metro", orientation='v', title_suffix="Train Data")
iplot(figure)


# In[ ]:


figure, counts = value_counts_barchart(data=test_data, column="geoNetwork_metro", orientation="v", title_suffix="Test Data")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=train_data, column="geoNetwork_metro", orientation='v', title_suffix="Test Data")
iplot(figure)


# #### GeoNetwork  Subcontinent

# In[ ]:


figure,  _ = value_counts_barchart(data=train_data, column="geoNetwork_subContinent", title_suffix="Train Data")

iplot(figure)


# In[ ]:


figure,  _ = value_counts_barchart(data=test_data, column="geoNetwork_subContinent", title_suffix="Test Data")

iplot(figure)


# #### GeoNetwork  Continent

# In[ ]:


figure,  _ = value_counts_barchart(data=train_data, column="geoNetwork_continent", title_suffix="Train Data")

iplot(figure)


# In[ ]:


figure,  _ = value_counts_barchart(data=test_data, column="geoNetwork_continent", title_suffix="Test Data")

iplot(figure)


# #### GeoNetwork  Region

# In[ ]:


figure, counts = value_counts_barchart(data=train_data, column="geoNetwork_region", orientation="v", title_suffix="Train Data")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=train_data, column="geoNetwork_region", orientation='v', title_suffix="Train Data")
iplot(figure)


# In[ ]:


figure, counts = value_counts_barchart(data=test_data, column="geoNetwork_region", orientation="v", title_suffix="Test Data")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=test_data, column="geoNetwork_region", orientation='v', title_suffix="Test Data")

iplot(figure)


# #### GeoNetwork  Country

# In[ ]:


figure, counts = value_counts_barchart(data=train_data, column="geoNetwork_country", orientation="v", title_suffix="Train Data")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=train_data, column="geoNetwork_country", orientation='v', title_suffix="Train Data")

iplot(figure)


# In[ ]:


figure, counts = value_counts_barchart(data=test_data, column="geoNetwork_country", orientation="v", title_suffix="Test Data")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=test_data, column="geoNetwork_country", orientation='v', title_suffix="Test Data")

iplot(figure)


# ### Totals Information

# In[ ]:


column_info.loc[column_info["ColumnName"].str.contains("totals_"), :]


# #### Totals - New Visits

# In[ ]:


figure, counts = value_counts_barchart(data=train_data, column="totals_newVisits", title_suffix="Train Data", orientation='h')

iplot(figure)


# In[ ]:


figure, counts = value_counts_barchart(data=test_data, column="totals_newVisits", title_suffix="Test Data", orientation='h')

iplot(figure)


# #### Total - Bounces

# In[ ]:


figure, counts = value_counts_barchart(data=train_data, column="totals_bounces", title_suffix="Train Data", orientation='h')

iplot(figure)


# In[ ]:


figure, counts = value_counts_barchart(data=test_data, column="totals_bounces", title_suffix="Test Data", orientation='h')

iplot(figure)


# #### Totals - Page Views

# In[ ]:


fig, count = value_counts_linechart(data=train_data, column="totals_pageviews", title_suffix="Train Data")
iplot(fig)


# In[ ]:


figure, counts = value_counts_linechart(data=test_data, column="totals_pageviews", title_suffix="Test Data")

iplot(figure)


# #### Totals - Hits

# In[ ]:


figure, counts = value_counts_linechart(data=train_data, column="totals_hits", title_suffix="Train Data")

iplot(figure)


# In[ ]:


figure, counts = value_counts_linechart(data=test_data, column="totals_hits", title_suffix="Test Data")

iplot(figure)


# ### Traffic Source Information

# In[ ]:


column_info.loc[column_info["ColumnName"].str.contains("trafficSource_"), :]


# #### TrafficSource - Source 

# In[ ]:


figure, counts = value_counts_barchart(data=train_data, column="trafficSource_source", title_suffix="Train Data", orientation="v")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=train_data, column="trafficSource_source", title_suffix="Train Data", orientation="v")

iplot(figure)


# In[ ]:


figure, counts = value_counts_barchart(data=test_data, column="trafficSource_source", title_suffix="Test Data", orientation="v")

if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=test_data, column="trafficSource_source", title_suffix="Test Data", orientation="v")

iplot(figure)


# #### TrafficSource - Medium 

# In[ ]:


figure, counts = value_counts_barchart(data=train_data, column="trafficSource_medium",  orientation="v", title_suffix="Train Data")

iplot(figure)


# In[ ]:


figure, counts = value_counts_barchart(data=test_data, column="trafficSource_medium",  orientation="v", title_suffix="Test Data")

iplot(figure)


# #### TrafficSource - isVideoAd 

# In[ ]:


figure, counts = value_counts_barchart(data=train_data, column="trafficSource_adwordsClickInfo_isVideoAd", orientation="h", title_suffix="Train Data")

iplot(figure)


# In[ ]:


figure, counts = value_counts_barchart(data=test_data, column="trafficSource_adwordsClickInfo_isVideoAd", orientation="h", title_suffix="Test Data")

iplot(figure)


# #### TrafficSource - AdwordsClickInfoPage

# In[ ]:


figure, count = value_counts_linechart(data=train_data, column="trafficSource_adwordsClickInfo_page", title_suffix="Train Data")
iplot(figure)


# In[ ]:


figure, count = value_counts_linechart(data=test_data, column="trafficSource_adwordsClickInfo_page", title_suffix="Test Data")
iplot(figure)


# ### Training Labels

# In[ ]:


train_data["log_revenue"] = np.log1p(np.array(train_data["totals_transactionRevenue"], dtype='float64'))


# In[ ]:


train_data["isRevenue"] = train_data["log_revenue"]!=0


# In[ ]:


fig, _ = value_counts_barchart(data=train_data, column="isRevenue", orientation='h')
iplot(fig)


# In[ ]:


non_zero_values = list(filter(lambda x: x!=0, list(train_data["log_revenue"])))


# In[ ]:


data = [go.Histogram(x=non_zero_values)]
layout = go.Layout(title="Nonzero Revenue Distribution",xaxis=dict(title="Revenue (log1p)"), yaxis=dict(title="Frequency"))
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# ### Visit Counts Analysis

# #### Train Data

# In[ ]:


train_subset = train_data[["fullVisitorId", "date"]]


# In[ ]:


visit_counts_train = train_subset.groupby("fullVisitorId").count()


# In[ ]:


visit_counts_train.rename(columns={"date":"count"}, inplace=True)


# In[ ]:


counts = dict(visit_counts_train["count"].value_counts())


# In[ ]:


fig, _ = value_counts_linechart(data=counts, title="Visit Counts")


# In[ ]:


iplot(fig)


# In[ ]:


fig, _ = merged_tail_linechart(data=visit_counts_train, column="count")
iplot(fig)


# ---

# #### Test Data

# In[ ]:


test_subset = test_data[["fullVisitorId", "date"]]


# In[ ]:


visit_counts_test = test_subset.groupby("fullVisitorId").count()


# In[ ]:


visit_counts_test.rename(columns={"date":"count"}, inplace=True)


# In[ ]:


counts = dict(visit_counts_test["count"].value_counts())


# In[ ]:


fig, _ = value_counts_linechart(data=counts, title="Visit Counts Test Data")
iplot(fig)


# In[ ]:


fig, _ = merged_tail_linechart(data=visit_counts_test, column="count", title_suffix="Test Data")
iplot(fig)


# ### Date/ Time Analysis

# #### Train Data

# In[ ]:


visit_per_day_train = train_subset.groupby("date", as_index=False).count()


# In[ ]:


revenue_visit_per_day = train_data.loc[train_data["isRevenue"], ["date", "fullVisitorId"]].groupby("date", as_index=False).count()


# In[ ]:


visit_per_day_train.rename(columns={"fullVisitorId":"count"}, inplace=True)


# In[ ]:


revenue_visit_per_day.rename(columns={"fullVisitorId":"count"}, inplace=True)


# In[ ]:


trace1 = go.Scatter(x=visit_per_day_train["date"], y=visit_per_day_train["count"], name="All visits")
trace2 = go.Scatter(x=revenue_visit_per_day["date"], y=revenue_visit_per_day["count"], name="Visits with revenue", yaxis="y2")
fig = tools.make_subplots(rows=2, cols=1, specs=[[{}], [{}]], shared_xaxes=True, vertical_spacing=0.01)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
iplot(fig)


# #### Test Data

# In[ ]:


visit_per_day_test = test_subset.groupby("date", as_index=False).count()


# In[ ]:


visit_per_day_test.rename(columns={"fullVisitorId":"count"}, inplace=True)


# In[ ]:


data=[go.Scatter(x=visit_per_day_test["date"], y=visit_per_day_test["count"])]
iplot(data)

