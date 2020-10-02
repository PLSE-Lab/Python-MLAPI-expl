#!/usr/bin/env python
# coding: utf-8

# ## 1- How big is Basch data? ##
# in this first section we'll do a little inspection of the data file and see what we can learn.

# In[ ]:


import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


INPUT_PATH = "../input/%s.csv"
matplotlib.style.use('ggplot')


# In[ ]:


def get_failure_rate(file_name):
    # will calculate the error rate from data on the file based on response col
    # Respons: 1 = Failed QC , 0 = Passed QC
    rows = pd.read_csv(INPUT_PATH % file_name, usecols=["Response"])
    failure_rate = rows[rows.Response == 1].size / float(rows[rows.Response == 0].size)
    return failure_rate


def explore_data_size():
    # explore the size (rows, cols) of each file
    data_files = ['train_numeric', 'train_date', 'train_categorical', 'test_numeric',
                  'test_date', 'test_categorical']
    stats = []
    for file_name in data_files:
        cols = pd.read_csv(INPUT_PATH % file_name, nrows=1)
        rows = pd.read_csv(INPUT_PATH % file_name, usecols=["Id"])
        stats.append({'File': file_name, 'Rows': rows.shape[0], 'Columns': cols.shape[1]})
    # convert the result into a DataFrame so we can do plotting.
    df = pd.DataFrame(stats, columns=["File", "Rows", "Columns"])
    failure_rate = get_failure_rate('train_numeric')
    df["Error"] = 0
    df.loc[df.File == 'train_numeric', 'Error'] = failure_rate
    return df
    

# due to time/memory limitation I decided to comment this part, but you can run it localy if you need to
# res = explore_data_size()
# res.plot(kind='bar', x='File', y='Columns', title="Fig:1 - Columns per file",
#         figsize=(13,5), fontsize=14)
# print(res)


# The file size exploration results
# 
#  
# 
#                    File     Rows  Columns     Error
#     0      train_numeric  1183747      970  0.005845
#     1         train_date  1183747     1157  0.000000
#     2  train_categorical  1183747     2141  0.000000
#     3       test_numeric  1183748      969  0.000000
#     4          test_date  1183748     1157  0.000000
#     5   test_categorical  1183748     2141  0.000000

# From the results above we can say:
# 
#  1. We are dealing with more 1 million record per file, so we just need to be careful not read the whole file at once, especially if you have limited memory.
#  2. The number of columns per file is also huge 1000 to 2000 per file. Each column represents a feature the corresponds to some measurement on a station or a machine.
#  3. Each file contains the "Id" column which represent the unique identifier of the product unit that we are tracking
#  4. Finally we notice that the error rate is about 0.5%, so it will be challenging to try to predict such small number from this huge data set. But we'll see...

# ## 2- Based on the features, how many stations, lines, failure rate per station? ##
# in this section we'll take a look at the features or the columns and try to see how many station and line of production we have? how many features does each station have? how many products were processed in each station? and what's the failure rate per station?

# In[ ]:


def get_features(feature_list):
    # function to group features by station or line of production sinc the convention is:
    # L1_S15_F232 means Line 1, Station 15, Feature 232
    line_features = {}
    station_features = {}
    lines = set([item.split('_')[0] for item in feature_list])
    stations = set([item.split('_')[1] for item in feature_list])

    for l in lines:
        line_features[l] = [item for item in feature_list if '%s_' % l in item]

    for s in stations:
        station_features[s] = [item for item in feature_list if '%s_' % s in item]

    return (line_features, station_features)


# In[ ]:


def explore_features(data_set_name):
    # how many units processed in each station
    # also percentage of failed/passed QS in each station
    features = pd.read_csv(INPUT_PATH % data_set_name, nrows=1).drop(["Response", "Id"], axis=1).columns.values
    line_features, station_features = get_features(features)
    # create a dataframe cols: station, features_count
    sdf = pd.DataFrame(list({int(key[1:]): len(station_features[key]) for
                             key in station_features.keys()}.items()),
                       columns=["Station", "FeatureCount"])
    ldf = pd.DataFrame(list({int(key[1:]): len(line_features[key]) for
                             key in line_features.keys()}.items()),
                       columns=["Line", "FeatureCount"])
    
    stations_plot = sdf.plot(x="Station", y="FeatureCount", kind="bar",
                             title="Fig:2 - Features by Station",
                             figsize=(13,6), fontsize=12)
    
    return line_features, station_features

lf, sf = explore_features("train_numeric")


# Wow! most stations has less than 20 features, except station 24, 25 which has more than 200 features each! those must be the biggest machines in this chocolate mousse factory.
# 
# So let's dive into the next section where we will try to calculate the number of units processed by each station (machine) and also what's the failure rate for each station.

# In[ ]:


def explore_product_by_station(data_set_name):
    # need to see how many product per station
    # features = pd.read_csv(INPUT_PATH % data_set_name, nrows=1).drop(["Response", "Id"], axis=1).columns.values
    # line_features, station_features = get_features(features)
    station_features = sf
    dwb_result = []
    ppbs_result = []
    rows = 0
    # inside this loop we will try to calculate different data sets, this way we only read the
    # files once since it takes a while and produce multiple results
    for station in station_features:
        station_data = pd.read_csv(
            INPUT_PATH % data_set_name,
            usecols=station_features[station] + ['Id', 'Response'])
        
        # need to get how many rows in this data set
        if not rows:
            rows = station_data.shape[0]
        # need to store how many processed units in each station
        # only if all features in that station has value we conside unit processed in this station
        ppbs_result.append(
            {'Station': int(station[1:]),
             'Processed': station_data[station_features[station]].isnull().all(axis=1).sum()})
        
        # data without blanks (dwb): take all rows and drop any row that has any blank
        # in any station column
        dwb = station_data.dropna(how="any")
        dwb_result.append([int(station[1:]), dwb.shape[0],
                       dwb[dwb["Response"] == 1].shape[0],
                       dwb[dwb["Response"] == 0].shape[0]])
        
    # convert list to a dataframe and prepare for the plot
    dwb_df = pd.DataFrame(
        dwb_result, columns=["Station", "Count", "Failed", "Passed"]).sort_values(by=["Station"])
    # calculate the error rate for each station
    dwb_df["Error"] = dwb_df.apply(lambda row: float(row["Failed"]) / row["Passed"], axis=1)
    
    # plot the stacked bar tot product/ station
    my_plot = dwb_df[["Station", "Failed", "Passed"]].plot(kind="bar", stacked=True, x="Station",
        title="Fig:3 - Products by station", figsize=(13,6), fontsize=12)
    my_plot.set_xlabel("Stations")
    my_plot.set_ylabel("Record Count")
    
    # plot error rate per station
    my_plot = dwb_df[["Station", "Error"]].plot(
        kind="bar", x="Station",
        title="Fig:4 - Error by station", figsize=(13,6), fontsize=12)
    my_plot.set_xlabel("Stations")
    my_plot.set_ylabel("Error %")
    
    # process the ppbs result to see how many products has been process by each station.
    ppbs_df = pd.DataFrame(
        ppbs_result, columns=['Station', 'Processed']).sort(columns=['Station'])
    # calculate the missed product for each station: all count - processed count
    ppbs_df["Missed"] = ppbs_df["Processed"].apply(lambda x: rows - x)
    # the plot
    ppbs_df.plot(x="Station", kind="bar", stacked=True,
            title="Fig: 5 - Products processed by each station",
            figsize=(13,6), fontsize=12)
    

explore_product_by_station("train_numeric")


# ## So what can we say by just looking at the 3 plots above: ##
# 
#  1. Station 24, 25 has by far the most number of features, more than 200 feature per stations! at the same time most of the stations has less than 20 features. And worth noticing that station 32 has only one feature.
# 
#  2. Not all stations have the same number of records or product! Does it mean that each product doesn't go through each station? could it be that we have different types of products and each one get processed through different station? Based on my understanding on manufacturing if let's say we are producing Orange Juice, then each juice box will go through the same process and the same machines until the juice is ready. Can someone tell me why the numbers are not equal for all the stations?
# 
#  3. The suspicious station 32 has really a minimal number of records. So it doesn't process a lot of products.
# 
#  4. Also worth noting that station 32 has by far most of QC failures almost 0.5%. At the same time all of the other stations has less than 0.1% as a failure rate! But wait S32 doesn't process a lot of records, so 0.5% of low number of records is minimal. However if each other station has a failure rate of 0.1% of the total production quantity then this number will add up eventually.
