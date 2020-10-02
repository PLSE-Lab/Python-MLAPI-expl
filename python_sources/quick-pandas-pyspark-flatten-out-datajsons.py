#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Just import all the important packages you need (pandas / pyspark chiefly)
and load your datasets up!

for pyspark with jupyter, use :
downloading spark from a mirror : 
http://mirrors.estointernet.in/apache/spark/spark-2.3.2/spark-2.3.2-bin-hadoop2.7.tgz
tar -xzvf <spark-downloaded.tgz above>
mv <spark-above> spark

and set your ~/.bashrc as :
export SPARK_HOME=<curr-working-dir>/spark
export PATH=$SPARK_HOME/bin:$PATH

export PYSPARK_DRIVER_PYTHON=ipython
export PYSPARK_DRIVER_PYTHON_OPTS='notebook'

source ~/.bashrc
"""

import pandas as pd
import numpy as np
import pyspark
from pyspark.sql import SparkSession
import sys, os, json, time

train_df = pd.read_csv('./input/train.csv',header='infer')
test_df = pd.read_csv('./input/test.csv',header='infer')


# In[ ]:


"""
separate out columns that are already flattened, i.e not jsons
"""
train_df_1 = train_df[['channelGrouping','date','fullVisitorId','sessionId','socialEngagementType','visitId','visitNumber','visitStartTime']]
test_df_1 = test_df[['channelGrouping','date','fullVisitorId','sessionId','socialEngagementType','visitId','visitNumber','visitStartTime']]


# In[ ]:


"""
create separate files for 'device', 'geoNetwork', 'totals', 'trafficSource' json attribs
"""

def write_to_file(mode,df,tag):
    f = open(mode+"_df_"+str(tag)+".json", "w")
    for index, row in df.iterrows():
        try:
            print >> f, row[tag]
        except:
            print "mode: "+str(mode)+"pos: "+str(index)+" for tag: "+tag 
            print >> f, "{}"
    f.close()

write_to_file('train',train_df['device'].str.replace('""','"').replace('"{','"').replace('}"','').to_frame(),'device')
write_to_file('train',train_df['geoNetwork'].str.replace('""','"').replace('"{','"').replace('}"','').to_frame(),'geoNetwork')
write_to_file('train',train_df['totals'].str.replace('""','"').replace('"{','"').replace('}"','').to_frame(),'totals')
write_to_file('train',train_df['trafficSource'].str.replace('""','"').replace('"{','"').replace('}"','').to_frame(),'trafficSource')

write_to_file('test',test_df['device'].str.replace('""','"').replace('"{','"').replace('}"','').to_frame(),'device')
write_to_file('test',test_df['geoNetwork'].str.replace('""','"').replace('"{','"').replace('}"','').to_frame(),'geoNetwork')
write_to_file('test',test_df['totals'].str.replace('""','"').replace('"{','"').replace('}"','').to_frame(),'totals')
write_to_file('test',test_df['trafficSource'].str.replace('""','"').replace('"{','"').replace('}"','').to_frame(),'trafficSource')


# In[ ]:


"""
redundant but neat : load the training and testing dataframes back, these together form part-2
for train and test features set, respectively
"""
train_df_device = spark.read.json('train_df_device.json').toPandas()
train_df_geo = spark.read.json('train_df_geoNetwork.json').toPandas()
train_df_totals = spark.read.json('train_df_totals.json').toPandas()
train_df_traffic = spark.read.json('train_df_trafficSource.json').toPandas()

test_df_device = spark.read.json('test_df_device.json').toPandas()
test_df_geo = spark.read.json('test_df_geoNetwork.json').toPandas()
test_df_totals = spark.read.json('test_df_totals.json').toPandas()
test_df_traffic = spark.read.json('test_df_trafficSource.json').toPandas()

"""
time to concat : final outputs tr_df and te_df for training and testing dataframes, respectively
"""
tr_df = pd.concat([train_df_1.reset_index(drop=True), train_df_device.reset_index(drop=True)], axis=1)
tr_df = pd.concat([tr_df.reset_index(drop=True), train_df_geo.reset_index(drop=True)], axis=1)
tr_df = pd.concat([tr_df.reset_index(drop=True), train_df_totals.reset_index(drop=True)], axis=1)
tr_df = pd.concat([tr_df.reset_index(drop=True), train_df_traffic.reset_index(drop=True)], axis=1)

te_df = pd.concat([test_df_1.reset_index(drop=True), test_df_device.reset_index(drop=True)], axis=1)
te_df = pd.concat([te_df.reset_index(drop=True), test_df_geo.reset_index(drop=True)], axis=1)
te_df = pd.concat([te_df.reset_index(drop=True), test_df_totals.reset_index(drop=True)], axis=1)
te_df = pd.concat([te_df.reset_index(drop=True), test_df_traffic.reset_index(drop=True)], axis=1)

"""
Now feature engg. run models and shine on ;)
"""

