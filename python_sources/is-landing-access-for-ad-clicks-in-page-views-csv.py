#!/usr/bin/env python
# coding: utf-8

# As I checked if access information for landing page of ad clicks is in page_views.csv for clicks_test.csv in [this script](https://www.kaggle.com/its7171/outbrain-click-prediction/leakage-solution/discussion), I tryed to check for clicks_train.csv to estimate this efects.

# In[ ]:


import numpy as np
import pandas as pd

# set None for full data
#nrows=None
nrows=10000000

# Is access information for landing page of ad click in page_views.csv?
df_train = pd.read_csv('../input/clicks_train.csv', nrows=nrows)
df_ad = pd.read_csv('../input/promoted_content.csv', nrows=nrows,
    usecols  = ('ad_id','document_id'),
    dtype={'ad_id': np.int, 'uuid': np.str, 'document_id': np.str})
df_events = pd.read_csv('../input/events.csv',nrows=nrows,
    usecols  = ('display_id','uuid','timestamp'),
    dtype={'display_id': np.int, 'uuid': np.str, 'timestamp': np.int})
df_train = pd.merge(df_train, df_ad, on='ad_id', how='left')
df_train = pd.merge(df_train, df_events, on='display_id', how='left')
df_train['usr_doc'] = df_train['uuid'] + '_' + df_train['document_id']
df_train = df_train.set_index('usr_doc')
time_dict = df_train[['timestamp']].to_dict()['timestamp']
# set page_views.csv for full data
# f = open("../input/page_views.csv", "r")
f = open("../input/page_views_sample.csv","r")
line = f.readline().strip()
head_arr = line.split(",")
fld_index = dict(zip(head_arr,range(0,len(head_arr))))
total = 0
while 1:
    line = f.readline().strip()
    if nrows is not None and total == nrows:
        break
    total += 1
    if line == '':
        break
    arr = line.split(",")
    usr_doc = arr[fld_index['uuid']] + '_' + arr[fld_index['document_id']]
    if usr_doc in time_dict:
        #don't use timestamp yet.
        #time_diff = time_dict[usr_doc] - int(arr[fld_index['timestamp']])
        #if abs(time_diff) < 600:
            # set -1 if found that this user sow this document
            time_dict[usr_doc] = -1

df_train=df_train.reset_index()
df_train['fixed_timestamp'] = df_train['usr_doc'].apply(lambda x: time_dict[x])
found_in_page_views = set(df_train[df_train['fixed_timestamp'] < 0].index)
clicked = set(df_train[df_train['clicked'] == 1].index)
all_ids = set(df_train.index)

TP = len(clicked & found_in_page_views)
FP = len(found_in_page_views - clicked)
FN = len(clicked - found_in_page_views)
recall = TP/float(TP+FN)
precision = TP/float(TP+FP)
print('TP:{}'.format(TP))
print('FP:{}'.format(FP))
print('FN:{}'.format(FN))
print('recall:{0:.1f}%'.format(recall*100))
print('precision:{0:.1f}%'.format(precision*100))


# For full data, this output would be:
# 
# <pre>
# TP:724749
# FP:31813
# FN:16149844
# recall:4.3%
# precision:95.8%
# </pre>

# Only 4.3% of Click data is found in page_views.csv.
# This percentage is far smaller than I thought.
# I thought that landing page log should be in page_views.csv, if some ad link is clicked.
# Where is remaining 95.7% access?
# Does page_views.csv includes only 4.3% sampling data?
# Or Outbrain does not have all page view data for landing page of ads?
# 
# update: I guess that page_views.csv includes access logs for all the page which has ads in it. If ad landing page dose not have ads, the access for the landing page will not be recorded in page_views.csv. So only 4.3% of ad may have ads in the landing page.
#  
# On the other hand if access information for landing page of ad clicks is found in page_views.csv, 95.8% of them are clicked.
# Suppose test data has same high precision, this feature would be useful.
