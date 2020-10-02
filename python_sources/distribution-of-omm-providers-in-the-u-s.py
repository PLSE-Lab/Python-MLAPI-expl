#!/usr/bin/env python
# coding: utf-8

# In[17]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import csv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# In[18]:


# Load CMS Data
df_physcians = pd.read_csv('../input/Physician_Compare_National_Downloadable_File.csv', usecols=[11,24])
omm_only_df = df_physcians.loc[df_physcians['Primary specialty'] == 'OSTEOPATHIC MANIPULATIVE MEDICINE']

# Load 2015 Population Data
df_population = pd.read_json('''{"State":{"0":"AL","1":"AK","2":"AZ","3":"AR","4":"CA","5":"CO","6":"CT","7":"DC","8":"DE","9":"FL","10":"GA","11":"HI","12":"ID","13":"IL","14":"IN","15":"IA","16":"KS","17":"KY","18":"LA","19":"ME","20":"MD","21":"MA","22":"MI","23":"MN","24":"MS","25":"MO","26":"MT","27":"NE","28":"NV","29":"NH","30":"NJ","31":"NM","32":"NY","33":"NC","34":"ND","35":"OH","36":"OK","37":"OR","38":"PA","39":"RI","40":"SC","41":"SD","42":"TN","43":"TX","44":"UT","45":"VT","46":"VA","47":"WA","48":"WV","49":"WI","50":"WY"},"State Population":{"0":4858979,"1":738432,"2":6828065,"3":2978204,"4":39144818,"5":5456574,"6":3590886,"7":672228,"8":945934,"9":20271272,"10":10214860,"11":1431603,"12":1654930,"13":12859995,"14":6619680,"15":3123899,"16":2911641,"17":4425092,"18":4670724,"19":1329328,"20":6006401,"21":6794422,"22":9922576,"23":5489594,"24":2992333,"25":6083672,"26":1032949,"27":1896190,"28":2890845,"29":1330608,"30":8958013,"31":2085109,"32":19795791,"33":10042802,"34":756927,"35":11613423,"36":3911338,"37":4028977,"38":12802503,"39":1056298,"40":4896146,"41":858469,"42":6600299,"43":27469114,"44":2995919,"45":626042,"46":8382993,"47":7170351,"48":1844128,"49":5771337,"50":586107}}''')


# In[24]:


# Calculate Per Capita Distribution
merged_df = pd.merge(omm_only_df, df_population, on='State', how='left')
omm_only_df_grouped_by_state = merged_df.groupby(['State']).size().to_frame('Number of OMM Providers')
omm_only_df_grouped_by_state['State'] = omm_only_df_grouped_by_state.index
merged_df = pd.merge(omm_only_df_grouped_by_state, df_population, on='State', how='inner')
merged_df['OMM Providers Per Capita'] =  merged_df['Number of OMM Providers'] / merged_df['State Population']


# In[25]:


# Graph Results
merged_df.plot(x='State', y='OMM Providers Per Capita', kind='bar')

