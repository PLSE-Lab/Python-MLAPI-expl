#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data1 = pd.read_csv("/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneyCompactResults.csv")
data2 = pd.read_csv("/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WRegularSeasonCompactResults.csv")


# In[ ]:


frames = [data2, data1]
data = pd.concat(frames, ignore_index = True)
data['T1ID'] = data['WTeamID']
data['T2ID'] = data['LTeamID']
data['T1loc'] = data['WLoc']
data['T1Score'] = data['WScore']
data['T2Score'] = data['LScore']


# In[ ]:


for i in range(len(data)):
    if i % 1000 == 1:
        print('I = ', i)
    if (data.loc[i])['T1ID'] > (data.loc[i])['T2ID']:
        (data['T1ID']).loc[i], (data['T2ID']).loc[i], (data['T1Score']).loc[i], (data['T2Score']).loc[i] =         (data['T2ID']).loc[i], (data['T1ID']).loc[i], (data['T2Score']).loc[i], (data['T1Score']).loc[i]
        if (data['T1loc']).loc[i] == 'H':
            (data['T1loc']).loc[i] = 'A'
        elif (data['T1loc']).loc[i] == 'A':
            (data['T1loc']).loc[i] = 'H'


# In[ ]:


data.to_csv (r'rmwl.csv',header=True) #Don't forget to add '.csv' at the end of the path

