#!/usr/bin/env python
# coding: utf-8

# My purpose with this project is to create an interactive map of Alaska with a significant portion of the data in this dataset available as an aid in emergency planning. Whether I have the technical skills to succeed or not is, of course, debatable at this point. So let's find out.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


facilities = pd.read_csv('../input/AK Airport Facilities Data.csv')
runways = pd.read_csv('../input/AK Airport Runway Data.csv')
schedules = pd.read_csv('../input/AK Airport Schedules.csv')
remarks = pd.read_csv('../input/AK Airports Remarks.csv')

facilities.info()
runways.info()
schedules.info()
remarks.info()


# The immediate challenge is to ensure that I have cleaned each of these files and then combine them into a single file so I can access all of any given airport's information at once without having to go to multiple DataFrames to look it up. I'll start with the facilities information and work my way down from there.

# In[ ]:


facilities.info(verbose=True, null_counts=True)


# Looks like I have a few columns with null values, but not all of the information here is relevant to the end state that I am trying to achieve. Time to trim this sucker down to what I actually need to do this project.

# In[ ]:


facilities = facilities.drop(['Region', 'State', 'CountyState', 'StateName', 'EffectiveDate', 'TrafficPatternAltitude', 'LandAreaCoveredByAirport', 'ResponsibleARTCCID', 'ResponsibleARTCCComputerID', 'ResponsibleARTCCName', 'CertificationTypeDate', 'OperationsCommuter', 'ContractFuelAvailable'], axis=1)


# In[ ]:


facilities.fillna(value=0, axis=1)
facilities.info(verbose=True, null_counts=True)

