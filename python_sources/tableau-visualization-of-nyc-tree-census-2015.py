#!/usr/bin/env python
# coding: utf-8

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


# 2015 Street Tree Census - Tree Data
# 
# This is a dataset hosted by the City of New York. The city has an open data platform found here: https://opendata.cityofnewyork.us/ and they update their information according the amount of data that is brought in. I will analysed and explored the dataset and built a interactive dahboard using Tableau which helpsto gain deeper insights, with the ability to analyze the data at a more granular level
# 
# Tree Census, was conducted by volunteers and staff organized by NYC Parks & Recreation and partner organizations. Tree data collected includes tree species, diameter and perception of health. Accompanying blockface data is available indicating status of data collection and data release citywide.
# 
# * Dashboard Link - https://public.tableau.com/profile/vivek2206#!/vizhome/NewYorkTreeCensus2015/Dashboard1

# In[ ]:


# Import Tableau Visualisation 
from IPython.display import IFrame
IFrame('https://public.tableau.com/views/NewYorkTreeCensus2015/Dashboard1?:showVizHome=no&:embed=true', width=1200, height=925)


# In[ ]:




