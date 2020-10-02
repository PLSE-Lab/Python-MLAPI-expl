#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/survey_results_public.csv')


# In[ ]:


data.head(10)


# In[ ]:


data['Country'].value_counts()[0:10].plot(kind='bar',figsize=(12,10))


# In[ ]:


data['EducationImportant'].value_counts().plot(kind='pie',figsize=(12,10))


# In[ ]:


data[data['Country'] == 'Romania']['EducationImportant'].value_counts().plot(kind='pie',figsize=(12,10))


# In[ ]:


(
    data
    .groupby('EducationImportant')['JobSatisfaction']
    .mean()
    .sort_values(ascending=False)
)


# In[ ]:


romania = data[(data['Country'] == 'Romania') & (data['EducationImportant'] != None)]


# In[ ]:


(
    romania
    .groupby('EducationImportant')['JobSatisfaction']
    .mean()
    .sort_values(ascending=False)
)


# In[ ]:


romania['JobSatisfaction'].mean()


# In[ ]:


(
    romania
    .groupby('EducationImportant')['JobSatisfaction']
    .count()
    .sort_values(ascending=False)
)


# In[ ]:




