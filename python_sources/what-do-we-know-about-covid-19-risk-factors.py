#!/usr/bin/env python
# coding: utf-8

# # What do we know about COVID-19 risk factors?
# ## Task Details
# ### What do we know about COVID-19 risk factors? What have we learned from epidemiological studies?
# 
# ### Specifically, we want to know what the literature reports about:
# 
# * Data on potential risks factors
# * Smoking, pre-existing pulmonary disease
# * Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities
# * Neonates and pregnant women
# * Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.
# * Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors
# * Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups
# * Susceptibility of populations
# * Public health mitigation measures that could be effective for control

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


import functools
from IPython.core.display import display, HTML
Tasks2 = pd.read_csv('/kaggle/input/Task2.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks2_table=HTML(Tasks2.to_html(escape=False,index=False))
display(Tasks2_table)

