#!/usr/bin/env python
# coding: utf-8

# # What has been published about ethical and social science considerations?
# ## Task Details
# ### What has been published concerning ethical considerations for research? What has been published concerning social sciences at the outbreak response?
# 
# ### Specifically, we want to know what the literature reports about:
# 
# * Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019
# * Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight
# * Efforts to support sustained education, access, and capacity building in the area of ethics
# * Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences.
# * Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)
# * Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed.
# * Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media.

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
Tasks9 = pd.read_csv('/kaggle/input/Task9.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks9_table=HTML(Tasks9.to_html(escape=False,index=False))
display(Tasks9_table)

