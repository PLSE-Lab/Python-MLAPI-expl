#!/usr/bin/env python
# coding: utf-8

# # What is known about transmission, incubation, and environmental stability?
# ## Task Details
# ### What is known about transmission, incubation, and environmental stability? What do we know about natural history, transmission, and diagnostics for the virus? What have we learned about infection prevention and control?
# 
# ### Specifically, we want to know what the literature reports about:
# 
# 1. Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.
# 1. Prevalence of asymptomatic shedding and transmission (e.g., particularly children).
# 1. Seasonality of transmission.
# 1. Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).
# 1. Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).
# 1. Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).
# 1. Natural history of the virus and shedding of it from an infected person
# 1. Implementation of diagnostics and products to improve clinical processes
# 1. Disease models, including animal models for infection, disease and transmission Tools and studies to monitor phenotypic change and potential adaptation of the virus  Immune response and immunity
# 1. Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings
# 1. Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings
# Role of the environment in transmission

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


Tasks1 = pd.read_csv('/kaggle/input/task1answer/Task1.csv')


# In[ ]:


import functools
from IPython.core.display import display, HTML
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks1_table=HTML(Tasks1.to_html(escape=False,index=False))
display(Tasks1_table)

