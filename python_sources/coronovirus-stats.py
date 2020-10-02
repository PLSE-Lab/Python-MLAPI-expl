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


all_people_in_world = 7650000000
confirmen_covid = 132000
CFR_total = confirmen_covid / all_people_in_world * 100
CFR_total


# we have 262 countries in the world
# Justin Trudeau and Jair Messias Bolsonaro had positive corona test

# In[ ]:


all_premiers = 262
confirmen_pr = 2
CFR_premiers  = confirmen_pr / all_premiers * 100
CFR_premiers


# Serie A, also called Serie A TIM due to sponsorship by TIM, is a professional league competition for football clubs located at the top of the Italian football league system and the winner is awarded the Scudetto and the Coppa Campioni d'Italia.
# confirmed_players - numbers of players had positive test on coronavirus

# In[ ]:


total_italian_players = 600
confirmed_players = 2
CFR_seria_A = confirmed_players / total_italian_players *100
CFR_seria_A


# let's try to estimate the actual number of cases

# In[ ]:


real_cases = CFR_seria_A * all_people_in_world
real_cases

