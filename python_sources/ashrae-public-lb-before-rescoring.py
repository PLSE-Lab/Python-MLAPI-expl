#!/usr/bin/env python
# coding: utf-8

# ### This is the ASHRAE public leaderboard scores before rescoring
# 
# I only saved the first 138 pages of the list.
# 
# - Total number of teams was 3675
# - Gold medal zone was 1-17
# - Silver medal zone was 18-183 
# - Bronze medal zone was 184-367 
# 
# In the case of 3594 teams listed in final_leaderboard_order.csv my estimate is:-
# 
# - Total number of teams was 3594
# - Gold medal zone was 1-17
# - Silver medal zone was 18-180
# - Bronze medal zone was 181-360 
# 
# ### Please upvote if you find it useful

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


print(os.listdir("../input/ashrae-public-lb-b4-rescoring/"))


# In[ ]:


get_ipython().system("cp '../input/ashrae-public-lb-b4-rescoring/ASHRAE_Public_LB_onClosing_Gold_0-17_Silver_18-183_Bronze_184-367.pdf' .")

get_ipython().system('ls -la')

