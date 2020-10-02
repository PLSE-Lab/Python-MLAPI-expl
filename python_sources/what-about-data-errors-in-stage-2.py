#!/usr/bin/env python
# coding: utf-8

# There are many error in the training data that we get through env.get_training_data(). Ok, no problem, we can clear and filter the bad data.
# 
# But I see also many errors in future data in env07 (you can see below). There are many extraordinary values in 'returnsOpenNextMktres10' between '2017-01-01 00:00:00+00:00' and  '2018-04-25 00:00:00+00:00', such as 2000% etc.Is env07['returnsOpenNextMktres10'] is used for leaderboard score calculation? If yes - this extraordinary values add big mistake to the score.
# 
# And what about stage 2? Should we expect similary errors in score calculation for stage 2?
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from IPython.display import display, HTML
from datetime import datetime, timedelta
from itertools import chain
from kaggle.competitions import twosigmanews

# Any results you write to the current directory are saved as output.


# In[ ]:


env = twosigmanews.make_env()


# In[ ]:




df = env._var07[(env._var07['time'] > '2017-01-01 00:00:00+00:00') & (env._var07['time'] < '2018-04-25 00:00:00+00:00')]
df = df[df['universe'] == 1]
display(pd.DataFrame(df['returnsOpenNextMktres10'].sort_values()))

