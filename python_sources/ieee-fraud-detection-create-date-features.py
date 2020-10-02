#!/usr/bin/env python
# coding: utf-8

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

# Any results you write to the current directory are saved as output.


# In this kernel, I am trying to create a new features using the "TransactionDT", please upvote if you find it is useful for you~
# The TransactionDT feature is a timedelta from a given reference datetime (not an actual timestamp).
# 
# 

# In[ ]:


import os
import math
from datetime import datetime
import warnings
warnings.simplefilter("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

import seaborn as sns
color = sns.color_palette()

import dexplot as dxp


# 

# 
