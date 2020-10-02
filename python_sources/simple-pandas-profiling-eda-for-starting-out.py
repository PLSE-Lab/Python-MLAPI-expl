#!/usr/bin/env python
# coding: utf-8

# ## A Simple Kernel of Pandas Profiles For The Datasets ##
# 
# I always like to have these on hand, so here is a simple public kernel making them available for your exploration.  I will be doing a more intensive exploratory data analysis soon, but this is just a simple bare bones kernel with processed pandas profiling reports of each of the datasets.  
# 
# These can take a very long time to process, so having them handy is very useful.
# 
# After checking out the data here, feel free to check out me [my EDA kernel here](https://www.kaggle.com/tpmeli/seasonal-decomposition-question-based-eda-of-m5)
# 
# ### Use the 'toggle details' to see more.  You can also download the html reports yourself with the html links above the report.###

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from pandas_profiling import ProfileReport
from IPython.display import HTML

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# train_sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
# sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
# calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
# submission_file = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')


# In[ ]:


#train_report = ProfileReport(train_sales, title='Sales Train Validation Profiling Report')
#train_report.to_file(output_file="train_report.html")


# ## Sell Prices Report ##

# In[ ]:


HTML(filename='/kaggle/input/m5-pandas-profiles-of-sell-prices-calendar/sell_prices_report.html')


# In[ ]:


#sell_prices_report = ProfileReport(sell_prices, title='Sell Prices Profiling Report')
#sell_prices_report.to_file(output_file="sell_prices_report.html")


# ## Calendar Report ##

# In[ ]:


HTML(filename='/kaggle/input/m5-pandas-profiles-of-sell-prices-calendar/calendar_report.html')

#calendar_report = ProfileReport(calendar, title='Calendar Profiling Report')
#calendar_report.to_file(output_file="calendar_report.html")


# ## Train Validation Report - Too Big 218 MB...##
# * If someone has a tip for how to display such a big html file, would you let me know?

# **What's coming next...**
# 
# * I'm planning on doing a full EDA and publishing that publically here.
# * I'm going to merge the information from these datasets and do another pandas report.
# * I'll publish an unsupervised exploration of these datasets.
# 
# Thanks for reading!  I hope it was helpful to you.
