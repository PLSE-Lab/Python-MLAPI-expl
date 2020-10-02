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
print(os.listdir("../input")

# Any results you write to the current directory are saved as output.


# In[ ]:


from IPython.display import IFrame


# **Hi everyone, this kernal is my visualization analysis for the Telco Customer Churn data. The visualization is done by Tableau.
# The following visualization have 3 sheets, each sheet contains different information. I just did part of the visualization from the data so not every column is used. **

# I created several calculated field in Tableau to make visualiztion more clear and concise. 
# Churn Customer: the number of Churn Customer
# Stayed Customer :  the number of Customer where Churn = no

# In[ ]:


IFrame("https://us-east-1.online.tableau.com/t/xyq/views/TelcoCustomerChurn/Story1?iframeSizedToWindow=true&:embed=y&:showAppBanner=false&:display_count=no&:showVizHome=no",width=1600,height=1000)


# From sheet 1, 
# first, we can see that the overall Churn rate from this data is 26.54%, we can use it as a benchmark in Churn rate comparison for specific services.
# 
# Second, on the right-hand side, we can see that the average monthly charge for churn customer is 74.44,  and for no churn customer is 61.27.  we can find that the customer who left the company 
# paid  13.17  more than stayed customer on average. The total charge showed that the customer who stayed are already spent almost  1000  more than churn customer on average.
# 
# The stacked bar chart showed the number of churn customer and stayed customer on the number of tenure month. we can clearly see that there are large number of customers left the business within first few months. we can conclude that the customer has more willingness to stay if they already use the service for long time.  For example, 61.99% of customer left the company after one month, and only 1.66% of customer who already used the service for 72 months left the business last month 
# The line chart shows the relationship between churn rate and tenure month, we can spot a clear downward trend. It proved our point further that the longer tenure month lead to lower churn rate.
# 

# On sheet 2 , from the first graph we can spot a postive relationship between tenure month and monthly charge which means that the longer customer stayed with the company, the higher they have to pay monthly. It could be caused by many reasons. The customer's new customer promotion might be expired, and the customer may add more services to their account could also cost higher monthly. On conclusion, we can reject the hypothesis that **the longer you stay, the cheaper you pay.**
# 
# **The bar charts shows the churn rate in specific services. The red percentage shows the churn rate and the dollar amount is the average monthly charge on average is the customer choose particular service.**
# 
# For example, we can find that fiber optic is expensive and it lead to really high churn rate. Furthermore, we can see that the churn rate is low with tech support but really high without tech support, but the customer pays more monthly charge on average with tech support.

# On chart 3 , 
# **The bar charts shows the churn rate in specific services. The red percentage shows the churn rate and the dollar amount is the average monthly charge on average is the customer choose particular service.**
# We can find some intersting insights from the churn rate from those bar chart. and it is depends on you to find out

# It is the first time I created a kernal so please excuse me for any mistakes. Recommendations and suggestions are very welcomed. I just want to share what I did with you guys and hopefully you can find it helpful. I really appreciate your time to look at my kernal and hope you can have a nice day.  Recommendations and suggestions are very welcomed.
# By the way, please upvote if you liked my work
