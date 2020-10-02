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


#pip install apyori


# In[ ]:


# Import Data from CSV file 
dataset = pd.read_csv('/kaggle/input/groceries/groceries - groceries.csv',sep=',')


# In[ ]:


# View the dataset
dataset.head(10)


# Here, the data represents the items a customer bought at a particular time at shopping center. Dataset contains 33 columns and 9835 columns where each row represents the number of items purchased and names of items in each respective columns. 
# 
# **NOTE: Neglect 'NAN' in columns because if a customer bought 3 items then next 29 columns will be empty (null or NAN).**
# 

# The goal here is to apply Apriori algorithm on the dataset and see the rules (support, confidence and lift). Below mentioned is the simple explainations of them:
# 
# **1) Support:**
# It is calculated to check how much popular a given item is. It is measured by the proportion of transactions in which an itemset appears. For example, there are 100 people who bought something from grocery store today, amoung those 100 people, there are 10 people who bought bananas. Hence, the support of people who bought bananas will be (10/100 = 10%).
# 
# **2) Confidence:**
# It is calculated to check how likely if item X is purchased when item Y is purchased. This is measured by the proportion of transactions with item X, in which item Y also appears. Suppose, there are 10 people who bought apple(out of 100), and from those 10 people, 6 people also bought yogurt. Hence, the confidence of (apple -> yogurt) is: (apple -> yogurt) / apple [i.e. 6/10 = 0.6].
# 
# **3) Lift:**
# It is calculated to measure how likely item Y is purchased when item X is purchased, while controlling for how popular item Y is. The formula for lift is: (lift = support (X ->Y) / (support(X) * support(Y)).
# 
# For further details please visit this link: https://www.kdnuggets.com/2016/04/association-rules-apriori-algorithm-tutorial.html
# 

# In[ ]:


# Apriori algorithm takes the list of items that were bought together as input. Hence, we need to get each row 
# as list (except 1st column and 'NAN' in the columns).
# Create a list of trasactions
transactions = []

# Add all the items from each row in one list( Neglect the 1st columns where all the items are in number (0-9))
for i in range(0, 9835):
    transactions.append([str(dataset.values[i,u]) for u in range(1, 33)])
    


# In[ ]:


# Training the Apriori Algorithms
from apyori import apriori
rules = apriori(transcations, min_support=0.0022, min_confidence=0.20, min_lift=3, min_length = 2)

# Min_support  = 3(3 times a day) * 7 (7 days a week) / 9835 = 0.0022
# Min_confidence = set it lower to get more relations between products (weak relations), if we set it high then 
# we might miss some. I have selected confidence of 0.20
# Min_lift = In order to get some relevant rules, I am setting min_lift to 3.


# In[ ]:


# Store rules in result variable
results = list(rules)

# See the items that were bought together with their support
results_list = []
for i in range(0, len(results)):
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]))


# In[ ]:


#Print results to see the common things bought together at market

