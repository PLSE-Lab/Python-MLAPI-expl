#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import h2o
h2o.init()
from matplotlib import pyplot as plt


# In[ ]:


data=h2o.import_file("../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv")
data.head()


# In[ ]:


data.describe()


# In[ ]:


data.columns


# Observation: there are a lot of defaults considering dafaults should not be common 

# In[ ]:


data["default.payment.next.month"].table()


# In[ ]:


data["PAY_0"].table()


# In[ ]:


df=pd.read_csv("../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv")
df


# get all College educated people

# In[ ]:


students=df[df["EDUCATION"]==2]


# rename column for code smoothness:

# In[ ]:


df = df.rename(columns={'PAY_0': 'PAY_1'})
df


# In[ ]:


students = df[df["EDUCATION"]==2]
students


# BAR GRAPHES: payment behavior of college educated people each month

# found in discussion:https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset/discussion/34608
# 
# Answering from knowledge of the industry, not with information directly from the dataset creator: I presume that the values from -2 to 0 more precisely mean the following:
# 
# -2 = Balance paid in full and no transactions this period (we may refer to this credit card account as having been 'inactive' this period)
# 
# -1 = Balance paid in full, but account has a positive balance at end of period due to recent transactions for which payment has not yet come due
# 
# 0 = Customer paid the minimum due amount, but not the entire balance. I.e., the customer paid enough for their account to remain in good standing, but did revolve a balance

# In[ ]:


for i in range(1,7):
    plt.figure()
    colors=['#FB3C14','#4214FB','#D11ED4','#67F1BB','#DCE22A']
    p=students["PAY_"+ str(i)].value_counts().plot(kind='bar', color=colors)
    plt.title("PAY_"+ str(i))
    plt.xlabel("Payment Status")
    plt.ylabel("Frequency")
    print(p)
    


# CONCLUSION: College educated people tend to pay minimum every month

# BAR GRAPHES: overall default behvaior of College educated people

# In[ ]:


plt.figure()
colors=['#67F1BB','#DCE22A']
p=students["default.payment.next.month"].value_counts().plot(kind='bar', color=colors)
plt.title("College Educated People Defaults")
plt.xlabel("Default Status")
plt.ylabel("Frequency")
print(p)


# College educated people show not to have alot of defualts compared to non-defualts BUT those defualts take up a high percentage of all defaults

# College educated people are responsible for 50% of all defaults

# In[ ]:


s=students["default.payment.next.month"].sum()
percentage=(s/df["default.payment.next.month"].sum())*100
percentage

