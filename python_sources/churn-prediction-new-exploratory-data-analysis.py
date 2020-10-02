#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import Image
Image("../input/churnimage/churn.PNG")


#  **Contents**
# 1.  **Understanding the data columns**
# 1.  **Data type of each column **
# 1. **Exploring individual features**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data=pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Any results you write to the current directory are saved as output.


# **1.Understanding the data columns**
# 

# In[ ]:



for i,v in enumerate(data.columns):
    print(i,v)
print("Total Number of Features {}".format(len(data.columns)))


# **2.Data type of each  column**

# In[ ]:


data.dtypes


# **3. Exploring individual features **

# In[ ]:


data.head(10)


# In[ ]:


def value_counts(column):
    plot=data[column].value_counts().plot.bar()
    print(data[column].value_counts())
    return plot


# In[ ]:


for i in data.columns[6:18]:
    value_counts(i)
    plt.show()


# **Percentage of Men and Women observations in dataset **

# In[ ]:


print("Percentage of Men  {}".format(3555/(3555+3488)*100))
print("Percentage of Women  {}".format(3488/(3555+3488)*100))


# In[ ]:


value_counts("SeniorCitizen")


# In[ ]:


value_counts("Dependents")


# In[ ]:


value_counts("Partner")


# In[ ]:


value_counts("PhoneService")


# In[ ]:


value_counts("MultipleLines")


# **Payment mode used by consumers**

# In[ ]:


data.dtypes
#value_counts("PaymentMethod")

plt.pie(x=[2365,1612,1544,1522],labels=["Electronic_Check","Mailed Check","Bank_Transfer","Credit Card"],autopct='%1.1f%%')



# **Digging deeper into Gender vs Payment mode.Just to get an insight of popular payment chosen by male and female**

# In[ ]:



payment_gender=[]
for i,v in zip(data['gender'],data['PaymentMethod']):
    a=i,v
    payment_gender.append(a)
    
    


# In[ ]:


data['Payment_Gender']=payment_gender
data['Payment_Gender'].value_counts().plot.bar()
print(data.Payment_Gender.value_counts())


# The bar chart above is a summary of Gender vs Payment mode.
# 
# * Electronic check :   Male  1195
# * Mailed Check :  Male  834
# * Bank Transfer Automatic : Female 788
# * Credit card Automatic: Male 770
# 
# The most sought after mode of transport of women is Bank Transfer automatic.

# In[ ]:


data.columns


# In[ ]:


data[['tenure','MonthlyCharges']]
plt.scatter(data['tenure'],data['MonthlyCharges'])

zero=data.loc[data['tenure']==0]

zero[['tenure','MonthlyCharges']]
zero


# In[ ]:


value_counts('Contract')


# In[ ]:




