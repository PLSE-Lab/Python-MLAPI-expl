#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# # A Journey through Prudential
# Author: BobC
# Date: Dec 16, 2015
# 
# Inspired by https://www.kaggle.com/omarelgabry/rossmann-store-sales/a-journey-through-rossmann-stores

# In[ ]:


# Imports

#utils
import os
import pylab

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# read the datasets (only looking at train for now)
train  = pd.read_csv("../input/train.csv")

#test      = pd.read_csv("../input/test.csv")


# 
# ## Class Imbalance
# 
# Class imbalance is something to be aware of when training.

# In[ ]:


sns.countplot(x='Response',data=train,palette="husl", order = range(1,9))


# ## Missing Data
# 
# There's a lot of missing data.   Let's take a look
# 
# First, let's find the columns with missing data.  As you can see, only 1% of the rows for Medical_History_10 have values

# In[ ]:


# Let's look at the size of the train dataset

print("train:  nrows %d, ncols %d" % train.shape)


# In[ ]:


# List features with missing values

print("%20s \tCount \tPct missing" % 'Feature')
for column_name, column in train.transpose().iterrows():
    naCount = sum(column.isnull())
    if naCount > 0:
       #print column_name, naCount, "Percent missing: %f%%" % 100.*naCount/train.shape[0]
       print("%20s \t%5d  \t%2.2f%%" % (column_name, naCount, 100.*naCount/train.shape[0]))


# 
# 
# ### Plots for those variables with large amounts of missing data
# 
# 
# #### Employment History
# 

# In[ ]:


# Plot distributions for Employment_Info_4 and Employment_Info_6

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
train["Employment_Info_4"].plot(kind='hist',bins=20,xlim=(0,1),ax=axis1)
axis1.set_xlabel("Employment_Info_4")
axis1.set_ylabel("Count")
train["Employment_Info_6"].plot(kind='hist',bins=50,xlim=(0,1),ax=axis2)
axis2.set_xlabel("Employment_Info_6")
axis2.set_ylabel("Count")


# Interesting distribution for *Employment_Info_6* (above)
# 
# 
# ### Insurance History
# 
# *Insurance_History_5* has a few large outliers, otherwise most of the data is is less than 0.02.  The data appears to be quantized

# In[ ]:


# Is there anything to learn in the data quantization for Insurance_History_5?

x = min(train["Insurance_History_5"][train["Insurance_History_5"]>0])
print("Min value > 0: %e   1/(Min value > 0) %f" % (x,1./x))


# In[ ]:


# List all of the Insurance_History_5 values greater than 0.02  (Max is 1.0)

train["Insurance_History_5"][train["Insurance_History_5"]>0.02]


# In[ ]:


# Plot distribution for Insurance_History_5 with two different x-axis scalings

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
train["Insurance_History_5"].plot(kind='hist',bins=20,xlim=(0,1),ax=axis1)
axis1.set_xlabel("Insurance_History_5")
axis1.set_ylabel("Count")

train["Insurance_History_5"][train["Insurance_History_5"]<0.034].plot(kind='hist',bins=60,xlim=(0,.05),ax=axis2)
#np.log10(train["Insurance_History_5"]+1).plot(kind='hist',bins=100,ax=axis2)
axis2.set_xlabel("Insurance_History_5 (scaled X)")
axis2.set_ylabel("Count")


# In[ ]:


pylab.rcParams['figure.figsize'] = (12.0, 8.0)
fig, axisArr = plt.subplots(2,2)
train["Family_Hist_2"].plot(kind='hist',bins=100,xlim=(0,1),ax=axisArr[0,0])
axisArr[0,0].set_xlabel("Family_Hist_2")
axisArr[0,0].set_ylabel("Count")
train["Family_Hist_3"].plot(kind='hist',bins=100,xlim=(0,1),ax=axisArr[0,1])
axisArr[0,1].set_xlabel("Family_Hist_3")
axisArr[0,1].set_ylabel("Count")

train["Family_Hist_4"].plot(kind='hist',bins=100,xlim=(0,1),ax=axisArr[1,0])
axisArr[1,0].set_xlabel("Family_Hist_4")
axisArr[1,0].set_ylabel("Count")
train["Family_Hist_5"].plot(kind='hist',bins=100,xlim=(0,1),ax=axisArr[1,1])
axisArr[1,1].set_xlabel("Family_Hist_5")
axisArr[1,1].set_ylabel("Count")


# In[ ]:


# Multiplying the data by 71 turns Family_Hist_4 into integer data
# So the original data may have had a range of 0-71 (assuming the data hasn't been shifted)
train["Family_Hist_4"][1:10]*71


# 
# 
# ### Medical History
# 
# Worth noting: *Medical History features with missing values aren't normalized. The max value is 240.*

# In[ ]:


train["Medical_History_1"].plot(kind='hist',xlim=(0,250),bins=100)
plt.xlabel("Medical_History_1")
plt.ylabel("Count")


# In[ ]:


# Plot distributions for Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32

pylab.rcParams['figure.figsize'] = (12.0, 8.0)
fig, axisArr = plt.subplots(2,2)
train["Medical_History_10"].plot(kind='hist',bins=100,xlim=(0,250),ax=axisArr[0,0])
axisArr[0,0].set_xlabel("Medical_History_10")
axisArr[0,0].set_ylabel("Count")
train["Medical_History_15"].plot(kind='hist',bins=100,xlim=(0,250),ax=axisArr[0,1])
axisArr[0,1].set_xlabel("Medical_History_15")
axisArr[0,1].set_ylabel("Count")

train["Medical_History_24"].plot(kind='hist',bins=100,xlim=(0,250),ax=axisArr[1,0])
axisArr[1,0].set_xlabel("Medical_History_24")
axisArr[1,0].set_ylabel("Count")
train["Medical_History_32"].plot(kind='hist',bins=100,xlim=(0,250),ax=axisArr[1,1])
axisArr[1,1].set_xlabel("Medical_History_32")
axisArr[1,1].set_ylabel("Count")


# 
# 
# ## More Medical History Fun
# 
# The features below don't have missing values, but do have interesting distributions.  They appear to be binary, with somewhat arbitrary integer offsets and scalings.  This makes them somewhat different that the medical histories plotted above.  Those are binomial and appear to have continous values.

# In[ ]:


# Plot the distributions for Medical_History_33, Medical_History_38, Medical_History_39 and Medical_History_40

pylab.rcParams['figure.figsize'] = (12.0, 8.0)
fig, axisArr = plt.subplots(2,2)
train["Medical_History_33"].plot(kind='hist',bins=20,xlim=(0,3),ax=axisArr[0,0])
axisArr[0,0].set_xlabel("Medical_History_33")
axisArr[0,0].set_ylabel("Count")
train["Medical_History_38"].plot(kind='hist',bins=20,xlim=(0,2),ax=axisArr[0,1])
axisArr[0,1].set_xlabel("Medical_History_38")
axisArr[0,1].set_ylabel("Count")

train["Medical_History_39"].plot(kind='hist',bins=20,xlim=(0,3),ax=axisArr[1,0])
axisArr[1,0].set_xlabel("Medical_History_39")
axisArr[1,0].set_ylabel("Count")
train["Medical_History_40"].plot(kind='hist',bins=20,xlim=(0,3),ax=axisArr[1,1])
axisArr[1,1].set_xlabel("Medical_History_40")
axisArr[1,1].set_ylabel("Count")


# 
# 
# ## Product Info

# In[ ]:


# Product_Info_2 is categorical with 19 categories

sns.countplot(x='Product_Info_2', data=train, 
              order=['A1','A2','A3','A4','A5','A6','A7','A8',
                     'B1','B2',
                     'C1','C2','C3','C4',
                     'D1','D2','D3','D4',
                     'E1'])


# The following plots are of the same data broken out for each Response.  
# 
# Be careful when looking at these plots.  The response data doesn't have a uniform distribution, and these plots haven't been normalized for either the response distribution, or the Product_Info_2 distribution.  

# In[ ]:


pylab.rcParams['figure.figsize'] = (10.0, 14.0)
f, axisarr = plt.subplots(4, 2)
for r in range(1,9):
    axs = axisarr[int((r-1)/2),(r-1)%2]
    sns.countplot(x='Product_Info_2', data=train[train["Response"]==r], 
              order=['A1','A2','A3','A4','A5','A6','A7','A8',
                     'B1','B2',
                     'C1','C2','C3','C4',
                     'D1','D2','D3','D4',
                     'E1'],ax=axs)
    axs.set_ylabel('Count')
    axs.set_xlabel('Response: '+str(r))


# 
# 
# ## Age vs Response

# In[ ]:


# Look at Responses as a function of age.   It's not surprising that responses differ as a function of age.

pylab.rcParams['figure.figsize'] = (10.0, 14.0)
f, axisarr = plt.subplots(4, 2)
for r in range(1,9):
    axs = axisarr[int((r-1)/2),(r-1)%2]
    train["Ins_Age"][train["Response"]==r].plot(kind='hist',bins=50,xlim=(0,1),ax=axs)
    axs.set_ylabel('Count')
    axs.set_xlabel('Response: '+str(r))


# In[ ]:




