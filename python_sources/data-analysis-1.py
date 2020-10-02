#!/usr/bin/env python
# coding: utf-8

# #Here is Starter Kernel with code to kickstart!

# # Import necessary libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# # Import the dataset in the i/p directory

# In[ ]:


data = pd.read_excel('../input/covid19-india/Complete COVID-19_Report in India.xlsx')
data.head()


# Looks like a simple ordered one yet definitely needs some cleaning and preprocessing.

# In[ ]:


def data_info():
    print(data.shape)
    print("****************************************##########*************************************")
    print(data.info())


# In[ ]:


data_info()


# Well we have got 3767 rows and 19 columns and quite a lot of missing values for many features. 

# Again, a sad part is that we are not buliding any ML models for predictions, we are just analying ang going to forecast future from given dataset.So no nee to eliminate any columns. Cool stuffs!

# # Exploratory Data Analysis [EDA]

# In[ ]:


data['Detected State'].value_counts()


# In[ ]:


viz = pd.value_counts(data['Detected State'], ascending=True).plot(kind='barh',fontsize='40',title='STATE-WISE Patient Distribution', figsize=(50,100))
viz.set(xlabel='Affected_Patient_Count',ylabel='States')
viz.xaxis.label.set_size(50)
viz.yaxis.label.set_size(60)
viz.title.set_size(50)
plt.show()


# Quite a big graph! LoL!

# In[ ]:


data['Current Status'].value_counts()

