#!/usr/bin/env python
# coding: utf-8

# ## General Analysis of the Video Game Industry ##
# 
# I want to do a general analysis of the dataset, evolution of the industry, correlation between variables and some interesting plots.
# 
# Thanks for stop by!
# 
# **This Kernel is in progress, any feedback is welcome!**

# In[ ]:


#Import libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Let's import the dataset
vg = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')
#Check shape of the table
print(vg.shape)


# Umh! Ok... That's interesting.  I can see that there are some columns like Critic_Score and Critic_Count that have NaN values. I guess we have to do some cleaning!

# In[ ]:


#Let's take a look at the data
vg.head(5)


# Umh! Ok... That's interesting.  I can see that there are some columns like Critic_Score and Critic_Count that have NaN values. I guess we have to do some cleaning!

# In[ ]:


#Second Look at the Data =)
vg.describe()


# In[ ]:


#Check for NaN Values on the Dataset
vg.isnull().any()


# In[ ]:


#Check for format on Columns
vg.info()


# We can see that User_Score is an object when it should be a Float.

# In[ ]:


#Best Selling Title
#Calculate Max Value for Global Sales
vg["Global_Sales"].max()
#Get Value on Global Sales and Show Column Name for it
vg[vg["Global_Sales"]== 82.530000000000001]["Name"]


# In[ ]:


# Max Selling by Year
vg.groupby(["Year_of_Release"])["Global_Sales"].max()
#How can I get the title too?


# In[ ]:


#Correlation between Rating and Revenue
vg[['Global_Sales','Critic_Score']].corr() 
#There is not correlation between score and sells


# In[ ]:


#How many Total Sells by Year
Total_Sells_By_Year = vg.groupby(["Year_of_Release"]).sum()["Global_Sales"]
Total_Sells_By_Year


# In[ ]:





# In[ ]:




