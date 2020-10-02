#!/usr/bin/env python
# coding: utf-8

# # Hi all
# 
# ### This is my first Notebook.
# 
# I am quite new to the industry and really looking forward to learning as much as I can.
# 
# **Please let me know if you have any comments on how things can be done better. Im super open to any feedback. ** **Seriously, anything.**
# 
# Hope you found something interesting out of this :) 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns 


# In[ ]:


#Import Data
raw_data = pd.read_csv('../input/Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv', index_col=0, parse_dates=True)
#display(raw_data)
raw_data.describe()


# ## Maxes look fine but Mins seem to be corrupted
# Plotting the distribution to get a better look.

# In[ ]:


# Histogram plot of features

plt.figure(figsize=(10,6)) 
# Add title 
plt.title("Hs") 
sns.distplot(a=raw_data['Hs'], kde=False) 

plt.figure(figsize=(10,6))
plt.title("Hmaz") 
sns.distplot(a=raw_data['Hmax'], kde=False) 

plt.figure(figsize=(10,6))
plt.title("Tz") 
sns.distplot(a=raw_data['Tz'], kde=False) 

plt.figure(figsize=(10,6))
plt.title("TP") 
sns.distplot(a=raw_data['Tp'], kde=False) 

plt.figure(figsize=(10,6))
plt.title("SST") 
sns.distplot(a=raw_data['SST'], kde=False) 

plt.figure(figsize=(10,6))
plt.title("Peak Direction") 
sns.distplot(a=raw_data['Peak Direction'], kde=False)  


# ## Outliers skewing data
# 
# Need to remove outliers in order to get better plots

# In[ ]:


# Remove Outliers
from scipy import stats
data= raw_data[(np.abs(stats.zscore(raw_data)) < 3).all(axis=1)]
data.describe()


# ## That's better!
# 
# Renaming the data now for ease of reference and understanding

# In[ ]:


#Rename Data
data = data.rename({'Hs': 'Average_Significant_WH', 'Hmax': 'Maximum_WH', 'Tz': 'ZeroUpcrossing_WP', 'Tp': 'PeakEnegery_WP', 'SST': 'Sea_Temp'}, axis=1)  # new method
data.index.names = ['Date']

#data.head()


# ## Replotting the distrubtion

# In[ ]:


#Different way to plot the distribution

plt.figure(figsize=(10,6)) 
# Add title 
plt.title("Sea_Temp") 
sns.kdeplot(data=data['Sea_Temp'], label="Sea_Temp", shade=True)


plt.figure(figsize=(10,6))
plt.title("Average_Significant_WH") 
sns.kdeplot(data=data['Average_Significant_WH'], label="Average_Significant_WH", shade=True)


plt.figure(figsize=(10,6))
plt.title("Maximum_WH") 
sns.kdeplot(data=data['Maximum_WH'], label="Maximum_WH", shade=True)


plt.figure(figsize=(10,6))
plt.title("ZeroUpcrossing_WP") 
sns.kdeplot(data=data['ZeroUpcrossing_WP'], label="ZeroUpcrossing_WP", shade=True)


plt.figure(figsize=(10,6))
plt.title("PeakEnegery_WP") 
sns.kdeplot(data=data['PeakEnegery_WP'], label="PeakEnegery_WP", shade=True)


plt.figure(figsize=(10,6))
plt.title("Peak Direction") 
sns.kdeplot(data=data['Peak Direction'], label="Peak Direction", shade=True)


# ## A much smoother looking distrubtion graph
# 
# My analysis at this point is that Average and Max Wave Height might be related so I plotted them together to see.

# In[ ]:


sns.jointplot(x=data['Average_Significant_WH'], y=data['Maximum_WH'], kind="kde", height = 8 , scale = 1) 


# ## Seems to be a very linear relationship

# In[ ]:


# Experimenting with heat plots. 
# Interesting is the Sea_Temp seems to have the same relation to PeakEnergy as it does Peak Direction.

correlation_matrix = data.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(correlation_matrix,vmax=0.8,square = True, annot = True)
plt.show()


# ## Confirming the WH relationship
# 
# Another interesting extraction is that Sea_Temp seems to affect Peak_Energy and Peak Direction the same.
# Is this true?

# In[ ]:


#Pairplot is a nice abstraction of most of the work done so far

sns.pairplot(data)


# ## There doesnt seem to be any significant differences from what we have learned so far
# 
# There might be some new relationships such as how the Peak_Enegery approaches a mid point as Average Significant_WH increaes but im not exactly convinced of this.
# 
# Any thoughts?
# 

# ## Time for time plots

# In[ ]:


cols_plot = ['Average_Significant_WH', 'PeakEnegery_WP', 'Sea_Temp', 'Maximum_WH', 'ZeroUpcrossing_WP','Peak Direction',]
axes = data[cols_plot].plot(marker=',', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)


# ## At this point the data looks a bit crazy for the 2019 period 
# 
# Im not really sure what this means. My inital thoughts would be if I was builinding a model it might be best to cut this data out as it is quite messy.
# 
# Is this a correct assumption?
# 
# Im also not sure if I like this graph type, any better timeplots?

# ## Now I think I have a good grasp on the data. Time to pull out some useful analysis
# 
# *I would love some suggestions on any other interesting information I can pull out.*

# ### What month of the year is the warmest vs coolest?

# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(14,6))
# Add title
plt.title("Sea Temp")
sns.lineplot(x=data.index.month, y="Sea_Temp", data=data)


# In[ ]:


#Simple max search shows Jan meaning more analysis is needed
print(data.loc[data['Sea_Temp'].idxmax()])
print('\n',data.loc[data['Sea_Temp'].idxmin()])

#Create Summer and Winter Data sets and get averages

Jan = data[(data.index.month.isin([1]))]
Feb = data[(data.index.month.isin([2]))]
July = data[(data.index.month.isin([7]))]
Aug = data[(data.index.month.isin([8]))]

JanAv = Jan['Sea_Temp'].mean()
FebAv = Feb['Sea_Temp'].mean()
JulyAv = July['Sea_Temp'].mean()
AugAv = Aug['Sea_Temp'].mean()

print("\nJan:",JanAv,"\nFeb:",FebAv,"\nJuly:",JulyAv,"\nAug:",AugAv)


# A max search returns the warmest entry in January.
# However, the lineplot(which displays a summary of all the entires) seems to be showing Feburary as the warmest month. A further analysis is conducted to find the answer.
# 
# It seems clear that Aug is the coolest month but July is considered just incase.
# 
# The numbers support:
# 
# **Warmest = Feb**
# 
# **Coolest = Aug**
# 
# This shows that I should of just trusted the lineplot XD

# ### How big are the waves in summer vs winter?

# In[ ]:


plt.figure(figsize=(14,6))
# Add title
plt.title("Av. WH")
sns.lineplot(x=data.index.month, y="Average_Significant_WH", data=data)

plt.figure(figsize=(14,6))
# Add title
plt.title("Max WH")
sns.lineplot(x=data.index.month, y="Maximum_WH", data=data)


# In[ ]:


#Create Summer and Winter Data sets and get averages

Summer = data[(data.index.month.isin([1,2,12]))]
Winter = data[(data.index.month.isin([6,7,8]))]

SummerMax = Summer['Maximum_WH'].mean()
SummerSig = Summer['Average_Significant_WH'].mean()

WinterMax = Winter['Maximum_WH'].mean()
WinterSig = Winter['Average_Significant_WH'].mean()

print("Summer:",SummerMax,SummerSig,"\nWinter:",WinterMax,WinterSig)


# Average Max Wave Height in Summer = 2.2
# 
# Average Max Wave Height in Winter = 1.7
# 
# Average Significant Wave Heigh in Summer = 1.3
# 
# Average Significant Wave Heigh in Winter = 1.0
# 

# ## Considering future work with more relational plots to find further insights
# 
# Looking to use more KDE plots and Autocorrelation

# In[ ]:


# from pandas.plotting import autocorrelation_plot
# autocorrelation_plot(data['PeakEnegery_WP'])

# sns.kdeplot(data['PeakEnegery_WP'], data['Peak Direction'])


# #### As you can see. My analysis at answering the questions is very rudimental.
# #### I would love to see more tools and suggestions to help really break down these questions.
# 
# ### Looking forward to hearing from you all!  :)
