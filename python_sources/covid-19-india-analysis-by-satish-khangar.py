#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Satish Khangar  Covid-19 INDIA Analysis
get_ipython().run_line_magic('reset', '-f')
get_ipython().run_line_magic('matplotlib', 'inline')
# for numerical operations
import numpy as np
# to store and analysis data in dataframes
import pandas as pd
import seaborn as sns
# For plotting
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
# For data processing
from sklearn.preprocessing import StandardScaler
# OS related
import os
import datetime


# In[ ]:


# Display output not only of last command but all commands in a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


# Set pandas options to display results
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000


# In[ ]:


# Go to folder containing data file
#os.chdir("C:\\Users\\satish\\Desktop\\COVID-19_Project\\covid19-corona-virus-india-dataset")  ['covid19-in-india', 'covid19-india']
os.chdir("../input/covid19india")
os.listdir()            # List all files in the folder


# In[ ]:


df=pd.read_csv("tests_daily.csv")
# df = pd.read_csv("../input/covid19-india/tests_daily.csv", delimiter='\t', encoding='utf-8', low_memory=False)
    
df.updatetimestamp = df.updatetimestamp.astype('datetime64[ns]')
df.totalpositivecases = df.totalpositivecases.astype('float')

df['month'] = pd.DatetimeIndex(df['updatetimestamp']).month
df['year'] = pd.DatetimeIndex(df['updatetimestamp']).year
df


# In[ ]:


# 5.0 Draw a normal scatter plot
px.scatter(df,
          x = "updatetimestamp",
          y = "totalpositivecases",
        #  range_x=[0,85],
        #  range_y=[0,120] ,
          )


# 
# 

# In[ ]:


#  Draw a normal scatter plot

px.scatter(df,
          x = "month",
          y="totalpositivecases",
          range_x=[1,12],
          range_y=[0,20000] ,
         
           )


# In[ ]:


#  Plot scatter plot, a totalpositivecases monthwise


px.scatter(df,
          x = "month",
          y = "totalpositivecases",
          range_x=[0,6],
          range_y=[0,24000] ,
          animation_frame = "totalpositivecases",  # Animate/show scatter plot
                                     
          animation_group = "month"   # Identify which circles match which ones across
                                     #  frames
          )


# In[ ]:


#Histogram on individualstestedperconfirmedcase
px.histogram(data_frame = df, x ='individualstestedperconfirmedcase')
#plt.subplot(1,2,2)
px.histogram(data_frame = df,
                    x ='individualstestedperconfirmedcase',
                   nbins =20,
                   template="plotly_dark", # Available themes: ["plotly", "plotly_white", "plotly_dark",
                                           #     "ggplot2", "seaborn", "simple_white", "none"]
                                           # https://plotly.com/python/templates/
                   width = 10,    # in inches in interval [10, inf]
                   height = 10    # in interval [10,inf]
            )


# In[ ]:


#Monthwise total positive case
fig = px.bar(df, x='month', y='totalpositivecases')
fig.show()


# In[ ]:


#State level Analysis
dt=pd.read_csv("state_level_latest.csv")
#dt
#Remove unused row
dt.drop([0,1],axis=0, inplace = True) 
#Remove unused column
dt.drop(['statenotes'],axis=1, inplace = True) 

dt


# In[ ]:


# 5.0 Draw a normal bar plot  scatter
px.bar(dt,
          x = "state",
          y = "confirmed",
         
        #  range_x=[0,85],
        #  range_y=[0,120] ,
          )


# In[ ]:


#  Draw a normal scatter plot  for statewise confirmed case
px.scatter(dt,
          x = "statecode",
          y = "confirmed",
         
        #  range_x=[0,85],
        #  range_y=[0,120] ,
          )


# In[ ]:


#  Plot Animited scatter plot for statewise deaths case analysis 

px.scatter(dt,
          x = "state",
          y = "deaths",
         
         range_x=[0,1000],
         range_y=[0,1500] ,
          animation_frame = "statecode",  # Animate/show scatter plot
                                     
          animation_group = "state"   # Identify which circles match which ones across
                                     #  frames
          )


# In[ ]:


#  Draw a normal scatter plot  for statewise deaths case
px.scatter(dt,
          x = "statecode",
          y = "deaths",
         
          range_x=[0,30],
          range_y=[0,1200] ,
          )


# In[ ]:





# In[ ]:




