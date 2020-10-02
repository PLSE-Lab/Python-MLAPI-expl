#!/usr/bin/env python
# coding: utf-8

# # H1B Visa dataset Analysis

# _The H1B Visa data set contains data regarding acceptance of applicants in a particular Year and States/places (These are the data we are analyzing)._
# 

# ### Libraries and Packages required
# 	_In order to analyze the data set, we have to import the following libraries.(For the current github code)_

# In[ ]:


import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
from IPython.display import HTML,SVG
import seaborn as sns
import numpy as np 


# _To analyse a data, we have to read the cvs file, which can be done using pandas._
# _Below snipet will help to understand, how we can read a csv file._

# In[ ]:


h1b = pd.read_csv("../input/h1b_kaggle.csv")


# #### **Analyzing the columns and rows using the Apis head and tail**
# 

# In[ ]:


h1b.head()


# In[ ]:


# Similar output as above, but the data is fetched from the bottom i.e. last rows.
h1b.tail()


# #### **Deleting unwanted columns for Analysis**

# In[ ]:


h1b = h1b.iloc[:,1:]
h1b = h1b.iloc[:, :8]


# #### **Analysing if there are lot of NAN's**

# In[ ]:


# With Nan 
h1b["CASE_STATUS"].value_counts()


# In[ ]:


# Analyse all the case status except for the NAN
h1b = h1b.dropna()

# Taking only one column of certified status
h1b["CASE_STATUS"].value_counts()


# **Note :** _Since there is no much diference in the data,droping the NaNs_

# 
# #### **Segregating data for each year based on the outcome of each application**
# 
# 
# _Taking only the columns required into the year data frame and then filtering it, based on the status._

# In[ ]:


# Cleaning the data for Analysing Approvals and rejections
year = h1b.drop(h1b.columns[[1,2,3,4,5,7]], axis=1)
year = year[year["CASE_STATUS"] == "CERTIFIED"]
Certified_per_year = year["YEAR"].value_counts()


# In[ ]:


year = h1b.drop(h1b.columns[[1,2,3,4,5,7]], axis=1)
year = year[year["CASE_STATUS"] == "DENIED"]
Denied_per_year = year["YEAR"].value_counts()


# In[ ]:


year = h1b.drop(h1b.columns[[1,2,3,4,5,7]], axis=1)
year = year[year["CASE_STATUS"] == "CERTIFIED-WITHDRAWN"]
CW_per_year = year["YEAR"].value_counts()


# In[ ]:


year = h1b.drop(h1b.columns[[1,2,3,4,5,7]], axis=1)
year = year[year["CASE_STATUS"] == "WITHDRAWN"]
W_per_year = year["YEAR"].value_counts()


# In[ ]:


year = h1b.drop(h1b.columns[[1,2,3,4,5,7]], axis=1)
year = year[year["CASE_STATUS"] == "PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED"]
P_per_year = year["YEAR"].value_counts()


# In[ ]:


year = h1b.drop(h1b.columns[[1,2,3,4,5,7]], axis=1)
year = year[year["CASE_STATUS"] == "REJECTED"]
R_per_year = year["YEAR"].value_counts()


# In[ ]:


year = h1b.drop(h1b.columns[[1,2,3,4,5,7]], axis=1)
year = year[year["CASE_STATUS"] == "INVALIDATED"]
I_per_year = year["YEAR"].value_counts()


# **Note :** _Similar operation done for each status_

# **_Ploting scatter plot based on the status for each year._**

# In[ ]:


labels1 = 'INVALIDATED', 'PENDING', 'WITHDRAWN','CERTIFIED-WITHDRAWN', 'REJECTED','DENIED','CERTIFIED'

plt.plot(I_per_year,'kh', P_per_year,'bo', W_per_year,'y*', CW_per_year,'g^',R_per_year,'cs',Denied_per_year,'m+',Certified_per_year,'rp')
plt.legend(('INVALIDATED', 'PENDING', 'WITHDRAWN','CERTIFIED-WITHDRAWN', 'REJECTED','DENIED','CERTIFIED'))
plt.show()


# 
# 
# ### **Analysing the application status based on states**
# 
# #### **Seperating state and city name from the column worksite**
# 

# In[ ]:


h1b1 = []
h1 = []
for i in h1b["WORKSITE"]:
    h1 = i.split(',')
    h1b1.append(h1[1])


# #### **Converting it into data frame**
# 
# _In order to analyse the data, we have to convert the list to a data frame._

# In[ ]:


df = pd.DataFrame({'col':h1b1})
df['col'].value_counts()


# #### **Concating two data frame**
# 
# _Taking both the data frame required into a list and then by using pandas concatinating two together._

# In[ ]:


data_frame = [h1b,df]
data_frame = pd.concat(data_frame, axis=1)


# #### **Segregating or cleaning data for analysis**

# In[ ]:


# Certified by states
data_frame = data_frame.dropna()
states = data_frame.drop(data_frame.columns[[1,2,3,4,5,6,7]], axis=1)


# 
# **_Filtering the data frame based on the status of each application and states_**

# In[ ]:


certified_states = states[states["CASE_STATUS"] == "CERTIFIED"]
denied_states = states[states["CASE_STATUS"] == "DENIED"]
cw_states = states[states["CASE_STATUS"] == "CERTIFIED-WITHDRAWN"]
w_states = states[states["CASE_STATUS"] == "WITHDRAWN"]


# 
# #### **Ploting Graph**
# 
# **_Ploting line plot for the applications status based on states_**
# > 

# 
# _Sorting the x-axis i.e the states for proper alignment of data_

# In[ ]:


h = sorted(set(h1b1))


# _Plot graph for all the states_

# In[ ]:


objects = h

y_pos = np.arange(len(objects))

performance0 = denied_states['col'].value_counts().sort_index()
performance1 = certified_states['col'].value_counts().sort_index()
performance2 = cw_states['col'].value_counts().sort_index()
performance3 = w_states['col'].value_counts().sort_index()


plt.plot(y_pos, performance0, label = 'Denied')
plt.plot(y_pos, performance1, label = 'Certified')
plt.plot(y_pos, performance2, label = 'Certified-Withdrawn')
plt.plot(y_pos, performance3, label = 'Withdrawn')

plt.xticks(y_pos, objects,rotation=90)
plt.xlabel('States')
plt.ylabel('Applications')
plt.title('No. of Applicants status of H1B Visa based on states')
plt.legend()
plt.show()


# ### **Conclusion **
# 
# Using the data in the database, the analysis helped to understand,
# how many individulas where selected each year and also, which states are where
# most individuals are selected throughout these years(2011-2016).
# 
# 
# 
# 
# ### **Github**
# 
# https://github.com/jvargh81/H1B-Visa_analysis
# 
# 
# ### **Author**
# **Jerrin Joe Varghese**
