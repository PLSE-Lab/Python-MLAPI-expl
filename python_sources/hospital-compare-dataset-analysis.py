#!/usr/bin/env python
# coding: utf-8

# Hospital Compare is a consumer-oriented website that provides information on how well hospitals provide recommended care to their patients. 
# Hospital Compare allows consumers to select multiple hospitals and directly compare performance measure information related to heart attack, heart failure, pneumonia, surgery and other conditions. These results are organized by:
# *  1. General information
# * 1. Survey of patients' experiences
# * 1. Timely & effective care
# * 1. Complications
# * 1. Readmissions & deaths
# * 1. Use of medical imaging
# * 1. Payment & value of care
# Hospital Compare data contains information on over 100 measures and more than 4000 hospitals.
# The method assigns each hospital between 1 and 5 stars on the basis of the hospitals overall performance on selected measures. 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataset = pd.read_csv('../input/HospInfo.csv')
dataset.head()


# **From the below information, we can see that there are a lot of missing values for the footnote measure and thus we can get rid of those columns**

# In[ ]:


dataset.info()


# In[ ]:


dataset.drop(['Hospital overall rating footnote','Mortality national comparison footnote','Safety of care national comparison footnote','Readmission national comparison footnote','Patient experience national comparison footnote','Effectiveness of care national comparison footnote','Timeliness of care national comparison footnote','Efficient use of medical imaging national comparison footnote'],axis=1,inplace=True)


# In[ ]:


dataset.info()


# **Lets check out the total number of hospitals in each state**

# In[ ]:


Total_state = pd.value_counts(dataset['State'])
Total_state = pd.DataFrame(Total_state)
Total_state = Total_state.reset_index()


# In[ ]:


Total_state.columns = ['State', 'Number of Hospitals']


# In[ ]:


dims = (10, 10)
fig, ax = plt.subplots(figsize=dims)
ax = sns.barplot(x = 'Number of Hospitals', y = 'State', data = Total_state)
ax.set(xlabel = 'Number of Hospitals', ylabel = 'States')
ax.set_title('Number of Hospitals per State')


# **From the above barplot, we can see that Texas has the maximum number of hospitals along with other top states in the Top 10! We can notice that American Samoa (AS), District of Columbia(DC), Guam(GU), Virgin Islands(VI), Northern Mariana Islands(MP) have very low number of hospitals and thus the state of healthcare in these states must be very poor!**

# ** Now Let us have a look at what are the different types of hospitals and their count in USA**

# In[ ]:


Hospital_Type_df = pd.value_counts(dataset['Hospital Type'])
Hospital_Type_df = pd.DataFrame(Hospital_Type_df)
Hospital_Type_df = Hospital_Type_df.reset_index()
Hospital_Type_df.columns = ['Hospital Type', 'Number of Hospitals']


# In[ ]:


ax = sns.barplot(x = 'Hospital Type', y= 'Number of Hospitals', data = Hospital_Type_df)
ax.set(xlabel = 'Type of Hospital', ylabel = 'Number of Hospitals')
ax.set_title('Count of the different Types of Hospitals(Acute/Critical/Childrens)')


# From the above barplot, we can see that USA has Acute Care Hospitals in maximum number as compared to Critical Access and Children's Hospital. 

# **It would be interesting to find out who has the maximum ownerships on the hospitals in USA? Does the government own majority of the hospitals or are they owned by Private organizations! Let's find out!**

# In[ ]:


Hospital_owner = pd.value_counts(dataset['Hospital Ownership'])
Hospital_owner = pd.DataFrame(Hospital_owner)
Hospital_owner = Hospital_owner.reset_index()
Hospital_owner.columns = ['Hospital Ownership', 'Number of Hospitals']


# In[ ]:


dims = (10, 10)
fig, ax = plt.subplots(figsize=dims)
ax = sns.barplot(y = 'Hospital Ownership', x= 'Number of Hospitals', data = Hospital_owner)
ax.set(xlabel = 'Hospital Ownership', ylabel = 'Number of Hospitals')
ax.set_title('Count of the different Types of Hospital Ownership')


# **From the above graph , we can see that majority of the hospitals i.e more than half of the hospitals in the US is owned by voluntary non-profit organizations. We can say, that around 800 hospitals are owned privately!**

# **Next, I am curious to know what is the average rating given to the hospitals ? Are there more 1-star hospital which means that the healthcare facility is very poor! ORR are there more number of 5-star rated hospitals, which means that the healthcare facility is excellent (which is doubt) ! Let's check out. **

# In[ ]:


dataset['Hospital overall rating'].unique()


# **We can see that there are few hospitals whose ratings are "Not Available". We can remove those rows from our new dataset for Hospital rating before counting the number of hospitals for each rating** 

# In[ ]:


Hospital_rating = dataset.drop(dataset[dataset['Hospital overall rating']=='Not Available'].index)
Hospital_rating['Hospital overall rating'].unique()


# In[ ]:


Hospital_rating = pd.value_counts(Hospital_rating['Hospital overall rating'])
Hospital_rating = pd.DataFrame(Hospital_rating)
Hospital_rating = Hospital_rating.reset_index()
Hospital_rating.columns = ['Hospital Rating', 'Number of Hospitals']


# In[ ]:


dims = (10, 5)
fig, ax = plt.subplots(figsize=dims)
ax = sns.barplot(x = 'Hospital Rating', y = 'Number of Hospitals', palette="BuGn_d",data = Hospital_rating)
ax.set(xlabel = 'Hospital Rating', ylabel = 'Number of Hospitals')


# **Here, we can see that there are majority hospitals with 3-star rating. There are very little hospitals with extreme ratings i.e 1 or 5**

# 
