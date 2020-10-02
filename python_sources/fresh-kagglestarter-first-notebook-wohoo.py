#!/usr/bin/env python
# coding: utf-8

# #Introduction:
# Beginner Notebook with step by step documentation. The goal is to transition from just online courses to real world problems.
# 
# #Remarks:
# * I have no educational background in CS
# * Feel free to contribute for a better learning experience
# * Good luck everyone

# #Step 1: Import all the things

# In[ ]:


import numpy as np #linAlgebra
import pandas as pd #dataFrames ans data manipulation
import sklearn as sk # basic ML; Models will come directly from sklearn 
import matplotlib as mpl #basic import for dataviz
import matplotlib.pyplot as plt #convenient way to use pyplot 
import seaborn as sns #extended dataviz
from datetime import datetime #conerting the string date to a normal date format

get_ipython().run_line_magic('matplotlib', 'inline')
#more things will come while I reach the different steps 

#Good idea from SKS to not let my notebook explode
pd.options.display.max_columns = 999


# #Step 2: get a first idea about the dataset
# I will use SRKSimple Exploration Notebook - Zillow Prize as a string point

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# ##Import all datasets

# In[ ]:


#properties_2016_df = pd.read_csv("../input/properties_2016.csv")
#sample_submission_df = pd.read_csv("../input/sample_submission.csv")
train_2016_df = pd.read_csv("../input/train_2016.csv", parse_dates = ["transactiondate"])
#Datadictionary is a file where the different datapoints are described
#Got an error when importing the first two files. I think it has something to do with the memory. Ill just start with the training set


# In[ ]:


train_2016_df.shape


# In[ ]:


train_2016_df.head()


# ##What do we see:
# 1. The parcelid is a unique(? we'll see) ID for a certain object
# 2. The logerror is the target variable for this competition. It is defined as followed: logerror= log(Zestimate) - log(SalesPrice)
# 3. The transactiondate is obviously the date of this transaction
# 
# ## Now let us explore the dataset in three major bullets:
# ###Bullet 1: parcelid
# * How many parcelid's do we have (hint 90811 rows)
# * Is the parcelid a unique value ? If not, how often are parcel id's been "traded"
# 
# ###Bullet 2: logerror
# * Whats the range of the logerror
# * Are there any outliers
# * How is the distribution among the logerror
# 
# ###Bullet 3: transactiondate
# * How many Transaction have been made in certain time periods
# * How is the distribution of the logerror among the different time periods

# ##Bullet 1: parcelid
# ###Answers:
# * How many parcelid's do we have (hint 90811 rows)
# 1. 90811 parcelid's in the training set
# * Is the parcelid a unique value ? If not, how often are parcel id's been "traded"
# 1. Most of them are unique, 127 are listed 2 times and only 1 is listed three times. We'll investigate it later when we have an understand of all the different files

# In[ ]:


parcelid_duplicates = train_2016_df.groupby("parcelid").size().reset_index().rename(columns={0:'count'})
parcelid_duplicates["count"].value_counts()


# ##Bullet 2: logerror
# ###Answers:
# * Whats the range of the logerror
# 1. -4.605 -4.737
# * Are there any outliers
# 1. yes the min max value, I will cut everything what is bigger than 6 sigma 
# * How is the distribution among the logerror
# 1. pretty much normal distribution

# In[ ]:


print(train_2016_df["logerror"].max())
print(train_2016_df["logerror"].min())


# In[ ]:


train_2016_df_sorted = train_2016_df.sort_values(by = "logerror")
train_2016_df_sorted.head(10)


# In[ ]:


train_2016_df_sorted.tail(10)


# In[ ]:


#sorted scatterplot
x = range(train_2016_df_sorted.shape[0])
y = train_2016_df_sorted["logerror"]
plt.scatter(x,y)
plt.show()


# In[ ]:


logerror_duplicates = train_2016_df.groupby("logerror").size().reset_index().rename(columns={0:'count'})
#print(logerror_duplicates["count"].value_counts(bins = 20))
x = logerror_duplicates["logerror"]
y = logerror_duplicates["count"]
plt.hist(logerror_duplicates["logerror"], bins = 20)
plt.show()


# ###Bullet 3: transactiondate
# ###Answers:
# * How many Transaction have been made in certain time periods
# * How is the distribution of the logerror among the different time periods

# Before Starting I need to change the transactiondate from string to datetime

# In[ ]:


train_2016_df["transactiondate"].iloc[2].dtype


# In[ ]:


x = train_2016_df["transactiondate"]
y = train_2016_df["logerror"]
plt.scatter(x,y)
plt.show()


# In[ ]:




