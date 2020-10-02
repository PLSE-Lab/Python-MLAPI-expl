#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#My first attempt at appyling what I've learned in Data Science to a Big Dataset. 
# As first dataset, I will load the most recent one and try to do some statistics out of it. 


# In[ ]:


from pyspark import SparkContext
from pyspark.sql.session import SparkSession


# In[ ]:


sc = SparkContext()


# In[ ]:


spark = SparkSession(sc)


# In[ ]:


park_viol2018 = spark.read.csv('../input/parking-violations-issued-fiscal-year-2018.csv', header=True, inferSchema=True)


# In[ ]:


import pyspark.sql.functions as F


# In[ ]:


missing_per_col = []
for c in park_viol2018.columns:
    missing_per_col.append(park_viol2018.filter(F.isnull(park_viol2018[c])).count())


# In[ ]:


#By doing this, I was able to select which are the columns I'm interested in. 
#With a first check, I will drop columns whose nonzero elements are above a threshold, then I will 
#consider dropping rows. 


# In[ ]:


missing_per_col


# In[ ]:


#I would like to zip column names and relative number of missing values 


# In[ ]:


cols_missings = list(zip(park_viol2018.columns, missing_per_col))


# In[ ]:


cols_missings


# In[ ]:


for c, val in cols_missings:
    if val > 1.6e6:
        park_viol2018 = park_viol2018.drop(c)


# In[ ]:


park_viol2018.columns


# In[ ]:


spark.conf.set("spark.sql.execution.arrow.enabled", "true")


# In[ ]:


park_viol2018_sampled = park_viol2018.sample(withReplacement=None, fraction=0.02)


# In[ ]:


import timeit
start_time = timeit.default_timer()
park_viol2018_sampled_pd = park_viol2018_sampled.toPandas()
elapsed = timeit.default_timer() - start_time


# In[ ]:


elapsed


# In[ ]:


park_viol2018_sampled_pd.shape


# In[ ]:


park_viol2018_sampled_pd.head()


# In[ ]:


#Distribution of samples should be more or less uniform; I will plot some basic statistics out of this data


# In[ ]:


park_viol2018_sampled_pd['Registration State'].value_counts().loc[lambda x: x > 1000].plot(kind='bar')


# In[ ]:


#The distribution is pretty obvious, since the dataset comprehends data from NY!


# In[ ]:


#Let's plot the vehicle years 


# In[ ]:


park_viol2018_sampled_pd['Vehicle Year'].value_counts().loc[lambda x: x>100].plot(kind='bar', rot=45)


# In[ ]:


#The majority of the vehicle years is not reported, and apart from those values we see that newer
#vehicles dominate the scene


# In[ ]:


#Let's also first plot the distribution of the violation codes, then we'll go and see 
#what violation codes represent with the provided xls files


# In[ ]:


park_viol_code_group = park_viol2018_sampled_pd.groupby('Violation Code', as_index=True)
park_viol_code_group


# In[ ]:


most_violated_codes = park_viol_code_group['Violation Code'].count().sort_values(ascending=False).head(10)


# In[ ]:


most_violated_codes.plot(kind='bar')


# In[ ]:


#Top 10 of the violation codes! 


# In[ ]:


#Let's try to read the .xlsx containing the codes directly.


# In[ ]:


violation_codes_df = pd.read_excel('../input/ParkingViolationCodes Nov 2018.xlsx')


# In[ ]:


violation_codes_df.head()


# In[ ]:


#Great, I will try to perform a join on the most violated codes. First, let's give a look at it 


# In[ ]:


most_violated_codes.head(3)


# In[ ]:


most_violated_codes_df = pd.DataFrame(most_violated_codes)


# In[ ]:


most_violated_codes_df.head(5)


# In[ ]:


most_violated_codes_df = most_violated_codes_df.rename({'Violation Code': 'Count'},axis='columns')


# In[ ]:


most_violated_codes_df.head()


# In[ ]:


most_violated_codes_with_desc =pd.merge(most_violated_codes_df,violation_codes_df,
                                        left_on=most_violated_codes_df.index,
                                        right_on='VIOLATION CODE')


# In[ ]:


most_violated_codes_with_desc


# In[ ]:


# We see that the most common violation in this sample is the 'no parking-street cleaning', followed 
# by 'fail to display muni meter receipt'.


# In[ ]:


#As a last step of this first analysis I'll group by date and check day by day the amount of violations 


# In[ ]:


viol_by_dates_df = park_viol2018_sampled_pd.groupby('Date First Observed', as_index=False).count()


# In[ ]:


viol_by_dates_df = pd.DataFrame(viol_by_dates_df[['Date First Observed','Plate ID']]).rename({'Plate ID': 'count'})


# In[ ]:


viol_by_dates_df.head()


# In[ ]:


viol_by_dates_df = viol_by_dates_df.iloc[2:,:]


# In[ ]:


viol_by_dates_df['Date First Observed'] = pd.to_datetime(viol_by_dates_df['Date First Observed'], 
                                                        format='%Y/%m/%d')

