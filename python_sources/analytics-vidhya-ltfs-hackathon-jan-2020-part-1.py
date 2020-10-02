#!/usr/bin/env python
# coding: utf-8

# *THIS NOTEBOOK IS FIRST IN SERIES OF THREE NOTEBOOKS *
# *     *FIRST ONE IS FOR DATA VISUALIZATION *
# *     *SECOND ONE IS FOR FEATURE GENERATION *
# *     *THIRD ONE IS FOR MODELLING *
#     
# 
#  

# # LTFS Data Science FinHack 2
# LTFS receives a lot of requests for its various finance offerings that include housing loan, two-wheeler loan, real estate financing and micro loans. The number of applications received is something that varies a lot with season. Going through these applications is a manual process and is tedious. Accurately forecasting the number of cases received can help with resource and manpower management resulting into quick response on applications and more efficient processing.
#     
#     
#  # Problem Statement
#  
#  You have been appointed with the task of forecasting daily cases for next 3 months for 2 different business segments aggregated at the country level keeping in consideration the following major Indian festivals (inclusive but not exhaustive list): Diwali, Dussehra, Ganesh Chaturthi, Navratri, Holi etc. (You are free to use any publicly available open source external datasets). Some other examples could be:
#      Weather Macroeconomic variables Note that the external dataset must belong to a reliable source.
#     Data Dictionary The train data has been provided in the following way:
#     For business segment 1, historical data has been made available at branch ID level For business segment 2, historical data has been made available at State level.
#     Train File Variable Definition application_date Date of application segment Business Segment (1/2) branch_id Anonymised id for branch at which application was received state State in which application was received (Karnataka, MP etc.) zone Zone of state in which application was received (Central, East etc.) case_count (Target) Number of cases/applications received
#     Test File Forecasting needs to be done at country level for the dates provided in test set for each segment.
#     Variable Definition id Unique id for each sample in test set application_date Date of application segment Business Segment (1/2)
#     
#     
# # Evaluation
# 
#  Evaluation Metric The evaluation metric for scoring the forecasts is *MAPE (Mean Absolute Percentage Error) M with the formula:
#  Where At is the actual value and Ft is the forecast value.
#  The Final score is calculated using MAPE for both the segments using the formula
#     
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


sample_submission = pd.read_csv("/kaggle/input/analytics-vidhya-ltfs-2/sample_submission.csv", parse_dates=['application_date'])
train = pd.read_csv("/kaggle/input/analytics-vidhya-ltfs-2/train.csv",parse_dates=['application_date'])
test = pd.read_csv("/kaggle/input/ltfs-2/test_1eLl9Yf.csv", parse_dates=['application_date'])


# ## Visualizing the dataset

# In[ ]:


# CHECKING THE SHAPES OF THE FILES

train.shape,test.shape,sample_submission.shape


# In[ ]:


# LOOKING AT FIRST 5 ROWS OF TRAIN AND TEST FILES

train.head() , test.head() 


# CLEARLY TEST HAS FEWER COLUMNS THAN THE TRAIN FILE.

# In[ ]:


# CHECKING FOR MISSING VALUES AND COLUMN TYPES

train.info(), test.info()


# TRAIN FILE HAS MISSING VALUES IN 'BRANCH_ID' AND 'ZONE' COLUMN
# 
# TEST FILE HAS NO MISSING VALUES
# 

# In[ ]:


# We can also visualize the missing values using the following command: 

train.isna().sum(), test.isna().sum()


# 
# THIS CLEARLY SHOWS THAT THERE ARE 13804 MISSING VALUES IN THE 'ZONE' AND "BRANCH_ID" COLUMN

# In[ ]:


# CHECKING FOR UNIQUE COLUMN VALUES

train.nunique(), test.nunique()


# THERE ARE ONLY TWO SEGMENTS : 1 AND 2
# 
# WE HAVE ID COLUMN IN THE TEST DATASET , WE CAN DROP IT FOR OUR PREDICTION

# In[ ]:


import holidays


# In[ ]:


test.drop(['id'], axis=1, inplace=True)
train = train.sort_values('application_date').reset_index(drop = True)
test = test.sort_values('application_date').reset_index(drop = True)


# In[ ]:


train.application_date.min(), train.application_date.max()


# TRAIN DATASET IS BETWEEN 1ST APRIL 2017 TO 23 JULY 2019

# In[ ]:


test.application_date.min(), test.application_date.max()


# TEST DATASET IS BETWEEN 6 JULY 2019 TO 24 OCTOBER 2019. THERE MAY BE SOME OVERLAP OF TRAIN AND TEST DATASET.

# In[ ]:


train.groupby(['application_date','segment','branch_id']).mean().reset_index()


# In[ ]:


train_1=train[train['segment']==1].groupby(['application_date']).sum().reset_index()[['application_date','case_count']].sort_values('application_date').set_index('application_date')
train_2=train[train['segment']==2].groupby(['application_date']).sum().reset_index()[['application_date','case_count']].sort_values('application_date').set_index('application_date')
test_1=test[test['segment']==1][['application_date']].sort_values('application_date').set_index('application_date')
test_2=test[test['segment']==2][['application_date']].sort_values('application_date').set_index('application_date')


# In[ ]:


train_1.tail(10)


# TRAIN SET WITH SEGMENT-1 HAS VALUES TILL DATE  5 JULY 2019

# In[ ]:


train_2.tail(10)


# TRAIN DATA WITH SEGMENT-2 HAS VALUES TILL DATE 23 JULY 2019
# 

# In[ ]:


train_1.plot(style='.', figsize=(15,5), title='Train_data_segment 1')


# CLEARLY THERE ARE OUTLIERS IN THE DATASET. WE WILL SELECT ONLY DATA WITHOUT OUTLIERS.

# In[ ]:


#train_1.boxplot(by ='day', column =['total_bill'], grid = False)

train_1.boxplot( column =['case_count'], grid = True)


# In[ ]:


train_1.describe()


# I AM USING 3 STD DEVIATION FOR OUTLIER CONSIDERATION AS IT CONTAIN 95% OF THE DATA.
# 
#     We can see that outliers are mostly in the upper boundaries of the dataset. Dataset above 8700(mean + 3 std) are outliers.  

# In[ ]:


train_1.loc[train_1.case_count>=8700,'case_count']=np.nan


# **SEGMENT 2 DATA **

# In[ ]:


train_2.plot(style='.', figsize=(15,5), title='Train_data_segment 2')


# In[ ]:


# PLOTTING BOXPLOT FOR TRAIN DATA SEGMENT 2 

train_2.boxplot( column =['case_count'], grid = True)


# In[ ]:


train_2.describe()


#    THERE ARE NO OULIERS IN THE TRAIN DATA OF SEGMENT 2 

# *NOW WE ARE READY FOR FEATURE GENERATION. FEATURE GENERATION WILL BE COVERED IN NEXT NOTEBOOK 
# ANALYTICS VIDHYA LTFS HACKATHON JAN 2020 PART 2 *

# CREDITS :
# 1. https://github.com/rajat5ranjan/AV-LTFS-Data-Science-FinHack-2
# 1. https://github.com/KrishnaPriyaIITR/LTFS-Data-Science-FinHack-2/blob/master/LTFS_baseline_V31_br.ipynb 
