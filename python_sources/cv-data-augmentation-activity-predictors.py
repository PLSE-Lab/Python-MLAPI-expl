#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # I. Loading Relevant Data #

# In[ ]:


input_dir = "../input"
print(os.listdir(input_dir))


# ## I.1. Loading Raw Data Tables ##

# In[ ]:


raw_dir = "../input/data-science-for-good-careervillage"
professionals = pd.read_csv(os.path.join(raw_dir, 'professionals.csv'), parse_dates=True)
students = pd.read_csv(os.path.join(raw_dir, 'students.csv'), parse_dates=True)


# ## I.2. Load Supervised Machine Learing Data Set ##

# In[ ]:


examples_dir = os.path.join(input_dir,'cv-machine-learning-data-construction')
examples = pd.read_parquet(os.path.join(examples_dir,'positive_negative_examples.parquet.gzip'))


# In[ ]:


examples.shape


# In[ ]:


examples.sample(10)


# ** There are missing values on 'emails_date_sent' of some questions answered by professionals. We impute these missing values by the question creation date, i.e. 'questions_date_added'. These are the cases where answers for a question are not results of recommendations. **

# In[ ]:


print(examples[pd.isnull(examples['emails_date_sent'])].shape[0])
print(examples[(pd.isnull(examples['emails_date_sent'])) & (examples['matched']==1)].shape[0])


# In[ ]:


examples['emails_date_sent'] = examples.apply(lambda row: row['questions_date_added'] if pd.isnull(row['emails_date_sent']) else row['emails_date_sent'], axis=1)


# In[ ]:


examples[pd.isnull(examples['emails_date_sent'])].shape


# In[ ]:


examples.sample(10)


# In[ ]:


examples['emails_date'] = examples['emails_date_sent'].dt.date


# # II. Loading and Joining Activity Predictors #

# In[ ]:


activity_predictors_dir = os.path.join(input_dir,'cv-feature-engineering-activity-predictors')
print(os.listdir(activity_predictors_dir))


# ## II.1. Days From Joined Dates ##

# In[ ]:


days_from_joined_dates = pd.read_parquet(os.path.join(activity_predictors_dir,'days_from_joined_dates.parquet.gzip'))
days_from_joined_dates = days_from_joined_dates.stack().reset_index()
days_from_joined_dates = days_from_joined_dates.rename(columns={'level_0': 'activity_date',
                                                                0:'days_from_joined_dates'})
days_from_joined_dates['activity_date'] = days_from_joined_dates['activity_date'].dt.date
days_from_joined_dates.dtypes


# In[ ]:


days_from_joined_dates.sample(3)


# In[ ]:


examples = examples.merge(days_from_joined_dates, 
                          left_on=['emails_date', 'answer_user_id'],
                          right_on=['activity_date', 'professionals_id'], 
                          how='left')
examples = examples.drop(['activity_date', 'professionals_id'], axis=1)


# ** Fill 0 for missing values of 'Days From Joined Dates' since these are more likely the cases when professionals join after the question creation date. **

# In[ ]:


print(examples.shape)
print(examples.dropna().shape)


# In[ ]:


examples['days_from_joined_dates'] = examples['days_from_joined_dates'].fillna(0)


# In[ ]:


print(examples.shape)
print(examples.dropna().shape)


# In[ ]:


examples.sample(10)


# ## II.2. Days From Last Activities ##

# In[ ]:


days_from_last_activities = pd.read_parquet(os.path.join(activity_predictors_dir,'days_from_last_activities.parquet.gzip'))
days_from_last_activities = days_from_last_activities.stack().reset_index()
days_from_last_activities = days_from_last_activities.rename(columns={'level_0': 'activity_date',
                                                                      0:'days_from_last_activities'})
days_from_last_activities['activity_date'] = days_from_last_activities['activity_date'].dt.date


# In[ ]:


examples = examples.merge(days_from_last_activities, 
                          left_on=['emails_date', 'answer_user_id'],
                          right_on=['activity_date', 'professionals_id'], 
                          how='left')
examples = examples.drop(['activity_date', 'professionals_id'], axis=1)


# In[ ]:


examples.sample(10)


# ** Fill missing values of 'Days From Last Activities' by the maximum value since these are more likely the cases when professionals are not active any more. **

# In[ ]:


print(examples.shape)
print(examples.dropna().shape)


# In[ ]:


examples['days_from_last_activities'] = examples.apply(lambda row: 
                                                       row['days_from_joined_dates'] if np.isnan(row['days_from_last_activities'])
                                                       else row['days_from_last_activities'], axis=1)


# In[ ]:


print(examples.shape)
print(examples.dropna().shape)


# In[ ]:


examples.sample(10)


# ## II.3. Rolling Windowed Activity Counts ##

# In[ ]:


window_days = [100000, 365, 30]
for window in window_days:
    print('Process window: {}'.format(window))
    col_name = 'professional_activities_sum_{}'.format(window)
    cum_sum_professional_activities = pd.read_parquet(
        os.path.join(activity_predictors_dir, 'professional_activities_sum_{}.parquet.gzip'.format(window)))
    cum_sum_professional_activities = cum_sum_professional_activities.stack().reset_index()    
    cum_sum_professional_activities = cum_sum_professional_activities.rename(columns={0:col_name})
    cum_sum_professional_activities['activity_date'] = cum_sum_professional_activities['activity_date'].dt.date

    examples = examples.merge(cum_sum_professional_activities, 
                              left_on=['emails_date', 'answer_user_id'],
                              right_on=['activity_date', 'professionals_id'], 
                              how='left')
    examples = examples.drop(['activity_date', 'professionals_id'], axis=1)
    examples[col_name] = examples[col_name].fillna(0)
    print(examples.head(3))


# In[ ]:


examples.shape


# In[ ]:


examples.sample(10)


# In[ ]:


examples.to_parquet('positive_negative_examples.parquet.gzip', compression='gzip')


# In[ ]:


os.listdir()

