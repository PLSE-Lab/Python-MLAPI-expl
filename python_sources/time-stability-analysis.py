#!/usr/bin/env python
# coding: utf-8

# # Premises to check
# 
# * gender behave differently
# * Does age impact no-show, is this linear, exponential
# * Date time premisses
#   * the hour of the day
#   * day of the week
#   * period of the year - Months or seasons
#   * gap between schedule and appoinment affect the no show, because the patient forgets the appointment.
#   

# Some part of this code is based on https://www.kaggle.com/somrikbanerjee/predicting-show-up-no-show

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np;
import pandas as pd;
import math;
import matplotlib.pyplot as plt;
from matplotlib import pylab;
import seaborn as sns;
sns.set_style("whitegrid");

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.preprocessing import quantile_transform;

import statsmodels.discrete.discrete_model as sm;

# Any results you write to the current directory are saved as output.


# In[ ]:


dataset = pd.read_csv('../input/KaggleV2-May-2016.csv');
dataset.rename(columns = {'ApointmentData':'AppointmentData',
                         'Alcoolism': 'Alchoholism',
                         'Hipertension': 'Hypertension',
                         'No-show':'Status',
                         'Handcap': 'Handicap'}, inplace = True);
dataset.AppointmentDay = dataset.AppointmentDay.apply(np.datetime64);
dataset.ScheduledDay   = dataset.ScheduledDay.apply(np.datetime64);
dataset.Status = dataset.Status.apply(lambda x: 0 if x == 'No' else 1)
dataset.tail()


# Data cleaning and Variable creation

# In[ ]:


dataset['AwaitingDays'] = (dataset.AppointmentDay - dataset.ScheduledDay).dt.days;
dataset['AppointmentDayOfWeek'] = dataset.AppointmentDay.dt.dayofweek;
dataset['AppointmentMonth'] = dataset.AppointmentDay.dt.month;
dataset['ScheduledHour'] = dataset.ScheduledDay.dt.hour;
dataset['reference']  =dataset.ScheduledDay.dt.year *100 + dataset.ScheduledDay.dt.month;
dataset.Gender = dataset.Gender.apply(lambda x: 1 if x == 'M' else 0);
dataset['AgeBucket'] = dataset.Age.apply(lambda age: max(min(age,80),15));


#dataset.AwaitingTime = dataset.AwaitingTime.apply(abs);
dataset.head()


# In[ ]:


for field in ['AppointmentDayOfWeek','ScheduledHour','AwaitingDays', 'Age']:
    df = pd.DataFrame({
        'quantile':np.ceil(100*quantile_transform(dataset[field].values.reshape(-1, 1),n_quantiles=100)).flatten(),
        field : dataset[field],
        'Status':dataset['Status']        
    })
    df.groupby(by=['quantile']).mean().reset_index().plot(x=field,y='Status');


# In[ ]:


sns.barplot(x="AppointmentDayOfWeek", y="Status", hue="Gender", data=dataset);


# In[ ]:


for field in ['AppointmentDayOfWeek','ScheduledHour','AwaitingDays', 'Age']:
    df = pd.DataFrame({
        'quantile':np.ceil(5*quantile_transform(dataset[field].values.reshape(-1, 1),n_quantiles=5)).flatten()/5,
        field : dataset[field],
        'Status':dataset['Status'], 
        'reference':dataset['reference'], 
    });
    groups = df[['quantile',field]].groupby(by='quantile').mean().reset_index();
    df.drop(field, axis=1, inplace=True);
    df = df.merge(groups, left_on='quantile', right_on='quantile', how='inner');
    sns.pointplot(x="reference", y="Status", hue=field, data=df);
    plt.show();


# In[ ]:


df


# In[ ]:


for field in ['AppointmentDayOfWeek','ScheduledHour','AwaitingDays', 'Age']:
    df = pd.DataFrame({
        'quantile':np.ceil(10*quantile_transform(dataset[field].values.reshape(-1, 1),n_quantiles=10)).flatten(),
        'reference': dataset[],
        field : dataset[field],
        'Status':dataset['Status']        
    })
    df.groupby(by=['quantile']).mean().reset_index().plot(x=field,y='Status');


# In[ ]:


def probStatusCategorical(group_by):
    rows = []
    for item in group_by:
        for level in dataset[item].unique():
            rows.append({
                'Condition': item,
                'Level': level,
                'Probability': dataset[dataset[item] == level].Status.mean()
            });
    return pd.DataFrame(rows)

sns.barplot(data = probStatusCategorical(['Diabetes', 'Alcoholism', 'Hypertension']),
            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2'
           );
plt.show();
sns.barplot(data = probStatusCategorical(['SMS_received', 'Scholarship', 'Gender']),
            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2',
           );
#sns.show()


# In[ ]:





# In[ ]:


dataset.groupby(by='AgeBucket').mean().reset_index().plot(x='AgeBucket',y='Status',figsize=(10,3));


# In[ ]:


model = sm.Logit(dataset['Status'],dataset['AgeBucket'] )
print(model.fit().summary())

