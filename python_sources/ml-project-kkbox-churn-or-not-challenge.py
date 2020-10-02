#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"

import math 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import scipy as sp # scientific computing 
import seaborn as sns # visualization library
import time

from datetime import datetime
from collections import Counter
from subprocess import check_output

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/kkbox-churn-scala-label/"))
print(os.listdir("../input/kkbox-churn-prediction-challenge/"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# train dataset
df_train_file = "../input/kkbox-churn-scala-label/user_label_201703.csv"
df_train = pd.read_csv(df_train_file, dtype = {'is_churn': 'int8'})
df_train.head()
df_train.describe()
df_train.info()


# In[ ]:


# test dataset
df_test_file = "../input/kkbox-churn-prediction-challenge/sample_submission_v2.csv"
df_test = pd.read_csv(df_test_file)
df_test.head()
df_test.describe()
df_test.info()


# In[ ]:


# members dataset
df_members_file = "../input/kkbox-churn-prediction-challenge/members_v3.csv"
df_members = pd.read_csv(df_members_file)
df_members.head()
df_members.describe()
df_members.info(verbose = True, null_counts = True)


# In[ ]:


# analysising members dataset
# filling missing values
df_members['city'] = df_members.city.apply(lambda x: int(x) if pd.notnull(x) else "NAN")
# ignoring bd >= 100 and <=0 so setting both to -99999 to group properly
df_members['bd'] = df_members.bd.apply(lambda x: -99999 if float(x) <= 1 else x )
df_members['bd'] = df_members.bd.apply(lambda x: -99999 if float(x) >= 100 else x )
df_members['gender'] = df_members['gender'].fillna("others")
df_members['registered_via'] = df_members.registered_via.apply(lambda x: int(x) if pd.notnull(x) else "NAN")
df_members['registration_init_time'] = df_members.registration_init_time.apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN")
df_members.head()


# In[ ]:


# city analysis
plt.figure(figsize = (12,12))
plt.subplot(411) 
city_order = df_members['city'].unique()
# finding and removing NAN from numpy array as NAN messes up sorting algo
index = np.argwhere(city_order == "NAN")
# update city_order
city_order = np.delete(city_order, index)
city_order = sorted(city_order, key = lambda x: float(x))
sns.countplot(x = "city", data = df_members , order = city_order)
plt.ylabel('Count', fontsize = 12)
plt.xlabel('City', fontsize = 12)
plt.xticks(rotation = 'vertical')
plt.title("Frequency of City Count", fontsize = 12)
plt.show()


# We notice that city 1 is the most common of all members which can largely skew the data. Therefore, we will drop city column.

# In[ ]:


del df_members['city']
df_members.head()


# In[ ]:


# registered via analysis
plt.figure(figsize = (12,12))
plt.subplot(412)
R_V_order = df_members['registered_via'].unique()
R_V_order = sorted(R_V_order, key = lambda x: str(x))
R_V_order = sorted(R_V_order, key = lambda x: float(x))
sns.countplot(x = "registered_via", data = df_members, order = R_V_order)
plt.ylabel('Count', fontsize = 12)
plt.xlabel('Registered Via', fontsize = 12)
plt.xticks(rotation = 'vertical')
plt.title("Frequency of Registered Via Count", fontsize = 12)
plt.show()


# We notice that the main registered via are 3, 4 ,7 and 9. The rest are almost insignificant. Thus, we will group registered via into 5 main variables: 3, 4, 7, 9 and others

# In[ ]:


df_members['registered_via'].replace([-1, 1, 2, 5, 6, 8, 10, 11, 13, 14, 16, 17, 18, 19], 1, inplace = True)
# registered via analysis
plt.figure(figsize = (12,12))
plt.subplot(412)
R_V_order = df_members['registered_via'].unique()
R_V_order = sorted(R_V_order, key = lambda x: str(x))
R_V_order = sorted(R_V_order, key = lambda x: float(x))
sns.countplot(x = "registered_via", data = df_members, order = R_V_order)
plt.ylabel('Count', fontsize = 12)
plt.xlabel('Registered Via', fontsize = 12)
plt.xticks(rotation = 'vertical')
plt.title("Frequency of Registered Via Count", fontsize = 12)
plt.show()


# In[ ]:


# gender analysis
plt.figure(figsize = (12,12))
plt.subplot(413)
sns.countplot(x = "gender", data = df_members)
plt.ylabel('Count', fontsize = 12)
plt.xlabel('Gender', fontsize = 12)
plt.xticks(rotation = 'vertical')
plt.title("Frequency of Gender Count", fontsize = 12)
plt.show()
gender_count = Counter(df_members['gender']).most_common()
print("Gender Count " + str(gender_count))


# In[ ]:


# registration_init_time yearly trend
df_members['registration_init_time_year'] = pd.DatetimeIndex(df_members['registration_init_time']).year
df_members['registration_init_time_year'] = df_members.registration_init_time_year.apply(lambda x: int(x) if pd.notnull(x) else "NAN" )
year_count = df_members['registration_init_time_year'].value_counts()
print(year_count)
plt.figure(figsize = (12,12))
plt.subplot(311)
year_order = df_members['registration_init_time_year'].unique()
year_order = sorted(year_order, key = lambda x: str(x))
year_order = sorted(year_order, key = lambda x: float(x))
sns.barplot(year_count.index, year_count.values,order = year_order)
plt.ylabel('Count', fontsize = 12)
plt.xlabel('Year', fontsize = 12)
plt.xticks(rotation = 'vertical')
plt.title("Yearly Trend of registration_init_time", fontsize = 12)
plt.show()
year_count_2 = Counter(df_members['registration_init_time_year']).most_common()
print("Yearly Count " + str(year_count_2))


# In[ ]:


# registration_init_time monthly trend
df_members['registration_init_time_month'] = pd.DatetimeIndex(df_members['registration_init_time']).month
df_members['registration_init_time_month'] = df_members.registration_init_time_month.apply(lambda x: int(x) if pd.notnull(x) else "NAN")
month_count = df_members['registration_init_time_month'].value_counts()
plt.figure(figsize = (12,12))
plt.subplot(312)
month_order = df_members['registration_init_time_month'].unique()
month_order = sorted(month_order, key = lambda x: str(x))
month_order = sorted(month_order, key = lambda x: float(x))
sns.barplot(month_count.index, month_count.values, order = month_order)
plt.ylabel('Count', fontsize = 12)
plt.xlabel('Month', fontsize = 12)
plt.xticks(rotation = 'vertical')
plt.title("Monthly Trend of registration_init_time", fontsize = 12)
plt.show()
month_count_2 = Counter(df_members['registration_init_time_month']).most_common()
print("Monthly Count " + str(month_count_2))


# In[ ]:


# registration_init_time daily trend
df_members['registration_init_time_weekday'] = pd.DatetimeIndex(df_members['registration_init_time']).weekday_name
df_members['registration_init_time_weekday'] = df_members.registration_init_time_weekday.apply(lambda x: str(x) if pd.notnull(x) else "NAN" )
day_count = df_members['registration_init_time_weekday'].value_counts()
plt.figure(figsize = (12,12))
plt.subplot(313)
#day_order = training['registration_init_time_day'].unique()
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','NAN']
sns.barplot(day_count.index, day_count.values, order = day_order)
plt.ylabel('Count', fontsize = 12)
plt.xlabel('Day', fontsize = 12)
plt.xticks(rotation = 'vertical')
plt.title("Day-wise Trend of registration_init_time", fontsize = 12)
plt.show()
day_count_2 = Counter(df_members['registration_init_time_weekday']).most_common()
print("Day-wise Count " + str(day_count_2))


# In[ ]:


del df_members['registration_init_time']
df_members.head()


# In[ ]:


# birth date analysis
plt.figure(figsize = (12,8))
bd_order = df_members['bd'].unique()
bd_order = sorted(bd_order, key=lambda x: str(x))
bd_order = sorted(bd_order, key=lambda x: float(x))
sns.countplot(x="bd", data = df_members, order = bd_order)
plt.ylabel('Count', fontsize = 12)
plt.xlabel('BD', fontsize = 12)
plt.xticks(rotation = 'vertical')
plt.title("Frequency of BD Count", fontsize = 12)
plt.show()
bd_count = Counter(df_members['bd']).most_common()
print("BD Count " + str(bd_count))


# In[ ]:


del df_members['bd']
df_members.head()


# In[ ]:


# analysising userlogs data set
df_userlogs_file = "../input/kkbox-churn-prediction-challenge/user_logs_v2.csv"
df_userlogs = pd.read_csv(df_userlogs_file)
df_userlogs.head()
df_userlogs.describe()
df_userlogs.info(verbose = True, null_counts = True)


# In[ ]:


# groupby 
del df_userlogs['date']
counts = df_userlogs.groupby('msno')['total_secs'].count().reset_index()
counts.columns = ['msno', 'days_listened']
sums = df_userlogs.groupby('msno').sum().reset_index()
df_userlogs = sums.merge(counts, how='inner', on='msno')
df_userlogs.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
cols = df_userlogs.columns[1:]
log_userlogs = df_userlogs.copy()
log_userlogs[cols] = np.log1p(df_userlogs[cols])
ss = StandardScaler()
log_userlogs[cols] = ss.fit_transform(log_userlogs[cols])

for col in cols:
    plt.figure(figsize=(15,7))
    plt.subplot(1,2,1)
    sns.distplot(df_userlogs[col].dropna())
    plt.subplot(1,2,2)
    sns.distplot(log_userlogs[col].dropna())
    plt.figure()


# In[ ]:


# analysing transactions dataset
df_transactions_file = "../input/kkbox-churn-prediction-challenge/transactions_v2.csv"
df_transactions = pd.read_csv(df_transactions_file)
df_transactions.head()
df_transactions.describe()
df_transactions.info()


# In[ ]:


# payment_method_id count
plt.figure(figsize=(18,6))
plt.subplot(311)
sns.countplot(x = "payment_method_id", data = df_transactions)
plt.ylabel('Count', fontsize = 12)
plt.xlabel('payment_method_id', fontsize = 12)
plt.xticks(rotation = 'vertical')
plt.title("Frequency of payment_method_id Count in transactions Data Set", fontsize = 12)
plt.show()
payment_method_id_count = Counter(df_transactions['payment_method_id']).most_common()
print("payment_method_id Count " + str(payment_method_id_count))


# In[ ]:


del df_transactions['payment_method_id']
df_transactions.head()


# In[ ]:


# payment_plan_days count in transactions Data Set
plt.figure(figsize = (18,6))
sns.countplot(x = "payment_plan_days", data = df_transactions)
plt.ylabel('Count', fontsize = 12)
plt.xlabel('payment_plan_days', fontsize = 12)
plt.xticks(rotation='vertical')
plt.title("Frequency of payment_plan_days Count in transactions Data Set", fontsize = 12)
plt.show()
payment_plan_days_count = Counter(df_transactions['payment_plan_days']).most_common()
print("payment_plan_days Count " + str(payment_plan_days_count))


# In[ ]:


del df_transactions['payment_plan_days']
df_transactions.head()


# In[ ]:


# plan_list_price count in transactions Data Set
plt.figure(figsize = (18,6))
sns.countplot(x = "plan_list_price", data = df_transactions)
plt.ylabel('Count', fontsize = 12)
plt.xlabel('plan_list_price', fontsize = 12)
plt.xticks(rotation = 'vertical')
plt.title("Frequency of plan_list_price Count in transactions Data Set", fontsize = 12)
plt.show()
plan_list_price_count = Counter(df_transactions['plan_list_price']).most_common()
print("plan_list_price Count " + str(plan_list_price_count))


# In[ ]:


# actual_amount_paid count in transactions Data Set
plt.figure(figsize = (18,6))
sns.countplot(x = "actual_amount_paid", data = df_transactions)
plt.ylabel('Count', fontsize = 12)
plt.xlabel('actual_amount_paid', fontsize = 12)
plt.xticks(rotation = 'vertical')
plt.title("Frequency of actual_amount_paid Count in transactions Data Set", fontsize = 12)
plt.show()
actual_amount_paid_count = Counter(df_transactions['actual_amount_paid']).most_common()
print("actual_amount_paid Count " + str(actual_amount_paid_count))


# In[ ]:


#Correlation between plan_list_price and actual_amount_paid
df_transactions['plan_list_price'].corr(df_transactions['actual_amount_paid'], method = 'pearson') 


# In[ ]:


del df_transactions['actual_amount_paid']
df_transactions.head()


# In[ ]:


df_transactions['transaction_date'] = df_transactions.transaction_date.apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN")
df_transactions['membership_expire_date'] = df_transactions.membership_expire_date.apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN")
df_transactions['days_after'] = df_transactions['membership_expire_date'] - df_transactions['transaction_date']
del df_transactions['transaction_date']
del df_transactions['membership_expire_date']
df_transactions.head()


# In[ ]:


df_transactions.set_index('msno').index.get_duplicates()


# In[ ]:


# is_auto_renew count in transactions Data Set
plt.figure(figsize = (4,4))
sns.countplot(x = "is_auto_renew", data = df_transactions)
plt.ylabel('Count', fontsize = 12)
plt.xlabel('is_auto_renew', fontsize = 12)
plt.xticks(rotation = 'vertical')
plt.title("Frequency of is_auto_renew Count in transactions Data Set", fontsize = 6)
plt.show()
is_auto_renew_count = Counter(df_transactions['is_auto_renew']).most_common()
print("is_auto_renew Count " + str(is_auto_renew_count))


# In[ ]:


# is_cancel count in transactions Data Set
plt.figure(figsize = (4,4))
sns.countplot(x = "is_cancel", data = df_transactions)
plt.ylabel('Count', fontsize = 12)
plt.xlabel('is_cancel', fontsize = 12)
plt.xticks(rotation = 'vertical')
plt.title("Frequency of is_cancel Count in transactions Data Set", fontsize = 6)
plt.show()
is_cancel_count = Counter(df_transactions['is_cancel']).most_common()
print("is_cancel Count " + str(is_cancel_count))


# In[ ]:


# merge the training dataset with members, transaction, userlogs data set
df_training = pd.merge(left = df_train, right = df_members, how = 'left', on = ['msno'])
#df_training = pd.merge(left = df_training, right = df_transactions , how = 'left', on = ['msno'])
#df_training = pd.merge(left = df_training, right = log_userlogs, how = 'left', on = ['msno'])
df_training['gender'] = df_training['gender'].fillna("others")
gender = {'male': 0, 'female': 1, 'others' :2}
df_training['gender'] = df_training['gender'].map(gender)
#df_training['registered_via'] = df_training['registered_via'].fillna(0)
#df_training['registration_init_time_year'] = df_training['registration_init_time_year'].fillna(0)
#df_training['registration_init_time_month'] = df_training['registration_init_time_month'].fillna(0)
#df_training['registration_init_time_weekday'] = df_training['registration_init_time_weekday'].fillna("NaN")
days = {'Monday': 0, 'Tuesday': 1, 'Wednesday' : 2, 'Thursday' : 3, 'Friday' : 4, 'Saturday' : 5, 'Sunday' : 6, 'NaN' : 7}
df_training['registration_init_time_weekday'] = df_training['registration_init_time_weekday'].map(days)
#df_training['num_25'] = df_training['num_25'].fillna(0)
#df_training['num_50'] = df_training['num_50'].fillna(0)
#df_training['num_75'] = df_training['num_75'].fillna(0)
#df_training['num_985'] = df_training['num_985'].fillna(0)
#df_training['num_100'] = df_training['num_100'].fillna(0)
#df_training['num_unq'] = df_training['num_unq'].fillna(0)
#df_training['total_secs'] = df_training['total_secs'].fillna(0)
#df_training['days_listened'] = df_training['days_listened'].fillna(0)
df_training.to_csv('training.csv', index = False)


# In[ ]:


df_testing = pd.merge(left = df_test, right = df_members, how = 'left', on = ['msno'])
#df_testing = pd.merge(left = df_testing, right = df_transactions , how = 'left', on = ['msno'])
df_testing = pd.merge(left = df_testing, right = log_userlogs, how = 'left', on = ['msno'])
df_testing['gender'] = df_testing['gender'].fillna("others")
gender = {'male': 0, 'female': 1, 'others' :2}
df_testing['gender'] = df_testing['gender'].map(gender)
#df_testing['registered_via'] = df_testing['registered_via'].fillna(0)
#df_testing['registration_init_time_year'] = df_testing['registration_init_time_year'].fillna(0)
#df_testing['registration_init_time_month'] = df_testing['registration_init_time_month'].fillna(0)
#df_testing['registration_init_time_weekday'] = df_testing['registration_init_time_weekday'].fillna("NaN")
days = {'Monday': 0, 'Tuesday': 1, 'Wednesday' : 2, 'Thursday' : 3, 'Friday' : 4, 'Saturday' : 5, 'Sunday' : 6, 'NaN' : 7}
df_testing['registration_init_time_weekday'] = df_testing['registration_init_time_weekday'].map(days)
#f_testing['num_25'] = df_testing['num_25'].fillna(0)
#df_testing['num_50'] = df_testing['num_50'].fillna(0)
#df_testing['num_75'] = df_testing['num_75'].fillna(0)
#df_testing['num_985'] = df_testing['num_985'].fillna(0)
#df_testing['num_100'] = df_testing['num_100'].fillna(0)
#df_testing['num_unq'] = df_testing['num_unq'].fillna(0)
#df_testing['total_secs'] = df_testing['total_secs'].fillna(0)
#df_testing['days_listened'] = df_testing['days_listened'].fillna(0)
df_testing.to_csv('testing.csv', index = False)

