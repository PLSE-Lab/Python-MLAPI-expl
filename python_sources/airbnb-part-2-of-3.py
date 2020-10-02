#!/usr/bin/env python
# coding: utf-8

# #### Airbnb - Part 2 of 3 (Multivariate Analysis)
# 
# An initial bivariate and multivariate analysis of the features and the interplay between them. I use the data I cleaned in part 1 and build this analyses on top of that. I made use of some great suggestions made in kaggle kernels, the references for which are listed below. This analysis is by no means exhaustive, it is just a n initial eyballing of the data.

# In[ ]:


# Import relevant libraraies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats


# In[ ]:


pwd


# In[ ]:


# Load the data from Part 1
train = pd.read_csv("../input/airbnb-recruiting-new-user-bookings/train_users_2.csv")
train.head()


# In[ ]:


country_dict = {val:num for num,val in enumerate(set(train.country_destination))}
print(country_dict)


# #### Gender

# In[ ]:


train.gender.replace('-unknown-', np.nan, inplace=True)


# In[ ]:


plt.figure(figsize=(8,4))
num_women = sum(train.gender == 'FEMALE')
num_men = sum(train.gender == 'MALE')

m_des = train.loc[train.gender == 'MALE', 'country_destination'].value_counts()/num_women*100
f_des = train.loc[train.gender == 'FEMALE', 'country_destination'].value_counts()/num_men*100

m_des.plot(kind='bar', position=0, label='Male',color = 'red')
f_des.plot(kind='bar', position=1, label='Female',color='green')

plt.legend()
plt.xlabel('country')
plt.ylabel('count')

plt.show()


# There dosen't exist a 'major' bifurcaton on the basis of gender in country destinations, but an overall eyeballing a higher % of Females travelling to US compared to Males

# ###### Chi Square Square Significance Test for Gender vs Destination

# In[ ]:


chi_test_1 = train[(train.country_destination != 'NDF') & 
                   (train.country_destination != 'other') & 
                   (train.gender != 'OTHER') & 
                   (train.gender.notnull())]
chi_test_1 = chi_test_1[['id', 'gender', 'country_destination']]
chi_test_1.head()


# Hypothesis Testing:
# - Null Hypothesis: No relationship
# - Primary Hypothesis: Relationship Exits
# - Alpha = 5%

# In[ ]:


observed = chi_test_1.pivot_table('id', ['gender'], 'country_destination', aggfunc='count').reset_index()
del observed.columns.name
observed = observed.set_index('gender')
observed


# In[ ]:


chi2, p, dof, expected = stats.chi2_contingency(observed)


# In[ ]:


chi2


# In[ ]:


p


# p-value/significance less than Alpha, so reject Null Hypothesis --> Relationship b/w gender and destination exists

# **Age**

# In[ ]:


train.loc[train.age < 18, 'age'] = np.nan
train.loc[train.age > 95, 'age'] = np.nan


# In[ ]:


train.age = train.age.replace("NaN", np.nan)


# In[ ]:


plt.figure(figsize=(8,4))
young = sum(train.loc[train.age <= 36, 'country_destination'].value_counts())
old = sum(train.loc[train.age > 36, 'country_destination'].value_counts())

younger_des = train.loc[train.age <= 36, 'country_destination'].value_counts()/young*100
older_des = train.loc[train.age > 36, 'country_destination'].value_counts()/old*100

younger_des.plot(kind='bar', position=0, label='younger',color = 'orange')
older_des.plot(kind='bar', position=1, label='older',color='brown')

plt.legend()
plt.xlabel('country')
plt.ylabel('count')

plt.show()


# In[ ]:


chi_test_2 = train[(train.country_destination != 'NDF') & 
                   (train.country_destination != 'other') & 
                   (train.age.notnull())]
chi_test_2 = chi_test_2[['id', 'age', 'country_destination']]
chi_test_2.head()


# In[ ]:


# observed2 = chi_test_2.pivot_table('id', ['age'], 'country_destination', aggfunc='count').reset_index()
# del observed2.columns.name
# observed = observed2.set_index('age')
# observed


# chi2, p1, dof, expected = stats.chi2_contingency(observed)


# chi square test results --> Null Hypothesis

# Splitting age groups based on the Mean age (~36), we see a minor preference of the young gen to stay in US

# In[ ]:


train['age_subset'] = pd.cut(train["age"], [0, 30, 60, 90])
plt.figure(figsize=(16,4))
sns.countplot(x="age_subset",hue="country_destination", data=train[train['country_destination'] != 'NDF'])


# In[ ]:


# drop age_range
train.drop(['age_subset'], axis=1, inplace=True)


# #### Timeseries Analysis

# In[ ]:


train.date_account_created = pd.to_datetime(train.date_account_created)
train.date_first_booking = pd.to_datetime(train.date_first_booking)


# In[ ]:


# First Booking date
plt.figure(figsize=(16,4))
train.date_first_booking.value_counts().plot(kind='line')


# YoY increase in number of accounts created for Airbnb

# In[ ]:


plt.figure(figsize=(20,4))
train.time_subset = pd.cut(train.date_first_booking, [pd.to_datetime(20120101, format='%Y%m%d'), 
                                                              pd.to_datetime(20120601, format='%Y%m%d'),
                                                              pd.to_datetime(20130101, format='%Y%m%d'), 
                                                              pd.to_datetime(20130601, format='%Y%m%d'),
                                                              pd.to_datetime(20140101, format='%Y%m%d'),
                                                              pd.to_datetime(20140601, format='%Y%m%d')])
sns.countplot(train.time_subset)


# In[ ]:


# Time Period 2012-13
plt.figure(figsize=(12,4))
period_12_13 = train[train.date_first_booking >= pd.to_datetime(20120101, format='%Y%m%d')]
period_12_13 = period_12_13[period_12_13.date_first_booking < pd.to_datetime(20130101, format='%Y%m%d')]
period_12_13.date_first_booking.value_counts().plot(kind='line', linewidth=2, color='#FD5C64')
plt.show()


# In[ ]:


# Time Period 2013-14
plt.figure(figsize=(12,4))
period_13_14 = train[train.date_first_booking >= pd.to_datetime(20130101, format='%Y%m%d')]
period_13_14 = period_13_14[period_13_14.date_first_booking < pd.to_datetime(20140101, format='%Y%m%d')]
period_13_14.date_first_booking.value_counts().plot(kind='line', linewidth=2, color='#FD5C64')
plt.show()


# In[ ]:


# Time Period 201401-201406
plt.figure(figsize=(7,4))
period_14_15 = train[train.date_first_booking >= pd.to_datetime(20140101, format='%Y%m%d')]
period_14_15 = period_14_15[period_14_15.date_first_booking <= pd.to_datetime(20140701, format='%Y%m%d')]
period_14_15.date_first_booking.value_counts().plot(kind='line', linewidth=2, color='#FD5C64')
plt.show()


# All 3 time periods show a bump in the months of June-Nov, and a lower booking rate for for 1st half of the year

# In[ ]:


train.date_first_booking = pd.to_datetime(train.date_first_booking)


# In[ ]:


train.date_first_booking.describe()


# In[ ]:


weekdays2 = []
for date in train.date_first_booking:
    weekdays2.append(date.weekday())
weekdays2 = pd.Series(weekdays2)


# In[ ]:


plt.figure(figsize=(7,3))
sns.barplot(x = weekdays2.value_counts().index, y=weekdays2.value_counts().values, order=range(0,7))
plt.xlabel('Day')


# Weekedns show a lower number of bookings, and Mon-Wed seem to be hottest period when first bookings were made.

# ###### Date Account Created

# In[ ]:


train.date_account_created = pd.to_datetime(train.date_account_created)


# In[ ]:


train.date_account_created.describe()


# In[ ]:


plt.figure(figsize=(16,4))
train.date_account_created.value_counts().plot(kind='line')


# In[ ]:


# Time Period 2012-13
plt.figure(figsize=(12,4))
period_13_14 = train[train.date_account_created >= pd.to_datetime(20130101, format='%Y%m%d')]
period_13_14 = period_13_14[period_13_14.date_account_created < pd.to_datetime(20140101, format='%Y%m%d')]
period_13_14.date_account_created.value_counts().plot(kind='line', linewidth=2, color='#FD5C64')
plt.show()


# At a micro level the major bump seems to be in October, which coincides with the peaks First Bookings.

# In[ ]:


weekdays = []
for date in train.date_account_created:
    weekdays.append(date.weekday())
weekdays = pd.Series(weekdays)


# In[ ]:


plt.figure(figsize=(7,3))
sns.barplot(x = weekdays2.value_counts().index, y=weekdays2.value_counts().values, order=range(0,7))
plt.xlabel('Day')


# Weekday barplot is also largely similar to the date of first bookings

# In[ ]:


plt.figure(figsize=(12,6))
train[train['country_destination'] != 'NDF']['date_account_created'].value_counts().plot(kind='line', color='blue')
train[train['country_destination'] == 'NDF']['date_account_created'].value_counts().plot(kind='line',color='green')


# Number of users have increased, but with those who dont book a destination have as well

# In[ ]:


plt.figure(figsize=(20,6))
sns.barplot(x=train.age, y=train.country_destination, hue=train.gender, data=train)


# #### References

# 1. https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
# 2. https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners
# 3. https://www.kaggle.com/justk1/airbnb
# 4. https://www.kaggle.com/davidgasquez/user-data-exploration
# 5. https://www.kaggle.com/kevinwu06/airbnb-exploratory-analysis
# 6. https://www.kaggle.com/rounakbanik/airbnb-new-user-bookings

# ##### - Aditya
