#!/usr/bin/env python
# coding: utf-8

# #### Airbnb - Part 1 of 3 (Cleaning & Univariate Analysis)
# 
# Preprocessing of the raw data. I clean the age and gender fiels specifically as the show show positive correlation with the destination (verified vis hypothesis testing in part 2). This notebook further entails a univariate or on-the-go analysis of the features being cleaned. I made use of some great suggestions made in kaggle kernels, the references for which are listed below. This analysis is by no means exhaustive, it is just an initial eyballing of the data. 

# In[ ]:


# Import relevant libraraies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Load the data
train = pd.read_csv("../input/airbnb-recruiting-new-user-bookings/train_users_2.csv")
train.head()


# In[ ]:


train.columns


# In[ ]:


train.info()


# ### Age

# In[ ]:


print("Null value % in age is: " + "{0:.2%}".format(sum(train.age.isnull())/train.shape[0]))


# In[ ]:


train.age.describe()


# #### There must be some erroneus values in 'age', as the min = 1 and max = 2014. Let's see a more clear breakdown.

# In[ ]:


plt.figure(figsize=(20,6))
sns.countplot(train.age)
plt.xticks(rotation=90) 


# We know Airbnb has a above 18 years usage, and clearly people of ages 105 and 2014 (which show sigificant bumps) are not using the app. 
# #### Let's cleanup these age values and keep out dataset between 18 and 95

# In[ ]:


train.loc[train.age < 18, 'age'] = np.nan
train.loc[train.age > 95, 'age'] = np.nan


# In[ ]:


train.age = train.age.replace("NaN", np.nan)


# In[ ]:


plt.figure(figsize=(20,6))
sns.countplot(train.age)
plt.xticks(rotation=90) 


# In[ ]:


plt.figure(figsize=(6,6))
temp = train.age
sns.distplot(temp.dropna())


# In[ ]:


temp.dropna().describe()


# ###### Median (33) < Mean (36) -- Positive Skewness

# In[ ]:


print("Now the % of null values in age is: " + "{0:.2%}".format(sum(train.age.isnull())/train.shape[0]))


# In[ ]:


print("% of people with age <= 40: " + "{0:0.2%}".format(sum(train.age <= 40)/sum(train.age.notnull()))
     + "\n% of people with age > 40: "+ "{0:0.2%}".format(sum(train.age > 40)/sum(train.age.notnull())))


# ##### Age Notes:
# 1. ~40% data missing 
# 2. Majority (~3/4ths) are young or below 40 in remianing

# ### Gender

# In[ ]:


print("Null value % in gender is: " + "{0:.2%}".format(sum(train.gender.isnull())/train.shape[0]))


# In[ ]:


print("Unique values in Gender:",set(train.gender))


# Gender has some 'unknown' values which are not 'Other', so we can simply convert them to nan.

# In[ ]:


train.gender.replace('-unknown-', np.nan, inplace=True)


# In[ ]:


print("Unique values in Gender:",set(train.gender))


# In[ ]:


print("New null value % in gender is: " + "{0:.2%}".format(sum(train.gender.isnull())/train.shape[0]))


# Just like age, the null% in gender is also high, which is cancern that will need to be addressed in terms of modelling later on.

# In[ ]:


sns.countplot(train.gender)


# In[ ]:


print(train.gender.value_counts()/sum(train.gender.notnull())*100)


# ###### Gender Notes:
# 1. ~45% data missing 
# 2. Majority (53%) are Feamles in remianing
# 3. <0.2% are Other

# ### Language

# In[ ]:


print("Null value % in language is: " + "{0:.2%}".format(sum(train.language.isnull())/train.shape[0]))


# In[ ]:


print(set(train.language))


# In[ ]:


plt.figure(figsize=(20,6))
sns.countplot(train.language)


# In[ ]:


for i in set(train.language):
    print(i,": " + "{0:.2%}".format(sum(train.language == i)/sum(train.language.notnull())))


# ##### Language Notes:
# 1. No missing Data
# 2. Majority (~95%) are English speakers in remianing

# ### Country Destination

# In[ ]:


print(set(train.country_destination))


# In[ ]:


print("Null % is: " + "{0:0.2%}".format(sum(train.country_destination.isnull())/train.shape[0]))


# Note: 'NDF' or No Destination Found is differnt from 'Other' (apart from the ones mentioned) and nan/null values

# In[ ]:


plt.figure(figsize=(14,4))
sns.countplot(train.country_destination)


# In[ ]:


print(train.country_destination.value_counts()/sum(train.country_destination.notnull())*100)


# ##### Destination Country Notes:
# 1. Majority (~60%) is NDF
# 2. Second Highest is US (~30%), rest of the countries are in single digits or even less than that

# ### Signup Method, App, and Flow

# In[ ]:


print("Signup Method null % is: " + "{0:0.2%}".format(sum(train.signup_method.isnull())/train.shape[0]))
print("Signup Flow null % is: " + "{0:0.2%}".format(sum(train.signup_flow.isnull())/train.shape[0]))
print("Signup App null % is: " + "{0:0.2%}".format(sum(train.signup_app.isnull())/train.shape[0]))


# In[ ]:


sns.countplot(train.signup_method)


# In[ ]:


print(train.signup_method.value_counts()/sum(train.signup_method.notnull())*100)


# In[ ]:


sns.countplot(train.signup_flow)


# In[ ]:


sns.countplot(train.signup_app)


# In[ ]:


print(train.signup_app.value_counts()/sum(train.signup_app.notnull())*100)


# #### Signup Method, App, and Flow Notes:
# 2. Basic method is at ~70% and Facebook at ~28%, and Google is almost non-existent
# 3. Signup Flow seems irrelvant at this point, maybe in bivariate or multivariate analysis it will interplay with a feature
# 4. Web seems to be the predominant signup app (~85%), with others in single digits

# ### Affiliate Channel, Affiliate Provider, First Affiliate Tracked

# In[ ]:


print("Affiliate Channel null % is: " + "{0:0.2%}".format(sum(train.affiliate_channel.isnull())/train.shape[0]))
print("Affiliate Provider null % is: " + "{0:0.2%}".format(sum(train.affiliate_provider.isnull())/train.shape[0]))
print("First Affiliate Tracked null % is: " + "{0:0.2%}".format(sum(train.first_affiliate_tracked.isnull())/train.shape[0]))


# In[ ]:


print("Channel: ",set(train.affiliate_channel), "\nProvider: ", 
      set(train.affiliate_provider), "\nFirst Tracked: ",set(train.first_affiliate_tracked))


# In[ ]:


plt.figure(figsize=(14,4))
sns.countplot(train.affiliate_channel)


# In[ ]:


plt.figure(figsize=(20,6))
sns.countplot(train.affiliate_provider)
plt.xticks(rotation=45) 


# In[ ]:


plt.figure(figsize=(14,4))
sns.countplot(train.first_affiliate_tracked)


# #### Affiliate Channel, Affiliate Provider, First Affiliate Tracked:
# 1. None to low Null Values
# 2. Values dont necessarily seem relvant at this point, maybe in bivariate or multivariate analysis they will interplay with a feature

# ### First Device Type,	First Browser

# In[ ]:


print("First Device Type null % is: " + "{0:0.2%}".format(sum(train.first_device_type.isnull())/train.shape[0]))
print("First Browser null % is: " + "{0:0.2%}".format(sum(train.first_browser.isnull())/train.shape[0]))


# In[ ]:


print("First Device Type: ",set(train.first_device_type), "\nFirst Browser: ", 
      set(train.first_browser))


# In[ ]:


train.first_browser.replace('-unknown-',np.nan,inplace=True)


# In[ ]:


plt.figure(figsize=(20,4))
sns.countplot(train.first_device_type)


# In[ ]:


print(train.first_device_type.value_counts()/sum(train.first_device_type.notnull())*100)


# In[ ]:


plt.figure(figsize=(20,4))
sns.countplot(train.first_browser)
plt.xticks(rotation=45) 


# In[ ]:


print(train.first_browser.value_counts()/sum(train.first_browser.notnull())*100)


# #### First Device Type,	First Browser:
# Both major device types and browser seem in propotion to the popular devices and browsers.

# ### Date Account Created, Timestamp First Active, Date First Booking

# In[ ]:


print("Date Account Created % is: " + "{0:0.2%}".format(sum(train.date_account_created.isnull())/train.shape[0]))
print("Timestamp First Active null % is: " + "{0:0.2%}".format(sum(train.timestamp_first_active.isnull())/train.shape[0]))
print("Date First Booking % is: " + "{0:0.2%}".format(sum(train.date_first_booking.isnull())/train.shape[0]))


# In[ ]:


train.date_account_created = pd.to_datetime(train.date_account_created)
# print(train.date_account_created)


# In[ ]:


plt.figure(figsize=(80,8))
sns.countplot(train.date_account_created)
plt.xticks(rotation=90)


# In[ ]:


plt.figure(figsize=(16,4))
train.date_account_created.value_counts().plot(kind='line')


# In[ ]:


train.timestamp_first_active = pd.to_datetime(train.timestamp_first_active//1000000, format='%Y%m%d')
# print(train.timestamp_first_active)


# In[ ]:


plt.figure(figsize=(16,4))
train.timestamp_first_active.value_counts().plot(kind='line')


# In[ ]:


train.date_first_booking = pd.to_datetime(train.date_first_booking)


# In[ ]:


plt.figure(figsize=(16,4))
train.date_first_booking.value_counts().plot(kind='line')


# #### Date Account Created, Timestamp First Active, Date First Booking:
# 1. First booking date has about ~50% missing data
# 2. Timeseries plots of Timestamp and First active look similar at a high level, and may prove correlated in a bivariate analysis.

# In[ ]:


# An overarching look at the missing data
msno.matrix(train)


# In[ ]:


# train.to_csv('train_users_3.csv',index=False)


# #### References

# 1. https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
# 2. https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners
# 3. https://www.kaggle.com/justk1/airbnb
# 4. https://www.kaggle.com/davidgasquez/user-data-exploration
# 5. https://www.kaggle.com/kevinwu06/airbnb-exploratory-analysis
# 6. https://www.kaggle.com/rounakbanik/airbnb-new-user-bookings

# ##### - Aditya
