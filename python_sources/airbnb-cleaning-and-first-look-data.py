#!/usr/bin/env python
# coding: utf-8

# **AirBnB - Cleaning and First Look @ Data**
# 
# Sumary of Steps Taken in Kernel
# 
# Age: 
# 1. Drop null values
# 2. Change outliers
# 3. Visualize age across entire dataset
# 
# Gender:
# 1. Convert to accepted values
# 2. Analyze and visualize destination/gender trends
# 
# Dates:
# 1. Transform all to datetime notation
# 2. Examine average lengths between important dates
# 3. Examine discrepancies in recorded dates
# 4. Visualize account creation & booking dates
# 
# Categorical Variables:
# 1. Visualize trends between signup_method, signup_app and country destination
# 2. Summarize first_device_type and language data
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Draw inline
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#load users
users = pd.read_csv('../input/train_users_2.csv')


# First, clean data. How many elements are NaN? Do you need to drop any columns with missing values? Are there are any other values that are missing?

# In[ ]:


users.isnull().sum()


# **date_first_booking** is NaN for 124543 users; that makes sense as those are the users who did not book a trip anywhere

# **age** is NaN for 87990 users; that's not great as that seems like it would be an important variable

# In[ ]:


#examining age
users['age'] = users['age'].dropna()

age_plot = sns.countplot(users['age'])

#limiting tick frequency for ease of readability 
for ind, label in enumerate(age_plot.get_xticklabels()):
    if ind % 15 == 0:
        label.set_visible(True)
    else:
        label.set_visible(False)


# In[ ]:


#looks like there are ages at 1 and above 100?

users['age'].describe()


# In[ ]:


#transform outliers into NaN

users.loc[users['age'] < 15, 'age'] = np.NaN
users.loc[users['age'] > 100, 'age'] = np.NaN


# Next, we can evaluate gender in the dataset.

# In[ ]:


sns.countplot(users['gender'], palette = "deep")


# In[ ]:


#change unknowns to NaN
users.loc[users['gender'] == '-unknown-', 'gender'] = np.NaN
sns.countplot(users['gender'],  palette = "deep")


# In[ ]:


women = sum(users['gender'] == 'FEMALE')
men = sum(users['gender'] == 'MALE')
print('There are', women, 'women and', men, 'men in this dataset.')


# In[ ]:


#any trend in country destinations in users who put NaN as their gender?

#total number who have NaN genders, and their destinations (proportional)
na_genders = users.loc[users['gender'].isna(), 'country_destination'].value_counts().sum()
na_gender_countries = users.loc[users['gender'].isna(), 'country_destination'].value_counts() / na_genders * 100

#non NaN country destinations (proportional)
complete_genders = users.loc[users['gender'].notnull(), 'country_destination'].value_counts().sum()
complete_gender_countries = users.loc[users['gender'].notnull(), 'country_destination'].value_counts() / complete_genders * 100

compared_na_genders = pd.concat([na_gender_countries, complete_gender_countries], axis = 1)
compared_na_genders.columns = ['na gender countries', 'complete gender countries']
compared_na_genders

#so definitely greater proportion of folks who have NaN as a gender don't book a trip anywhere


# In[ ]:


#Is there an initial trend in where (filled out) genders are going?

#find number of female and male users

female = users.loc[users['gender'] == 'FEMALE', 'country_destination'].value_counts().sum()
male = users.loc[users['gender'] == 'MALE', 'country_destination'].value_counts().sum()

#scale according to total numbers of female and male users

female_destinations = users.loc[users['gender'] == 'FEMALE', 'country_destination'].value_counts() / female * 100
male_destinations = users.loc[users['gender'] == 'MALE', 'country_destination'].value_counts() / male * 100

gender_dest = pd.concat([female_destinations, male_destinations], axis=1)
gender_dest.columns = ['female destinations', 'male destinations']
gender_dest


# In[ ]:


ax = gender_dest.plot.bar(colormap = 'jet', title = 'Percentage of Gender Per Destination')
ax.set_xlabel("Country Destination")
ax.set_ylabel("Percentage")

# Looks just about the same for both groups - nothing stands out here


# Now, let's handle the dates - are they all in the correct format?

# In[ ]:


#checking what format the dates are in, as they're not in timestamp form (from users.dtypes)

print("Date Account Created","\n", users['date_account_created'].sample(3))
print("\n")
print("Date of First Booking", "\n", users['date_first_booking'].sample(3))


# In[ ]:


#convert dates to proper datetime notation

users['date_account_created'] = pd.to_datetime(users['date_account_created'], format = '%Y-%m-%d', errors='coerce')
users['date_first_booking'] = pd.to_datetime(users['date_first_booking'], format = '%Y-%m-%d', errors='coerce')


# In[ ]:


users['date_account_created'].describe()


# In[ ]:


users['date_first_booking'].describe()


# In[ ]:


#finding the average lengh between account created and first booking

import datetime as dt


# In[ ]:


users['Difference'] = users['date_first_booking'] - users['date_account_created']
print("Average length between account creation & first booking:", users['Difference'].mean())


# In[ ]:


users['Difference'].describe()

#How are there negative days between account creation & first booking?


# In[ ]:


users[users['Difference'] < pd.Timedelta(0)].sample(5)

#strange - looks like either AirBnb allows you book before creating an account, or this data has been entered incorrectly
#for now, going to disregard - but note the date columns might not be entirely trustworthy
#upon futhur research, can replace date values as needed


# In[ ]:


# visualize dates that accounts are created, and the first booking happens

grouped_create_date = users['date_account_created'].dt.year
grouped_first_date = users['date_first_booking'].dt.year

fig, ax = plt.subplots(1,2, figsize=(20, 7))
sns.countplot((grouped_create_date), ax=ax[0], hue = users['country_destination'])
sns.countplot((grouped_first_date), ax=ax[1], hue = users['country_destination'])

ax[0].set_xlabel('Date of Account Creation')
ax[0].set_ylabel('Number of Created Accounts')
ax[0].set_title('Accounts Created')

ax[1].set_xlabel('Date of First Booking')
ax[1].set_ylabel('Number of Bookings')
ax[1].set_title('Bookings')

fig.show()


# In[ ]:


#now we have to correct the format of 'timestamp_first_active'

print(type(users['timestamp_first_active'][0]))
print(users['timestamp_first_active'].sample(2))


# In[ ]:


users['timestamp_first_active'] = pd.to_datetime((users['timestamp_first_active']//1000000), format='%Y%m%d')


# In[ ]:


#finding average length between first active day and account creation

users['OG_Difference'] = users['date_first_booking'] - users['timestamp_first_active']
print("Average length between account first active & first booking:", users['OG_Difference'].mean())

#looks to be about the same as account creation & first booking - are account creation & first active very similar same?

users['First Lag'] = users['date_account_created'] - users['timestamp_first_active']
print("Average length between account first active & date account created:", users['First Lag'].mean())

#Looks like most people created their account on the same day they were first active


# In[ ]:


users['First Lag'].describe()

#although one user waited 1456 days after they were first active to create an account!


# In[ ]:


#explore categorical variables - do any have initial trends?

sns.countplot(users['signup_method'], hue = users['country_destination'])

#setting legend outside of display box
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:


sns.countplot(users['signup_app'], hue = users['country_destination'])

#setting legend outside of display box
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:


first_device_type = users['first_device_type'].value_counts()
pd.DataFrame(first_device_type).transpose()


# In[ ]:


language = users['language'].value_counts()
pd.DataFrame(language).transpose()
