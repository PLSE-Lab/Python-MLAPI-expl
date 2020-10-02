#!/usr/bin/env python
# coding: utf-8

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
        assessment_dataset_path = os.path.join(dirname, filename)
        assessment_data = pd.read_csv(assessment_dataset_path)

# Any results you write to the current directory are saved as output.
assessment_data.head()


# **Here are the first 5 records of the Jabama assessment dataset.**
# It looks like there is a misspelling in the last column's title!

# In[ ]:


assessment_data = assessment_data.rename(columns = {'Date' : 'date', 'purchase_amonut' : 'purchase_amount'})
assessment_data.head()


# **The columns' titles are normalized.** Now, let us find out if there is any missing value in the dataset:

# In[ ]:


print('Is there any missing value in the assessment data?',
      assessment_data.isnull().values.any())


# In[ ]:


import time
import datetime

normalized_date = []
min_date = time.mktime(datetime.datetime.strptime(assessment_data.loc[0, 'date'], "%m/%d/%Y").timetuple())
max_date = time.mktime(datetime.datetime.strptime(assessment_data.loc[0, 'date'], "%m/%d/%Y").timetuple())

for i, row in assessment_data.iterrows():
    timestamp = time.mktime(datetime.datetime.strptime(row['date'], "%m/%d/%Y").timetuple())
    if min_date > timestamp:
        min_date = timestamp
    if max_date < timestamp:
        max_date = timestamp
    normalized_date.append(int(timestamp))
    
assessment_data['normalized_date'] = normalized_date
assessment_data.head()


# It is handier to manipulate or compare records' dates in a **datetime format** compared to the initial date format of the assessment data.

# In[ ]:


from datetime import datetime

date_domain = max_date - min_date

for i, row in assessment_data.iterrows():
    assessment_data.loc[i, 'normalized_date'] = (int(row['normalized_date']) - min_date) / date_domain
    
assessment_data.head()


# The **normalized_date** contains normalized values (ranging from 0 to 1) corresponding to the date attribute of each record.

# In[ ]:


print('Starting date: ', datetime.fromtimestamp(min_date).date())
print('Ending date: ', datetime.fromtimestamp(max_date).date())

print('The number of records: ', len(assessment_data))
print('The number of unique user IDs: ', len(assessment_data['user_id'].unique()))
print('The number of unique devices: ', len(assessment_data['device'].unique()))
print('The number of unique locations: ', len(assessment_data['location'].unique()))


# * So far, we know that the assessment data includes **786 purchases** made by **18 users** by **23 different devices** from **14 different locations** during **a four month period** from May 1 to August 31 in year 2014.

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

plt.figure(figsize=(18,12))
plot = sns.distplot(a=assessment_data["normalized_date"]*4*31, kde=True, color='yellow')
plot.set(xlabel ='The days from May 1 to August 31', ylabel ='The frequency of purchases')
plt.show()


# * It could be concluded that the frequency of purchases has peaked in **the early days of June** and also in **the early days of July**.

# In[ ]:


plt.figure(figsize=(18,12))
plot = sns.distplot(a=assessment_data["purchase_amount"], kde=True, color='orange')
plot.set(xlabel ='The amount of purchase', ylabel ='Frequency')
plt.show()


# * It could be concluded that **most of the purchases** have been made **around 100,000 (Rials)**, and purchases worth more than 400,000 (Rials) are rare.

# In[ ]:


sizes = []
devices = assessment_data['device'].unique()

for device in devices:
    sizes.append(len(assessment_data[assessment_data['device'] == device]))
    
sizes

fig, ax = plt.subplots(figsize=(18,12))

ax.pie(sizes,
       labels= devices,
      autopct='%1.1f%%')

ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# * It could be concluded that the **most frequently-used devices** are the **"Macbook Pro"** and the **"Lenovo Thinkpad"**. The least frequently-used device is **"Windows Surface"**.

# In[ ]:


amounts = []
devices = assessment_data['device'].unique()

for device in devices:
    amounts.append(assessment_data[assessment_data['device'] == device]['purchase_amount'].sum() /
                   len(assessment_data[assessment_data['device'] == device]))

fig, ax = plt.subplots(figsize=(18,12))
    
ax.pie(amounts,
       labels= devices,
      autopct='%1.1f%%')

ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# Although the average amount of purchases made using all devices are almost the same,
# * It could be concluded that purchases with **bigger amounts** have been made using the **"Windows Surface"**. Also in the previously shown piechart it was concluded that the least frequently-used device is the "Windows Surface".
# * On the other hand, purchases with **smaller amounts** belong to the **"Acer Aspire Notebook"** users.

# In[ ]:


amounts = []
locations = assessment_data['location'].unique()

for location in locations:
    amounts.append(assessment_data[assessment_data['location'] == location]['purchase_amount'].sum())

fig, ax = plt.subplots(figsize=(18,12))
    
ax.pie(amounts,
       labels= locations,
      autopct='%1.1f%%')

ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# Depict the proportional total amount of purchases distribution in different regions.
# * It could be concluded that the biggest spending region is Tehran, which is twice bigger than the second region, Razavi Khorasan.
# * I could be concluded that Kerman is the smallest spending region.

# In[ ]:


amounts = []
locations = assessment_data['location'].unique()

for location in locations:
    amounts.append(assessment_data[assessment_data['location'] == location]['purchase_amount'].sum() /
                  len(assessment_data[assessment_data['location'] == location]))

fig, ax = plt.subplots(figsize=(18,12))
    
ax.pie(amounts,
       labels= locations,
      autopct='%1.1f%%')

ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# Depict the average amount of purchases made in each region.
# * Although it was concluded in the previous piechart that Kerman is the smallest spending region, here it is shown that concerning the number of purchases, users from **Kerman** have made **the biggest purchases**, as well as **Fars**.
# 

# In[ ]:


plt.figure(figsize=(18,12))
sns.scatterplot(x=assessment_data['location'], y=assessment_data["purchase_amount"], hue=assessment_data['user_id'], legend = 'full')
plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
plt.show()


# Depict the distribution of purchases based on their amount for each region, colored based on the user IDs.
# * It could be concluded that two biggest purchases have been made from Qom and Khuzestan (outliers).

# In[ ]:


plt.figure(figsize=(18,12))
sns.scatterplot(x=assessment_data["normalized_date"]*31*4, y=assessment_data["purchase_amount"], hue=assessment_data["location"])
plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
plot.set(xlabel ='Day', ylabel ='The amount of purchase')
plt.show()


# Display the distribution of purchases by their amount during the time, colored based on their region.
# *In this plot, the conclusions of previously shown scatterplot could be tracked by their time of occurance.*

# In[ ]:


plt.figure(figsize=(18,12))
sns.scatterplot(x=assessment_data['user_id'], y=assessment_data["purchase_amount"], hue=assessment_data['device'], palette = 'Set2')
plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
plt.show()


# Depict the distribution of purchases made by each user (the horizontal axis) based on their amount (the vertical axis) colored by the device.
# * It could be concluded that every user has used multiple devices for different purchases.
# * Users with user IDs 1, 3, 11 and 12 have made outlier purchases (way out of their usual purchases amount range)

# In[ ]:


amounts = []
users = assessment_data['user_id'].unique()

for user in users:
    amounts.append(assessment_data[assessment_data['user_id'] == user]['purchase_amount'].sum() /
                   len(assessment_data[assessment_data['user_id'] == user]))

fig, ax = plt.subplots(figsize=(18,12))
    
ax.pie(amounts,
       labels= users,
      autopct='%1.1f%%')

ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# Depict the average amount of purchases made by different users. It could be seen that **the biggest spending customer** is user ID 4, in contrary to user ID 15.

# In[ ]:


plt.figure(figsize=(18,12))

for i in range(len(assessment_data['user_id'].unique())):
    user_id = assessment_data['user_id'].unique()[i]
    sns.lineplot(x=assessment_data[assessment_data['user_id'] == user_id]['normalized_date']*31*4, y='purchase_amount', data=assessment_data[assessment_data['user_id'] == user_id], legend='brief')
plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
plt.show()
# plot = sns.distplot(a=assessment_data["normalized_date"]*4*31, kde=True, color='yellow')
# plot.set(xlabel ='The days from May 1 to August 31', ylabel ='The frequency of purchases')
# plt.show()


# Depict the purchasing trend of every user over the time, distinguished by their color.
# * It could be concluded that most of the **big purchases** have been made about **end of the months** (the May, June and July).

# In[ ]:


print('Average purchase amount: ', assessment_data['purchase_amount'].sum() / len(assessment_data))


# In[ ]:


print('Average sales amount per day: ', assessment_data['purchase_amount'].sum() / (31*4))
print('Average sales amount per week: ', assessment_data['purchase_amount'].sum() / (31*4/7))
print('Average sales amount per month: ', assessment_data['purchase_amount'].sum() / 4)


# In[ ]:


print('Average sales amount per user: ', assessment_data['purchase_amount'].sum() / len(assessment_data['user_id'].unique()))

amounts = []
users = assessment_data['user_id'].unique()

for user in users:
    amounts.append(assessment_data[assessment_data['user_id'] == user]['purchase_amount'].sum())

print('Minimum of total amount of purchases of users: ', np.min(amounts))    
print('Median of total amount of purchases of users: ', np.median(amounts))
print('Maximum of total amount of purchases of users: ', np.max(amounts))

