#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


kiva = pd.read_csv ('/kaggle/input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
kiva.head()


# In[ ]:


kiva.info()


# In[ ]:


kiva['country']


# In[ ]:


kiva['country'].unique()


# In[ ]:


kiva['country'].nunique()


# In[ ]:


kiva.isna().any()


# In[ ]:


kiva.isna().sum()


# In[ ]:


kiva.describe()


#  # Reading Data for Benin

# In[ ]:


Benin = kiva[kiva['country'] =='Benin']
Benin.head()


# View Data

# In[ ]:


Benin.tail()


# In[ ]:


Benin.info()


# In[ ]:


Benin.describe()


# In[ ]:


Benin.columns


# *Getting the data types in each column*

# In[ ]:


Benin.dtypes


# # Checking for missing values

# In[ ]:


Benin.isna()


# *Checking for columns with missing values*

# In[ ]:


Benin.isna().any()


# *****Confirming whether there is a column with all values missing*

# In[ ]:


Benin.isna().all()


# * *From the information above, we confirm that no column has all the values missing

# *****Getting a total of missing values per column*

# In[ ]:


Benin.isna().sum()


# In[ ]:


Benin


# In[ ]:


Benin.set_index('id', inplace=True)
Benin


# In[ ]:


Benin.reset_index()


# # Determining amount of loan allocated per sector per region

# In[ ]:


Benin_sector_region = Benin.groupby(['region', 'sector'])['loan_amount','lender_count'].sum().sort_values(by = 'loan_amount', ascending = False).reset_index() 
Benin_sector_region.head()


# * From the above information, Parakou region received the highest loan amount for the agriculture sector

# In[ ]:


Benin_sector_region.describe()


# In[ ]:


Benin_sector_region['sector'].unique()


# In[ ]:


Benin_sector_region['sector'].nunique()


# * Benin's data has only 2 regions reflected

# # **Grouping data based on the different sectors**

# In[ ]:


Benin_sector = Benin.groupby('sector')
Benin_sector


# In[ ]:


for sector, sector_df in Benin_sector:
 print(sector)
print(sector_df)


# ** using get_group to get information per sector*****

# In[ ]:


Benin_sector.get_group('Food')


# In[ ]:


Benin_sector.get_group('Arts')


# In[ ]:


Benin_sector.mean()


# # **Getting the top 10 sectors that received loans**

# In[ ]:


Benin_sector_activity = Benin.groupby(['sector', 'activity'])['loan_amount','lender_count'].sum().sort_values(by = 'loan_amount', ascending = False).reset_index() 
Benin_sector_activity.head(10)


# * Upon sorting, we find out that Retail sector's retail activity has the highest loan amount while from below we find out that Arts sector's Craft activity has the least loan_amount.
# 

# In[ ]:


Benin_sector_activity = Benin.groupby(['sector', 'activity'])['loan_amount','lender_count'].sum().sort_values(by = 'loan_amount', ascending = False).reset_index() 
Benin_sector_activity.tail(10)


# # Exploring highest loan amounts received per gender

# In[ ]:


Benin_sector_activity = Benin.groupby(['sector', 'activity','borrower_genders'])['loan_amount','lender_count'].sum().sort_values(by = 'loan_amount', ascending = False).reset_index() 
Benin_sector_activity.head(10)


# * From the result obtained, Females in retail sector and activity received highest loan amount

# In[ ]:


Benin['borrower_genders'].unique()


# In[ ]:


Benin['borrower_genders'].nunique()


# * The information above shows more values in gender than expected so we need to clean up

# In[ ]:


def fix_gender(gender):
    gender = str(gender)
    if gender.startswith('f'):
        gender ='female'
    else:
        gender = 'male'
    return gender


# In[ ]:


Benin['borrower_genders'] = Benin['borrower_genders'].apply(fix_gender)


# In[ ]:


Benin['borrower_genders'].nunique()


# * Now we can get clear data of the gender representation of the loans acquired

# In[ ]:


Benin_sector_activity = Benin.groupby(['sector', 'activity','borrower_genders'])['loan_amount','lender_count'].sum().sort_values(by = 'borrower_genders', ascending = False).reset_index() 
Benin_sector_activity.head(10)


# * From the top 10 sectors based on loan amount, we confirm that Agricultural sector has the highest loan amount with the male gender as the borrower, hence we can say the male gender is the highest borrower.

# In[ ]:


Benin_sector_repayment_interval = Benin.groupby(['sector', 'activity','borrower_genders', 'repayment_interval'])['loan_amount', 'term_in_months'].sum().sort_values(by = 'loan_amount', ascending = False).reset_index() 
Benin_sector_repayment_interval.head()


# # Visualization of Benin Data

# In[ ]:


import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes =True)

import plotly.express as px


# In[ ]:


plt.xticks(rotation = 75) 
plt.xlabel('sector') 
plt.ylabel('loan_amount')
plt.plot(Benin['sector'], Benin['loan_amount'])
plt.show()


# In[ ]:


plt.figure(figsize = (10,5))
plt.xticks(rotation = 75) 
plt.xlabel('sector') 
plt.ylabel('loan_amount')
plt.plot(Benin['sector'], Benin['loan_amount'])
plt.show()


# In[ ]:


plt.title('Sectors loan allocation')

plt.xlabel('sector')
plt.ylabel('loan_amount')

plt.xticks(rotation = 75)

plt.bar(Benin['sector'], Benin['loan_amount'])

plt.show()


# * From the bar graph representation, it is clear that the Agricultural sector received the highest mount of loan

# In[ ]:


plt.title('Activity loan allocation')

plt.xlabel('activity')
plt.ylabel('loan_amount')

plt.xticks(rotation = 50000000)

plt.bar(Benin['activity'], Benin['loan_amount'])

plt.show()


# In[ ]:


plt.figure(figsize = (10,5))

plt.title('Loan Amount by Sector')

plt.xticks(rotation = 75)

sns.barplot(x = 'sector', y = 'loan_amount', data = Benin_sector_activity, ci = None);


# * We notice that Agricultural sector has the highest loan amounts given

# In[ ]:


plt.figure(figsize = (10,5))
plt.title('Loan Amount by Sector')
plt.xticks(rotation = 75)
sns.barplot(x = 'sector', y = 'loan_amount', data = Benin.head(1000), ci =None, hue = 'borrower_genders');


# * we determine agricultural sector to have the highest loan amount and with the male gender as the highest borrower 

# In[ ]:


plt.figure(figsize = (10,5))

plt.title('Loan Amount by Sector')

plt.xticks(rotation = 75)

sns.barplot(x = 'sector', y = 'loan_amount', data = Benin, ci = None, hue = 'repayment_interval');


# In[ ]:


Benin['date'].head(10)


# In[ ]:


def create_day(date):
    day = pd.DatetimeIndex(date).day
    return(day)

Benin['day'] = create_day(Benin['date'])
Benin[['date', 'day']]. head()


# In[ ]:


Benin[['loan_amount', 'lender_count']].max()


# In[ ]:


px.scatter(Benin, x = 'loan_amount', y = 'lender_count', color = 'sector', title = 'Loan Amount vs. Lender Count',
           hover_data = ['funded_amount', 'borrower_genders'],  log_x = True, range_x= [100,50000], range_y=[0,16000],
           labels = {'loan_amount' : 'Loan Amount', 'lender_count' : 'Lender Count'}, animation_frame = 'day')


# In[ ]:


px.scatter(Benin, x = 'loan_amount', y = 'lender_count', color = 'sector', title = 'Loan Amount vs. Lender Count',
           hover_data = ['funded_amount', 'borrower_genders'],  log_x = True, range_x= [100,50000], range_y=[0,16000],
           labels = {'loan_amount' : 'Loan Amount', 'lender_count' : 'Lender Count'}, animation_frame = 'day', size_max=60, size ='lender_count' ,facet_row='borrower_genders', facet_col='repayment_interval')


# In[ ]:


px.scatter(Benin, x = 'loan_amount', y = 'lender_count', color = 'sector', title = 'Loan Amount vs. Lender Count',
           hover_data = ['funded_amount', 'borrower_genders'],  log_x = True, range_x= [100,50000], range_y=[0,16000],
           labels = {'loan_amount' : 'Loan Amount', 'lender_count' : 'Lender Count'}, animation_frame = 'day', size_max=60, size ='lender_count' ,facet_col='borrower_genders', facet_row = 'repayment_interval')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




