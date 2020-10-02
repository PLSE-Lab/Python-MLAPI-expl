#!/usr/bin/env python
# coding: utf-8

# # Payroll Data for the City of New York
# 
# <p align="center">
#   <img src="https://cdn.vox-cdn.com/thumbor/HV3csUPr4CK3UWm_eTJA5pmwG-c=/0x0:2200x1385/1200x800/filters:focal(924x517:1276x869)/cdn.vox-cdn.com/uploads/chorus_image/image/49853921/6sqft_skyline_2020.0.jpg"/>
# </p>
# 
# ## This is a dataset hosted by the City of New York. 
# 
# **Contents:**
# 
# 1. [Pay over the years - Distribution](#years_distro)
# 2. [Basis of Pay - Anuual, Daily, Hourly](#pay_basis)
# 3. [Highest & the Lowest Paying Agencies - Annually, Daily & Hourly](#agcy_high_low)
# 4. [Highest & the Lowest Paying Job Titles - Annually, Daily & Hourly](#title_high_low)
# 5. [Pay by Location](#loc_pay)
# 6. [Overtime Compensation](#overtime)
# 
# Let us deep dive into it by gathering some necessary library imports & take a sneak peak into the data!

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
import matplotlib
#matplotlib.rc['font.size'] = 9.0
matplotlib.rc('font', size=20)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('axes', labelsize=20)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('legend', fontsize=20)
matplotlib.rc('figure', titlesize=20)
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# # Payroll Information
# 
# Four (4) columns correspond to the actual $payroll information namely -
# 1. Base Salary
# 2. Regular Gross Pay (Base Salary + Allowances)
# 3. Over Time Pay (Total OT Paid)

# In[ ]:


data = pd.read_csv("../input/citywide-payroll-data-fiscal-year.csv")
data.sample(10)


# To keep things simple, I create another column `Total Pay` which is simply the sum of Over Time & Regular Pay.

# In[ ]:


data = data.drop(['Last Name','First Name'], axis=1)
data['Total Pay'] = data['Regular Gross Paid'] + data['Total OT Paid']


# <a id='years_distro'></a>
# # Starting things simple!
# 
# Let's visualize how the **total pay** has fared over the **years** in the form of a distribution plot.

# In[ ]:


data['Fiscal Year'] = data['Fiscal Year'].astype(str)
plt.figure(figsize=(8,5))
g = sns.FacetGrid(data, hue='Fiscal Year', size=10, hue_order=['2014',
                                                              '2015',
                                                              '2016','2017'], palette="Paired")
g.map(sns.kdeplot, "Total Pay", shade=True)
g.set_xticklabels(rotation=45)
g.add_legend()
plt.show()


# In[ ]:


data['Pay Basis'].unique()


# <a id='pay_basis'></a>
# # Pay Basis
# 
# Since there is a fair bit of difference in the payroll information based on the basis of pay - 
# 1. Pay by Annum (Annualy)
# 2. Pay by the Day
# 3. Pay by the Hour
# 
# So I thought it would be a good idea to segregate this data in order to analyse them differently since there is going to be a significant difference in the thier types of employers as well.

# In[ ]:


data_per_annum = data[data['Pay Basis'].isin([' per Annum',
                                           ' Prorated Annual',
                                           'per Annum','Prorated Annual'])].drop('Pay Basis',
                                                                                axis=1)
data_per_hour = data[data['Pay Basis'].isin([' per Hour',
                                           'per Hour'])].drop('Pay Basis', axis=1)
data_per_day = data[data['Pay Basis'].isin([' per Day',
                                           'per Day'])].drop('Pay Basis', axis=1)

print ("Per Annum Basis --> ",data_per_annum.shape,
       "\nPer Day Basis -- >", data_per_day.shape,
       "\nPer Hour Basis -- >", data_per_hour.shape)


# In[ ]:


dist_pay_type = [data_per_annum.shape[0], data_per_day.shape[0], data_per_hour.shape[0]]
plt.figure(figsize=(7,7))
plt.pie(dist_pay_type, labels=['Per Annum','Per Day','Per Hour'],
                  autopct='%1.1f%%', shadow=True, startangle=90,
             colors=['#66b3ff','#ff9999','#99ff99'])
plt.title("Pay Type Basis in the city of New York")
plt.show()


# In[ ]:


def plot_high_low_pay(col, count, pay_basis):
    
    if (pay_basis=='Annum'):
        highest_paying_annum = data_per_annum.groupby(str(col))['Total Pay'].mean().reset_index().sort_values('Total Pay', ascending=False).head(count)
        lowest_paying_annum = data_per_annum.groupby(str(col))['Total Pay'].mean().reset_index().sort_values('Total Pay', ascending=True).head(count)

        f, ax = plt.subplots(2,1,figsize=(20,25))
        ax1 = sns.barplot(x='Total Pay', y=str(col), data=highest_paying_annum, 
                      orient='h', ax=ax[0])
        ax1 = sns.barplot(x='Total Pay', y=str(col), data=lowest_paying_annum, 
                      orient='h', ax=ax[1])
        ax[0].set_xlabel("Average Total Pay")
        ax[1].set_xlabel("Average Total Pay")
        plt.show()
    elif (pay_basis == 'Day'):
        highest_paying_day = data_per_day.groupby(str(col))['Total Pay'].mean().reset_index().sort_values('Total Pay', ascending=False).head(count)
        lowest_paying_day = data_per_day.groupby(str(col))['Total Pay'].mean().reset_index().sort_values('Total Pay', ascending=True).head(count)

        f, ax = plt.subplots(2,1,figsize=(20, 25))
        ax1 = sns.barplot(x='Total Pay', y=str(col), 
                          data=highest_paying_day, orient='h', ax=ax[0])
        ax1 = sns.barplot(x='Total Pay', y=str(col), 
                          data=lowest_paying_day, orient='h', ax=ax[1])
        ax[0].set_xlabel("Average Total Pay")
        ax[1].set_xlabel("Average Total Pay")
        plt.show()
    elif (pay_basis=='Hour'):
        highest_paying_hour = data_per_hour.groupby(str(col))['Total Pay'].mean().reset_index().sort_values('Total Pay', ascending=False).head(count)
        lowest_paying_hour = data_per_hour.groupby(str(col))['Total Pay'].mean().reset_index().sort_values('Total Pay', ascending=True).head(count)

        f, ax = plt.subplots(2,1,figsize=(20, 25))
        ax1 = sns.barplot(x='Total Pay', y=str(col), 
                          data=highest_paying_hour, orient='h', ax=ax[0])
        ax1 = sns.barplot(x='Total Pay', y=str(col), 
                          data=lowest_paying_hour, orient='h', ax=ax[1])
        ax[0].set_xlabel("Average Total Pay")
        ax[1].set_xlabel("Average Total Pay")
        plt.show()


# <a id='agcy_high_low'></a>
# # Highest & the Lowest Paying Agencies 
# 
# ### Pay by Annum

# In[ ]:


plot_high_low_pay(col='Agency Name', count=10, pay_basis='Annum')


# #### Pay by Day

# In[ ]:


plot_high_low_pay(col='Agency Name', count=10, pay_basis='Day')


# #### Pay by Hour

# In[ ]:


plot_high_low_pay(col='Agency Name', count=10, pay_basis='Hour')


# <a id='title_high_low'></a>
# # Jobs that pay the Most & Least money in NYC
# 
# #### By Annum

# In[ ]:


plot_high_low_pay(col='Title Description', count=20, pay_basis='Annum')


# #### By Day

# In[ ]:


plot_high_low_pay(col='Title Description', count=20, pay_basis='Day')


# #### By Hour

# In[ ]:


plot_high_low_pay(col='Title Description', count=20, pay_basis='Hour')


# In[ ]:


data['Work Location Borough'] = data['Work Location Borough'].str.strip().str.upper()
location_pay = data.groupby('Work Location Borough')['Total Pay'].mean().reset_index().sort_values('Total Pay',ascending=False)


# <a id='loc_pay'></a>
# # Average pay by Location
# 
# **`Work Location Borough`** tells us the area that a particular agency in NYC belongs to so it is interesting to know which one of these areas have the highest pay!

# In[ ]:


sns.set_style("whitegrid")
plt.figure(figsize=(20,7))
sns.boxplot(x=data['Work Location Borough'], y=data['Total Pay'],
           data=location_pay, palette="coolwarm_r")
plt.xticks(rotation=90)
plt.show()


# <a id='overtime'></a>
# # Overtime Compensations
# 
# 1. Which agencies are making their employees work over time & by how much?
# 2. What is the average OverTime pay in these agencies?

# In[ ]:


data['Agency Name'] = data['Agency Name'].str.strip().str.upper()
ot_ = data.groupby('Agency Name')['OT Hours'].mean().reset_index().sort_values('OT Hours',ascending=False)
ot_ = ot_.head(10)
ot_pay = data.groupby('Agency Name')['Total OT Paid'].mean().reset_index().sort_values('Total OT Paid',ascending=False)
ot_pay = ot_pay.head(10)


# In[ ]:


#sns.set_style("whitegrid")
f, ax = plt.subplots(2,1, figsize=(20,13))
sns.barplot(y=ot_['Agency Name'], x=ot_['OT Hours'],
           data=ot_, palette="BuGn_r", orient='h',ax=ax[0])
sns.barplot(x=ot_pay['Agency Name'], y=ot_pay['Total OT Paid'],
           data=ot_pay, palette="ocean_r", ax=ax[1])
ax[0].set_xlabel("Average Over Time Hours")
ax[1].set_ylabel("Average Over Time Salary")
plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=90)
plt.show()


# **Fire Dept & Dept of correction** occupy the top 2 spots in the highest number of average overtime hours.

# ### That'd be all for now, I'll continue adding some more visuals as I explore this data. 
#  
# **Let me know what you guys think in the comments below or if you any suggestions on how this EDA can be further extended, feel free to describe it in the comments!!**
