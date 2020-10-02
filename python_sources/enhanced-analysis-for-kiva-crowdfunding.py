#!/usr/bin/env python
# coding: utf-8

# **Dataset Explained**
# 
# Kiva is actually a non profit organization which helps other low income organization or students from around the world to help them in their financial needs. The loan amount is funded via the internet.  The major dataset is "kiva_loans" which provides major details about the loans given to different organizations or persons by Kiva. The Fields under this dataset are explained below:
# 
# * **id** -  This is identity value for each row in the dataset.
# * **funded_amount** -  It is the amount which is funded by Kiva. As it is not necessary that all the amount asked for loan will be funded as whole.
# * **loan_amount** -  It is the loan amount asked by the borrower from the organization.
# * **activity** -  It is the work in which the borrower is engaged in .
# * **sector** -  The sector to which the borrowing organization or person belongs to.
# * **use** -  It is the purpose for which loan amount is needed.
# * **country_code** -  This is country code to which borrower belongs to.
# * **country** -  Name of the country to which borrower belongs.
# * **region** -  It is the region inside the country where the organization or person resides.
# * **currency** -  currency in which loan is lended by Kiva.
# * **partner_id** -  These are the unique id provided for field partners
# * **posted_time** -  It is the date and time when the loan was posted on Kiva.
# * **disbursed_time** -  It is the date and time when the loan was disbursed to the borrower.
# * **funded_time** -  It is the date and time when the loan was funded completely.
# * **term_in_months** -  Duration in months after which the loan has to be returned by the borrower.
# * **lender_count** -  Total number of lenders who have colaboratively funded the amount.
# * **tags** -  These are tags which describes the category of loan or loan type.
# * **borrower_genders** -  This is list having the gender of all the borrowers involved in a loan.
# * **repayment_interval** -  How frequently the amount of loan will be paid by the borrower.
# * **date** -  Date on which loan was posted.
# 
# 
# **Approach**
# 
# I've basically find out 6 important features out of the available keys based on which we can try to understand the data. I'm projecting different insights based upon these 6 key features.
# Right now i've tried to show the data based on four features out of those which will be help us to understand meaningful information about the data.
# These features are 
# 
# * Country - The country of the borrower who is getting the money from Kiva
# * Sectors -  Different sectors to which the borrowing entity belongs
# * Repayment Interval - This the interval after which the borrower has to pay back some amount of the loan.
# * Activity - This is the activity for which loan has been taken by the borrower. This is a kind of sub sector.
# 
# Initially we'll have a look at the dataset and its features to get a better understanding of the approach that we'll be taking to analyse the data

# In[15]:


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Akshay Sharma
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec

plt.style.use('dark_background')
loans_data = pd.read_csv('../input/kiva_loans.csv')


one_million = 1000000
one_thousand = 1000
one_hundred = 100

print(loans_data.keys())
loans_data.head()


# Here we've defined two different functions GenderDataForSectors and countBorrowers which are used for finding the total borrowers,male borrower and female borrowers sector wise out of our data and count total number of borrowers overall respectively.

# In[4]:


#From these attributes we can see that there are 18 features are available for the given data
#Out of these 18 features we can majorly try to look data in terms of 6 features
# sector,country,currency,partern_id,borrower-genders,repayment_interval
# So we'll show the data based on these features
#Right now i've projected data only based on 2 features other features i'll project soon

def GenderDataForSectors(sectors):
    total_count = 0
    sector_male = []
    sector_female = []
    for sector in sectors:
       
        current_sector = groupby_sectors.get_group(sector)['borrower_genders']
        sector_data = countBorrowers(current_sector)
       
        total_count = total_count + sector_data[0]
        sector_male.append(sector_data[1])
        sector_female.append(sector_data[2])
    return [total_count,sector_male,sector_female]

def countBorrowers(sector):
    item_count = 0
    male_count = 0
    female_count = 0
    for item in sector.dropna() :
        array = [s.replace(' ','') for s in item.split(',')]
        temp_array = np.array(array)
        item_count = item_count + len(temp_array)
        male_count = male_count + len(temp_array[temp_array == 'male'])
        female_count = female_count + len(temp_array[temp_array == 'female'])

    return (item_count,male_count,female_count)


# **Sector-wise Data Analysis**
# 
# Here we're going to visualize the data available based on the first key feature "Sectors". I have tried to show four things based on sector-wise data.
#     1. Total amount funded for each sector (Amount is available in millions irrespective of currency)
#     2. Total loan amount for each sector( Simillar to funded amount it is also in  millions irrespective of currency)
#     3. The gender based distribution of borrowers in each sector
#     4. Male and Female distribution of total borrowers overall.
#    
# From this data we can see that there is slight variation betweenthe Funded Amount and Loan Amount. In most of the sectors both of these amounts are nearly equal but for  few sectors like Agriculture and Food it differs slightly. 
# The number of female applying for loan is quite higher than the male. The number of female borrower is almost more than double to the number of male borrowers in every sector.
# If we consider the overall borrowers then female's are almost four times the number of male borrowers.
#  

# In[5]:


#Sector-wise data visualization
groupby_sectors = loans_data.groupby('sector')
sectors = np.unique(loans_data['sector'])

fig,ax = plt.subplots(2,2,figsize=(10,9))

#Funded amount for different sectors (in Million $)
fund_amt = groupby_sectors.sum()['funded_amount']/one_million
ax[0][0].bar(sectors,fund_amt)
ax[0][0].set_title("Funded Amount (million $)")
for ticks in ax[0][0].get_xticklabels():
    ticks.set_rotation(90)

#Loan given to different sectors (in Million $)
loan_amt = groupby_sectors.sum()['loan_amount']/one_million
ax[0][1].bar(sectors,loan_amt)
ax[0][1].set_title("Loan Amount (million $)")
for ticks in ax[0][1].get_xticklabels():
    ticks.set_rotation(90)

#Loan amount based on gender of borrowers
bar_width = 0.4
x_ticks = np.arange(len(sectors))
borrower_count,male_count,female_count = GenderDataForSectors(sectors)
total_male = np.sum(male_count)
total_female = np.sum(female_count)
ax[1][0].bar(x_ticks,np.divide(male_count,one_hundred),width=bar_width,color="red",label="Male")    
ax[1][0].bar(x_ticks+bar_width,np.divide(female_count,one_hundred),width=bar_width,color="blue",label="Female")
ax[1][0].legend()

ax[1][0].set_xticks(x_ticks)
ax[1][0].set_xticklabels(sectors)

ax[1][0].set_title("Gender Distribution (in hundreds)")
for ticks in ax[1][0].get_xticklabels():
        ticks.set_rotation(90)
        
#Male and female division of borrowers
ax[1][1].set_title("Male Female distribution ")
ax[1][1].pie([total_male,total_female],labels=['male','female'],colors=['red','blue'],autopct = '%1.1f%%',shadow=True)

fig.tight_layout(h_pad=0.8)    
fig.suptitle("Projections Sector-wise")


    



# **Country-wise Data Analysis**
# 
# Here is the insights of data based on Country specific data. In terms of countries i have represented data which tells
#     1. Total number of loans given to borrowers of each country
#     2. Total amount of loan given to borrowers of each country
#     3. Maximum duration of loan repayment period for each country
#     
# From country-wise analysis of data we get to know that although countries like Paraguay and Peru are not getting much loans but the the amount of loans borrowed is quite high. Not only these two countries but almost every country except Phillipines is having higher amount of loan as compared to the total number of loans given. Loan repayment tenure for the borrowers of Dominican Republic is highest as compared to other countries and it is getting longer repayment tenure for the amount borrowed whereas countries like Phillipines it's quite opposite. It's having lower repayment tenure for the huge of amount loans taken by the borrower

# In[6]:


#Country-wise data visualization
plt.figure(figsize=(22,15))
groupby_country = loans_data.groupby('country')
countries = np.unique(loans_data['country'])
x_ticks = np.arange(len(countries))
bar_width = 0.6
bar_space = 0.2
ax_c1 = plt.subplot2grid((3,3),(0,0),colspan=2)
ax_c2 = plt.subplot2grid((3,3),(1,0),colspan=2)
ax_c3 = plt.subplot2grid((3,3),(2,0),colspan=2)

#Total number of loans given to borrowers of each country
loan_count_country = groupby_country.count()['id']
ax_c1.bar(x_ticks,np.divide(loan_count_country,one_thousand),width = bar_width)
ax_c1.set_xticks([])
ax_c1.set_title("Number of loans given per country (in thousands)")

#Total amount of loan given to borrowers of each country
loan_amt_country = groupby_country.sum()['loan_amount']    
ax_c2.bar(countries,np.divide(loan_amt_country,one_million),width=bar_width)
ax_c2.set_xticks([])
ax_c2.set_title("Amount of loan given per country (in millions)")

#Maximum duration of loan repayment period for each country
max_loan_tenure_country = groupby_country.max()['term_in_months']
ax_c3.bar(countries,max_loan_tenure_country,width=bar_width)
ax_c3.set_title("Max Loan Tenure (in months)")


for ticks in ax_c3.get_xticklabels():
    ticks.set_fontsize(14)
    ticks.set_rotation(90)
    

plt.tight_layout()
plt.subplots_adjust(top=0.96)
plt.suptitle('Country-wise Projection')



# **Analysis based on Repayment Interval**
# 
# Here i've presented some of the ways to look at the repayment interval feature 
# 
# * Different loan repayment interval contribution to the whole dataset
# * Top 5 countries who had maximum irregular repayment of loan
# * Gender distribution for male and female based on repayment interval
# 
# About half of the loan was repayed in monthly interval as you can see in the pie chart that 51.1% of the loan was repayed in this way. Irregular loan repayment was also done by large number of borrowers i.e. 38.3% while only 0.1% of borrowers choosen to pay the loan on weekly basis. Among the countries who have been ahead in terms of paying the loan on irregular basis Phillipines contribution in this is huge. Other countries such as Kenya and Uganda are nowhere near to Phillipines. More than 140 thousand loans were taken from the people of Phillipines. I've projected the male and female contribution to the loan repayment in different intervals where as we've already seen that the contribution of females are tramendously larger than men in all the categories.

# In[9]:


#Based on repayment-interval

repayment_interval_grp = loans_data.groupby('repayment_interval')

#Different loan repayment interval contribution to the whole dataset
total_loans = loans_data.shape[0]
loans_payment_type = np.unique(loans_data['repayment_interval'])
loan_count_payment_type = [loans_data[loans_data['repayment_interval']==payment_type].shape[0] for payment_type in loans_payment_type]

plt.figure(figsize=(11,8))
gs = gridspec.GridSpec(4,3)
ax_r1 = plt.subplot(gs[:,:2])
ax_r1.pie(loan_count_payment_type,labels=loans_payment_type,autopct="%1.1f%%",colors=['red','blue','green','yellow'])
ax_r1.set_title('Contribution based on repayment-interval')

#Top 5 countries who had maximum irregular repayment of loan
ax_r2 = plt.subplot(gs[:2,-1])
bar_width = 0.3
irregular_payments = repayment_interval_grp.get_group('irregular')
irregular_groupby_country_count = irregular_payments.groupby('country').count()['repayment_interval']
irregular_groupby_countries = irregular_payments.groupby('country').count().index
irregular_data = pd.DataFrame({'country':irregular_groupby_countries,
                              'irregular_loans':np.array(irregular_groupby_country_count)})
irregular_data.sort_values(by='irregular_loans',inplace=True)

ax_r2.bar(irregular_data.country[-5:],irregular_data.irregular_loans[-5:]/one_thousand,width=bar_width,color=['red','orange','yellow','green','blue'])
ax_r2.set_title('Top 5 countries having irregular payment')
ax_r2.set_ylabel('No. of loans(in thousand)')

#Gender distribution for male and female based on repayment interval
ax_r3 = plt.subplot(gs[-2:,-1])
male_count_by_paymenttype = []
female_count_by_paymenttype = []
for pmt_type in loans_payment_type :
    pmt_type_data = loans_data[loans_data['repayment_interval'] == pmt_type]
    pmt_type_borrower_gender = pmt_type_data['borrower_genders']
    _,male_cnt,female_cnt =countBorrowers(pmt_type_borrower_gender)
    male_count_by_paymenttype.append(male_cnt)
    female_count_by_paymenttype.append(female_cnt)


xticks = np.arange(len(loans_payment_type))
ax_r3.bar(xticks,np.divide(male_count_by_paymenttype,one_thousand),label='male',width=bar_width,color='red')
ax_r3.bar(xticks+(bar_width),np.divide(female_count_by_paymenttype,one_thousand),label='female',width=bar_width,color='blue')
ax_r3.set_xticklabels(['','bullet','monthly','irregular','weekly'])
ax_r3.set_title('Gender dist in different repayment interval')
ax_r3.set_ylabel('No.of borrowers(in thousand)')
ax_r3.legend()
plt.tight_layout()


# **Analysis of Activities**
# 
# Here is the list of top 10 and bottom 10 activities for which loan has been borrowed.
# 
# We can see here that least amount of loan has been taken for Celebrations and Adult Care which is about 4 thousand and 2 thoussand respectively. In the list of top 10 activities, Farming,General Store and Agriculture have been the activities for which most of the loans were funded by Kiva. As all these three activities are related to food, we can say that most of the funding is required by the areas which are related to food. Not only this , all top activities are related to basic human needs. It seems that entrepreneurs are also focusing on the businesses which are related to basic human needs.

# In[16]:


#Visualization based on activities
grpby_activity = loans_data.groupby('activity').sum()
grpby_activity_loan_data = pd.DataFrame({'activity':grpby_activity.index,'total_loan_amt':grpby_activity['loan_amount']})
grpby_activity_loan_data.sort_values(by='total_loan_amt',inplace=True)

#Last 10 activities for which loan has been borrowed
plt.figure(figsize=(10,5))
plt.bar(grpby_activity_loan_data.activity[:10],np.divide(grpby_activity_loan_data['total_loan_amt'][:10],one_thousand),width=bar_width,color='red')
plt.xticks(rotation=90)
plt.yticks(np.arange(0,50,5))
plt.suptitle('Bottom 10 Activities (in loan amount)')
plt.ylabel('Total Loan Amount (in thousand dollars $')

#Top 10 activities for which loan has been borrowed
plt.figure(figsize=(10,5))
plt.bar(grpby_activity_loan_data.activity[-10:],np.divide(grpby_activity_loan_data['total_loan_amt'][-10:],one_million),width=bar_width,color='green')
plt.xticks(rotation=90)
plt.suptitle('Top 10 Activities (in loan amount)')
plt.ylabel('total Loan Amount (in million dollars $)')
#plt.xticks(range(5),last_five)


# 

# 

# 

# 
