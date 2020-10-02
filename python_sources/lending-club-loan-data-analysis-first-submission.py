#!/usr/bin/env python
# coding: utf-8

# ## Lending Club Loan Data Analysis
# 

# ### Data Cleaning and Exploratory Analysis 
# 
# In machine learning, you clean up the data and turn raw data into features from which you can derive the pattern. There are methods available to extract features that will be covered in upcoming sessions but it's very important to build the intuition. The process of data cleaning and visualization helps with that. In this assignment, we will try to manually identify the important features in the given dataset. 
# 
# ### Dataset: Lending Club data
# 
# https://www.lendingclub.com/info/download-data.action
# 
# Years of data to download: 2007-2011
# 
# Load the Lending Club data into a pandas dataframe. The data contains 42538 rows and 145 columns. Not all these columns contain meaningful (or any) information so they need to be cleaned. The loans are categorized into different grades and sub-grades. It would be interesting to see whether they have any impact on the interest rates or not.
# The process should lead us into default prediction, and finding the columns that directly predict how the loan will behave. These would be our most important features.
# 
# 
# 
# 
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, datetime

import warnings
warnings.filterwarnings(action="ignore")
import os
print(os.listdir("../input"))


# In[ ]:


loanData= pd.read_csv('../input/LoanStats.csv',header=1,error_bad_lines=False,skipfooter=2,engine='python')
loanData.isnull().all()
loanData.dropna(axis=1,how='all',inplace=True)  # drop all null columns
pd.set_option('display.max_columns', 65)
loanData.head()


# In[ ]:


loanData.drop(['emp_title','desc','title','application_type'],axis=1,inplace=True) #carries no useful info which can affect loan terms
loanData['id']=np.arange(1,42537)


# In[ ]:


print(loanData.policy_code.unique())  # drop it since it contains only 1 value so doesn't make any impact
print(loanData.loan_status.unique())
print(loanData.home_ownership.unique())
print(loanData.verification_status.unique())
print(loanData.pymnt_plan.unique())  ## drop it since it contains only 1 value i.e 'n' , so doesn't make any impact
print(loanData.disbursement_method.unique())   # drop it since it contains only 1 value i.e 'Cash' , so doesn't make any impact
print(loanData.hardship_flag.unique())         # drop it since it contains only 1 value i.e 'N' , so doesn't make any impact
print(loanData.tax_liens.unique())
print(loanData.pub_rec_bankruptcies.unique())
print(loanData.initial_list_status.unique())   #  drop it since it contains only 1 value i.e 'f' , so doesn't make any impact



# In[ ]:


loanData.drop(['policy_code','pymnt_plan','disbursement_method','hardship_flag','initial_list_status'],axis=1,inplace=True) # since contain only 1 type of value so doesn't make any impact


# ### int_rate to numeric

# In[ ]:


loanData['int_rate']=loanData.int_rate.str.extract('(\d+.\d+)')

loanData['int_rate']=loanData.int_rate.astype('float64')
loanData.int_rate.dtype


# ### cleaning loan_status and verification_status columns

# In[ ]:


loanData.loc[loanData.loan_status.str.contains('Fully Paid',na=False),'loan_status']='Paid'   
loanData.loc[loanData.loan_status.str.contains('Charged Off',na=False),'loan_status']='ChargedOff'  
loanData.loan_status.unique()


# In[ ]:


loanData['loan_status'] =loanData.loan_status.astype('category')
loanData.loan_status.unique()


# In[ ]:


loanData.loc[loanData.verification_status.str.contains('Not Verified',na=False),'verification_status']='NotVerify'  
loanData.loc[loanData.verification_status.str.contains('Verified',na=False),'verification_status']= 'Verify'   


# In[ ]:


loanData['verification_status'] =loanData.verification_status.astype('category')
loanData.verification_status.unique()


# ### Extracting useful info from columns by cleaning 

# In[ ]:


loanData['term'] =loanData.term.str.extract('(\d+)')
loanData['emp_length'] =loanData.emp_length.str.extract('(\d+)')
loanData['revol_util'] =loanData.revol_util.str.extract('(\d+.\d+)')
loanData['sub_grade']=loanData.sub_grade.str.extract('(\d+)')


# In[ ]:


loanData.term.dropna(inplace=True)
loanData['term']=loanData.term.astype('int')
#loanData['grade']=loanData.grade.astype('category')
loanData['home_ownership']=loanData.home_ownership.astype('category')
loanData['revol_util']=loanData.revol_util.astype('float')
#loanData['sub_grade']=loanData.sub_grade.astype('int')


# In[ ]:


loanData.emp_length.fillna(0,inplace=True)
loanData['emp_length']=loanData.emp_length.astype('int')
loanData.emp_length.unique()


# ### handling date columns

# In[ ]:


# handling date columns

loanData['issue_d'] = pd.to_datetime(loanData['issue_d'])
loanData['earliest_cr_line'] = pd.to_datetime(loanData['earliest_cr_line'])
loanData['last_pymnt_d'] = pd.to_datetime(loanData['last_pymnt_d'])
loanData['last_credit_pull_d'] = pd.to_datetime(loanData['last_credit_pull_d'])
loanData['settlement_date'] = pd.to_datetime(loanData['settlement_date'])
loanData['debt_settlement_flag_date'] = pd.to_datetime(loanData['debt_settlement_flag_date'])
loanData['next_pymnt_d'] = pd.to_datetime(loanData['next_pymnt_d'])

             


# ### handling sparsely populated column having less than 200 values

# In[ ]:


loanData.loc[loanData.settlement_amount.notnull(),['loan_amnt','issue_d','loan_status','settlement_status','settlement_amount','settlement_percentage','settlement_term','settlement_date','debt_settlement_flag_date ']]


# In[ ]:


loanData.drop(['debt_settlement_flag_date','settlement_term','settlement_status','settlement_date','settlement_percentage','settlement_amount'],axis=1,inplace=True) # since doesn't have enough values 


# In[ ]:


loanData.tax_liens.value_counts()


# In[ ]:


loanData.drop('tax_liens',axis=1,inplace=True)


# ### Relation between 'loan_amnt', 'funded_amnt', 'funded_amnt_inv'  fields
#        loan_amnt >= funded_amnt >= funded_amnt_inv

# In[ ]:


df=loanData.loc[:30000,['loan_amnt','funded_amnt','funded_amnt_inv']]
sns.pairplot(vars=['loan_amnt','funded_amnt','funded_amnt_inv'],data=df)


# ### % of loans for different purposes

# In[ ]:


df=loanData.groupby('purpose').id.count().reset_index()
df.rename(columns={'id':'no_of_loans'},inplace=True)
plt.figure(figsize=(10,10))
plt.pie(df.no_of_loans,labels=df.purpose,autopct='%.2f%%');
df.plot.bar(x='purpose',y='no_of_loans',figsize=(10,6));


# ### Geographical distribution of loans by state - CA state receives maximum loans

# In[ ]:


df=loanData.groupby('addr_state').id.count().sort_values().reset_index()
df.plot.bar('addr_state','id',figsize=(15,6))
plt.title('loan distribution by state')
df.head()


# ### Trend in Interest Rate by Year
#       
#        highest interest rate average in year 2009 and then great fall till during 2009-10

# In[ ]:


df=loanData.groupby(loanData.issue_d.dt.year).int_rate.mean()
df.plot(kind='line',figsize=(10,6))
plt.title('Change in Avg. Interest Rate by year');



# ### Yearly loan distribution
#       
#        # max loans are taken in 2011

# In[ ]:


df=loanData.groupby(loanData.issue_d.dt.year).id.count()
df.plot(kind='line',figsize=(10,6))
plt.title('Total loans taken per year');
plt.ylabel('No of loans issued')
df.head()


# In[ ]:


df=loanData.groupby(loanData.issue_d.dt.month).id.count().sort_values()
df.plot(kind='line',figsize=(8,6));
plt.title('Total No of loans taken by month ')
plt.xlabel('Month');
plt.ylabel('No of loans issued');

# maximum loans are taken during end of the year


# ### total loans paid in each grade

# In[ ]:


df=loanData.groupby('grade').loan_status.value_counts().unstack()
df.plot.bar()
plt.title('No of loans paid and charged_off according to grade');


# ### Loan Grade and subgrade Distribution (by interest rate)
#      
#      subgrades are distributed according to interest rates for each grade category i.e increase in subgrade with increase in int_rate for a particular grade category 

# In[ ]:


loanData.pivot_table(index='grade',columns='sub_grade',values='int_rate').plot.bar(figsize=(10,6))
plt.ylabel('Interest rate');


# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(data=loanData, x='grade',y='int_rate',hue='loan_status')
plt.title('Interest Rate IQR(range) with grade');


# ### loan status according to home ownership

# In[ ]:


df=loanData.groupby('home_ownership').loan_status.value_counts().unstack()
df.plot.bar(figsize=(10,6))
plt.title('No of loans paid and charged off according to home ownership')
df


# ### Number of loans with verified source

# In[ ]:


sns.countplot(data=loanData,x='verification_status')
plt.title('Source verification of loans');


# ### loan distribution by its amount
#     Maximum loans are taken in range $ 5000-15000 

# In[ ]:


plt.figure(figsize=(10,6))

sns.distplot(loanData.loan_amnt.fillna(0))


# This is my first ever submission on Kaggle. This was all based on pandas and visualisation library, I am yet to learn machine learning models hope to come with more analysis in near future using different techniques.

# In[ ]:





# In[ ]:




