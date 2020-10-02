#!/usr/bin/env python
# coding: utf-8

# # Playing with Lending Club's Loan Data

# In this kernel I will be going over the Lending Club Loan Data. The data has a lot of features, some are useful and some are not, some features have a lot of missing values too. I will try to clean this data and make it ready for a great predictive model. <br>
# Data Available at: https://www.kaggle.com/wendykan/lending-club-loan-data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.


# In[ ]:


loans = pd.read_csv("../input/loan.csv", low_memory=False) #Dataset


# In[ ]:


# Checking the Dimensions of our dataset

loans.shape


# In[ ]:


loans.columns


# Now we need to familiarize ourselves with this dataset, and we should understand what these columns represent.

# In[ ]:


description = pd.read_excel('../input/LCDataDictionary.xlsx').dropna()
description.style.set_properties(subset=['Description'], **{'width' :'850px'})


# In[ ]:


loans.info()


# In[ ]:


fig = plt.figure(figsize=(15,10))
sns.heatmap(loans.isna(),cmap='inferno')


# It is evident from the above heatmap that our dataset contains a lot of missing values and we can not use feature that has so many missing values.

# Another thing we would want to examine is that how many loans have a default loan status in comparison to other loans. A common thing to predict in datasets like these are if a new loan will get default or not. I'll be keeping loans with default status as my target variable.

# In[ ]:


loans['loan_status'].value_counts()


# In[ ]:


target = [1 if i=='Default' else 0 for i in loans['loan_status']]
loans['target'] = target
loans['target'].value_counts()


# In[ ]:


nulls = pd.DataFrame(round(loans.isnull().sum()/len(loans.index)*100,2),columns=['null_percent'])
#sns.barplot(x='index',y='null_percent',data=nulls.reset_index())
nulls[nulls['null_percent']!=0.00].sort_values('null_percent',ascending=False)


# There are several columns that fit into one of the following categories:
# 
# Unneccesary data such as the URL to view the loan <br>
# Redundant data. Loan description may be useful to some, but loan purpose fits our needs <br>
# User provided information. Employer titles may offer some insight into employment industry, but would need significant cleanup to provide useful statistics <br>
# Operational data. The next payment date for the loan at the time the data was generated is not relevant to us 

# In[ ]:


# Drop unneccesary columns
loans = loans.drop(['url', 'desc', 'policy_code', 'last_pymnt_d', 'next_pymnt_d', 'earliest_cr_line', 'emp_title'], axis=1)
loans = loans.drop(['id', 'title', 'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp', 'zip_code'], axis=1)


# In[ ]:


loans['member_id'].value_counts().head(5)


# Since no customer has taken loan again, we can drop member id too.

# In[ ]:


loans.drop(['member_id'], axis=1, inplace=True)


# Additionally, we identified some rows that we decided were not relevant to our needs. The 'loan_status' column is the source of our answer to the core question if people are paying the loans they take out. We found some records with a loan_status of  "Does not meet the credit policy". We believe these may be older loans that would simply not be accepted under LendingClubs current criteria. As these data points will provide no value moving forward, we've ecluded them from our data. Similiarily, recently issued loans could mislead our analysis, as no payment has been expected yet.

# In[ ]:


i = len(loans)
loans = pd.DataFrame(loans[loans['loan_status'] != "Does not meet the credit policy. Status:Fully Paid"])
loans = pd.DataFrame(loans[loans['loan_status'] != "Does not meet the credit policy. Status:Charged Off"])
loans = pd.DataFrame(loans[loans['loan_status'] != "Issued"])
loans = pd.DataFrame(loans[loans['loan_status'] != "In Grace Period"])
a = len(loans)
print(f"We dropped {i-a} rows, a {((i-a)/((a+i)/2))*100}% reduction in rows")


# # Exploratory Data Analysis

# Now I would like to know the distribution of Data types

# In[ ]:


# Number of each type of column
sns.set(rc={'figure.figsize':(15,5)})
sns.countplot(loans.dtypes,palette='viridis')
plt.title('Number of columns distributed by Data Types',fontsize=20)
plt.ylabel('Number of columns',fontsize=15)
plt.xlabel('Data type',fontsize=15)


# In[ ]:


# Let us see how many Object type features are actually Categorical
loans.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# We should have a look at the distribution of Employement Lengths too

# In[ ]:


sns.set(rc={'figure.figsize':(15,8)})
sns.countplot(loans['emp_length'],palette='inferno')
plt.xlabel("Length")
plt.ylabel("Count")
plt.title("Distribution of Employement Length For Issued Loans")
plt.show()


# Well, it can be seen that people who have worked for 10 or more years are more likely to take loans

# In[ ]:


sns.set(rc={'figure.figsize':(15,10)})
sns.violinplot(x="target",y="loan_amnt",data=loans, hue="pymnt_plan", split=True,palette='inferno')
plt.title("Payment plan - Loan Amount", fontsize=20)
plt.xlabel("TARGET", fontsize=15)
plt.ylabel("Loan Amount", fontsize=15);


# As expected, people who have defaulted their loans had no payment plan

# In[ ]:


sns.set(rc={'figure.figsize':(15,6)})
sns.boxplot(x='loan_amnt', y='loan_status', data=loans)


# Now I would like to know what kind of loans Lending Club usually issues?
# I guess this can be answered by having look at Loan Grade that Lending Club assigns.

# In[ ]:


sns.set(rc={'figure.figsize':(15,6)})
sns.countplot(loans['grade'], palette='inferno')


# Since most of the loans are of B Grade, we should have a look at their loan amounts too

# In[ ]:


loan_grades = loans.groupby("grade").mean().reset_index()

sns.set(rc={'figure.figsize':(15,6)})
sns.barplot(x='grade', y='loan_amnt', data=loan_grades, palette='inferno')
plt.title("Average Loan Amount - Grade", fontsize=20)
plt.xlabel("Grade", fontsize=15)
plt.ylabel("Average Loan Amount", fontsize=15);


# Average loan amount of B grade loans is the least of all grades. I guess the higher the grade, lower the loan amount is issued

# We should have a look at the interest rates offered for these loan grades.

# In[ ]:


sns.set(rc={'figure.figsize':(15,10)})
sns.violinplot(x="grade", y="int_rate", data=loans, palette='viridis', order="ABCDEFG",hue='target',split=True)
plt.title("Interest Rate - Grade", fontsize=20)
plt.xlabel("Grade", fontsize=15)
plt.ylabel("Interest Rate", fontsize=15);


# I think my previous assumption was right. The higher the loan amount, higher the interest rate.

# In[ ]:


sns.set(rc={'figure.figsize':(15,10)})
sns.violinplot(x="target",y="loan_amnt",data=loans, hue="application_type", split=True,palette='viridis')
plt.title("Application Type - Loan Amount", fontsize=20)
plt.xlabel("TARGET", fontsize=15)
plt.ylabel("Loan Amount", fontsize=15);


# So most of the defaulted loans were issued to individuals. Two or more people who take loans have lower chances of defaulting

# Let us see the overall distribution of interest rates now.

# In[ ]:


sns.set(rc={'figure.figsize':(15,5)})
sns.kdeplot(loans.loc[loans['target'] == 1, 'int_rate'], label = 'target = 1',shade=True)
sns.kdeplot(loans.loc[loans['target'] == 0, 'int_rate'], label = 'target = 0',shade=True);
plt.xlabel('Interest Rate (%)',fontsize=15)
plt.ylabel('Density',fontsize=15)
plt.title('Distribution of Interest Rate',fontsize=20);


# To which state most of the defaulted loan cases belong?

# In[ ]:


state_default = loans[loans['target']==1]['addr_state']

sns.set(rc={'figure.figsize':(15,5)})
sns.countplot(state_default, order=state_default.value_counts().index, palette='viridis')
plt.xlabel('State',fontsize=15)
plt.ylabel('Number of loans',fontsize=15)
plt.title('Number of defaulted loans per state',fontsize=20);


# In[ ]:


state_non_default = loans[loans['target']==0]['addr_state']

sns.set(rc={'figure.figsize':(15,5)})
sns.countplot(state_non_default, order=state_non_default.value_counts().index, palette='viridis')
plt.xlabel('State',fontsize=15)
plt.ylabel('Number of loans',fontsize=15)
plt.title('Number of not-defaulted loans per state',fontsize=20);


# ### Summary <br> <br>
# Since most of the customers have been employed for 10+ years, the **majority of Lending Club's customers are 30+ years of age**.  <br>
# Interest rate varies wildly, reaching **nearly 30%** for high-risk loans <br>
# Grade A has the **lowest interest rate** around 7% <br>
# Grade G has the **highest interest rate** above 25% <br>
# The **lower the grade, the higher loan amount** loan issued <br>
# Fully Paid loans tend to be smaller. This could be due to the age of the loans <br>
# Default has the highest count among other loan status. <br>
# **In Grace Period** and **Late(16~30 days)** have the highest loan amount and mean. <br>
# Most of the loans have interest rates between **12% and 18%** <br>
# All the loans that have been defaulted are from **individuals** rather than from two or more people. <br>
# **California** has the most defaulted and non-defaulted loans out of all the states in US <br>
# **States** are not a distinguishing feature for predicting the defaulted loans.

# # Cleaning The Data

# In[ ]:


loans.columns


# As I mentioned earlier, there are some columns/features that are not required. We have already dropped those features.

# In[ ]:


loans.shape


# In[ ]:


loans.head(5)


# Now I am going to drop columns that have more than 75% null values

# In[ ]:


nulls = pd.DataFrame(round(loans.isnull().sum()/len(loans.index)*100,2),columns=['null_percent'])
drop_cols = nulls[nulls['null_percent']>75.0].index
loans.drop(drop_cols, axis=1, inplace=True)


# In[ ]:


loans.shape


# In[ ]:


loans.head(5)


# In[ ]:


loans.columns


# Now we should convert date object columns to integer years or months so that we can easily encode other categorical features without exhausting our resources. For filling the dates, I am gonna use the most used dates in that feature

# In[ ]:


loans['issue_d']= pd.to_datetime(loans['issue_d']).apply(lambda x: int(x.strftime('%Y')))
loans['last_credit_pull_d']= pd.to_datetime(loans['last_credit_pull_d'].fillna("2016-01-01")).apply(lambda x: int(x.strftime('%m')))


# In[ ]:


loans.drop(['loan_status'],axis=1,inplace=True)


# In[ ]:


categorical = []
for column in loans:
    if loans[column].dtype == 'object':
        categorical.append(column)
categorical


# In[ ]:


loans = pd.get_dummies(loans, columns=categorical)


# In[ ]:


loans.shape


# In[ ]:


loans['mths_since_last_delinq'].fillna(loans['mths_since_last_delinq'].median(), inplace=True)


# In[ ]:


# Finally we are going to drop all the rows that contain null values
loans.dropna(inplace=True)


# In[ ]:


sns.set(rc={'figure.figsize':(15,8)})
sns.heatmap(loans.isna(),cmap='inferno')

