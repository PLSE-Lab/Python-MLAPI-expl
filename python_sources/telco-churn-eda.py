#!/usr/bin/env python
# coding: utf-8

# # Telco Churn Analysis

# **Dataset Info:**
# IBM Sample Data Set containing Telco customer data and showing customers left last month

# In[ ]:


#import the required libraries
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.ticker as mtick  
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


sns.set(style = 'white')


# **Load the data file **

# In[ ]:


telco_base_data = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# Look at the top 5 records of data

# In[ ]:


telco_base_data.head()


# Check the various attributes of data like shape (rows and cols), Columns, datatypes

# In[ ]:


telco_base_data.shape


# In[ ]:


telco_base_data.columns.values


# In[ ]:


# Checking the data types of all the columns
telco_base_data.dtypes


# In[ ]:


# Check the descriptive statistics of numeric variables
telco_base_data.describe()


# SeniorCitizen is actually a categorical hence the 25%-50%-75% distribution is not propoer
# 
# 75% customers have tenure less than 55 months
# 
# Average Monthly charges are USD 64.76 whereas 25% customers pay more than USD 89.85 per month

# ## Data Cleaning
# 

# **1.** Create a copy of base data for manupulation & processing

# In[ ]:


telco_data = telco_base_data.copy()


# **2.** Total Charges should be numeric amount. Let's convert it to numerical data type

# In[ ]:


telco_data.TotalCharges = pd.to_numeric(telco_data.TotalCharges, errors='coerce')
telco_data.isnull().sum()


# **3.** As we can see there are 11 missing values in TotalCharges column. Let's check these records 

# In[ ]:


telco_data.loc[telco_data ['TotalCharges'].isnull() == True]


# **4. Missing Value Treatement**

# Since the % of these records compared to total dataset is very low ie 0.15%, it is safe to ignore them from further processing.

# In[ ]:


#Removing missing values 
telco_data.dropna(how = 'any', inplace = True)


# **5.** Devide customers into bins based on tenure e.g. for tenure < 12 months: assign a tenure group if 1-12, for tenure between 1 to 2 Yrs, tenure group of 13-24; so on...

# In[ ]:


# Get the max tenure
print(telco_data['tenure'].max()) #72

# Group the tenure in bins of 12 months
labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

telco_data['tenure_group'] = pd.cut(telco_data.tenure, range(1, 80, 12), right=False, labels=labels)


# **6.** Remove columns not required for processing

# In[ ]:


#drop column customerID and tenure
telco_data.drop(columns= ['customerID','tenure'], axis=1, inplace=True)
telco_data.head()


# ## Data Exploration
# **1. ** Plot distibution of individual predictors by churn

# In[ ]:


for i, predictor in enumerate(telco_data.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=telco_data, x=predictor, hue='Churn')


# **2.** Convert the target variable 'Churn'  in a binary numeric variable i.e. Yes=1 ; No = 0

# In[ ]:


telco_data['Churn'] = np.where(telco_data.Churn == 'Yes',1,0)


# In[ ]:


telco_data.head()


# **3.** Convert all the categorical variables into dummy variables

# In[ ]:


telco_data_dummies = pd.get_dummies(telco_data)
telco_data_dummies.head()


# **9. ** Relationship between Monthly Charges and Total Charges

# In[ ]:


sns.lmplot(data=telco_data_dummies, x='MonthlyCharges', y='TotalCharges', fit_reg=False)


# Total Charges increase as Monthly Charges increase - as expected.

# **10. ** Churn by Monthly Charges and Total Charges

# In[ ]:


Mth = sns.kdeplot(telco_data_dummies.MonthlyCharges[(telco_data_dummies["Churn"] == 0) ],
                color="Red", shade = True)
Mth = sns.kdeplot(telco_data_dummies.MonthlyCharges[(telco_data_dummies["Churn"] == 1) ],
                ax =Mth, color="Blue", shade= True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')


# **Insight:** Churn is high when Monthly Charges ar high

# In[ ]:


Tot = sns.kdeplot(telco_data_dummies.TotalCharges[(telco_data_dummies["Churn"] == 0) ],
                color="Red", shade = True)
Tot = sns.kdeplot(telco_data_dummies.TotalCharges[(telco_data_dummies["Churn"] == 1) ],
                ax =Tot, color="Blue", shade= True)
Tot.legend(["No Churn","Churn"],loc='upper right')
Tot.set_ylabel('Density')
Tot.set_xlabel('Total Charges')
Tot.set_title('Total charges by churn')


# **Surprising insight ** as higher Churn at lower Total Charges
# 
# However if we combine the insights of 3 parameters i.e. Tenure, Monthly Charges & Total Charges then the picture is bit clear :- Higher Monthly Charge at lower tenure results into lower Total Charge. Hence, all these 3 factors viz **Higher Monthly Charge**,  **Lower tenure** and **Lower Total Charge** are linkd to **High Churn**.

# **11. Build a corelation of all predictors with 'Churn' **

# In[ ]:


plt.figure(figsize=(20,8))
telco_data_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')


# **Derived Insight: **
# 
# **HIGH** Churn seen in case of  **Month to month contracts**, **No online security**, **No Tech support**, **First year of subscription** and **Fibre Optics Internet**
# 
# **LOW** Churn is seens in case of **Long term contracts**, **Subscriptions without internet service** and **The customers engaged for 5+ years**
# 
# Factors like **Gender**, **Availability of PhoneService** and **# of multiple lines** have alomost **NO** impact on Churn
# 
# This is also evident from the **Heatmap** below

# In[ ]:


plt.figure(figsize=(12,12))
sns.heatmap(telco_data_dummies.corr(), cmap="Paired")


# With this I conclude the Exploration of Telco Data. I will explore the predictive modelling soon. **Thanks!!**

# In[ ]:




