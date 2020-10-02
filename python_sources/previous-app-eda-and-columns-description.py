#!/usr/bin/env python
# coding: utf-8

# <h2>Previous application: exploratory analysis and variables meaning</h2>
# 
# A simple analysis for the previous applications with the description of every column. Please leave a comment if you find any misleading information or have more details. The 'missing values' percentage in each item is given by the number of NaNs divided by the total number of rows.
# 
# Let's start by looking at the <b>previous_application.csv</b> columns:

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pay_df = pd.read_csv('../input/installments_payments.csv')
previous = pd.read_csv('../input/previous_application.csv')
previous.head()


# In[ ]:


def categorical_feature_bar_chart(feature_name, pallete = 'Blues_d'):
    count = previous[feature_name].value_counts(dropna = False).to_frame().reset_index()
    count.rename({'index': feature_name.lower(), feature_name: 'COUNT'}, axis=1, inplace= True)
    plt.figure(figsize=(10,5))
    plt.title("Number of applications by {}".format(feature_name))
    ax = sns.barplot(x = feature_name.lower(), y = 'COUNT', data= count, palette= pallete)

print("Number of rows: {}".format(len(previous)))
print("Number of unique previous applications (SK_ID_PREV): {}".format(previous['SK_ID_PREV'].nunique()))
print("Number of unique current applications (SK_ID_CURR): {}".format(previous['SK_ID_CURR'].nunique()))


# <h3>1. NAME_CONTRACT_TYPE</h3>
# 
# We have 3 possibles contract types: Consumer loans, Cash loans and Revolving loans. I will be using the description provided by [former HC analyst Anh](https://www.kaggle.com/c/home-credit-default-risk/discussion/63032) (check his post for more details).
# 1. <b>Revolving loans (credit card) </b>: Loan applicant is given a credit limit, he/she can spend/withdraw in a month within that credit limit, and at the end of the month HC will inform him minimum payment he needs to make.
# 2. <b>Consumer loans (Point of Sale - POS)</b>: Loan applicant is given a credit limit to buy a goods (phone, laptop) and will need to repay that credit monthly, 30 days interval each month.
# 3. <b>Cash loans</b>: Loan applicant is given a lump sum of cash. He/she can spend for whatever purpose and will need to repay that credit monthly, 30 days interval each month.
# 
# Missing values: 0 (but we have 346 "XNA")

# In[ ]:


categorical_feature_bar_chart('NAME_CONTRACT_TYPE', pallete = 'Blues_d')


# <h3>2. NAME_CONTRACT_STATUS</h3>
# 
# If the application was approved or refused by Home Credit. The status can also be Canceled or Unused offer if the consumer havent used the credit.
# 
# Missing values: 0

# In[ ]:


categorical_feature_bar_chart('NAME_CONTRACT_STATUS', pallete = 'Greens_d')


# <h3>3. AMT_ANNUITY</h3>
# 
# Monthly payments (including interests) at application time. Clients can change their annuity which you would see in balance and payments table. We have some high annuity values (up to 418k), but most applications are below 20k.
# 
# Missing values: 22.2%

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_ANNUITY")
ax = sns.distplot(previous["AMT_ANNUITY"].dropna())


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_ANNUITY - LIMITED TO 60k")
ax = sns.distplot(previous[previous["AMT_ANNUITY"] < 60000]["AMT_ANNUITY"].dropna())


# <h3>4. AMT_APPLICATION</h3>
# 
# How much credit did the client ask for.
# 
# Missing values: 0

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_APPLICATION")
ax = sns.distplot(previous["AMT_APPLICATION"], color= 'g')


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_APPLICATION - LIMITED TO 1M")
ax = sns.distplot(previous[previous['AMT_APPLICATION'] <= 1000000]["AMT_APPLICATION"], color= 'g')


# <h3>5. AMT_CREDIT</h3>
# 
# How much credit the client received. This amount can be higher than AMT_APPLICATION, probably due to insurance purchase (see explanation for goods price). It can also be a smaller value depending on HC judgment (risk, credit limit...) 
# 
# Missing values: 0

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_CREDIT")
ax = sns.distplot(previous["AMT_CREDIT"].dropna(), color= 'r')


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_CREDIT - LIMITED TO 1M")
ax = sns.distplot(previous[previous['AMT_CREDIT'] <= 1000000]["AMT_CREDIT"], color= 'r')


# <h3>5. AMT_DOWN_PAYMENT</h3>
# 
# For Consumer loans, how much did the client payed out of pocket for the goods . For revolving loans and cash loans this value is mostly missing or zero. Following Ahn's explanation about Consumer loans:
# 
# > If the price of the goods (AMT_GOODS_PRICE) is 100 USD, he will need to pay out of pocket (AMT_DOWN_PAYMENT) 20 USD for example. 80 USD will be the loan from HC that paid directly to goods seller.
# 
# Missing values: 53%

# In[ ]:


revolving = previous[previous['NAME_CONTRACT_TYPE'] == 'Revolving loans']
cons = previous[previous['NAME_CONTRACT_TYPE'] == 'Consumer loans']
cash = previous[previous['NAME_CONTRACT_TYPE'] == 'Cash loans']
print("AMT_DOWN_PAYMENT missing values for Revolving loans: {:.2f}%".format(100*len(revolving[revolving['AMT_DOWN_PAYMENT'].isnull()])/len(revolving)))
print("AMT_DOWN_PAYMENT zero values for Revolving loans: {:.2f}%".format(100*len(revolving[revolving['AMT_DOWN_PAYMENT'] == 0])/len(revolving)))
print("AMT_DOWN_PAYMENT missing values for Cash loans: {:.2f}%".format(100*len(cash[cash['AMT_DOWN_PAYMENT'].isnull()])/len(cash)))
print("AMT_DOWN_PAYMENT zero values for Cash loans: {:.2f}%".format(100*len(cash[cash['AMT_DOWN_PAYMENT'] == 0])/len(cash)))


# <h3>6. AMT_GOODS_PRICE</h3>
# 
# Price of the good (cellphone, computer, microwave...) purchased. The credit amount can be higher due to insurance and therefore AMT_CREDIT = AMT_GOODS_PRICE + insurance. The boolean feature NFLAG_INSURED_ON_APPROVAL corresponds to one of the many options of insurance they have and we can have a higher value of AMT_CREDIT even with a false flag (Source: [competition host](https://www.kaggle.com/c/home-credit-default-risk/discussion/57054)).
# 
# Missing values: 23%

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_GOODS_PRICE")
ax = sns.distplot(previous["AMT_GOODS_PRICE"].dropna())


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_GOODS_PRICE - LIMITED TO 1M")
ax = sns.distplot(previous[previous['AMT_GOODS_PRICE'] <= 1000000]["AMT_GOODS_PRICE"])


# <h3>7. WEEKDAY_APPR_PROCESS_START and HOUR_APPR_PROCESS_START</h3>
# 
# 
# WEEKDAY_APPR_PROCESS_START: Which weekday (sunday, monday...) the client started the loan application process.
# 
# HOUR_APPR_PROCESS_START: Approximate hour that the loan application process started.
# 
# Missing values: zero for both features

# In[ ]:


categorical_feature_bar_chart('WEEKDAY_APPR_PROCESS_START', 'Blues_d')


# In[ ]:


categorical_feature_bar_chart('HOUR_APPR_PROCESS_START', 'Greens_d')


# <h3>8. DAYS_DECISION</h3>
# 
# How many days since the decision about previous application was made (time is always relative to current application). All values are negative as they represent events in the past.
# 
# Missing values: 0

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of DAYS_DECISION")
ax = sns.distplot(previous["DAYS_DECISION"].dropna(), color= 'navy')


# <h3>9. DAYS_TERMINATION</h3>
# 
# How many days since the application ended. Values are always negative, except for '365243' which means that this application still active (not completed yet). Refused/canceled/unused applications have missing values. A few approved applications have missing values, but I dont know the reason.
# 
# Missing values: 40%

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of DAYS_TERMINATION")
ax = sns.distplot(previous[previous["DAYS_TERMINATION"] < 365243]["DAYS_TERMINATION"].dropna(), color= 'orange')


# <h3>10. DAYS_FIRST_DRAWING</h3>
# 
# How many days since the first disbursement of the previous application. This variable seems to be exclusive for Revolving loans and Consumer/Cash loans usually have 365243 or missing values.
# 
# Missing values: 40%

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of DAYS_FIRST_DRAWING")
ax = sns.distplot(previous[previous["DAYS_FIRST_DRAWING"] < 365243]["DAYS_FIRST_DRAWING"].dropna(), color= 'green')


# <h3>11. DAYS_FIRST_DUE</h3>
# 
# Relative to the date of current application when was the first due supposed to be of the previous application. 365243 means that the due wasnt supposed to happen yet.
# 
# Missing values: 40%

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of DAYS_FIRST_DUE")
ax = sns.distplot(previous[previous["DAYS_FIRST_DUE"] < 365243]["DAYS_FIRST_DUE"].dropna(), color= 'red')


# <h3>12. DAYS_LAST_DUE_1ST_VERSION and DAYS_LAST_DUE</h3>
# 
# DAYS_LAST_DUE_1ST_VERSION
# 
# Relative to application date when was the <b>last due supposed to happen</b> (consideraing initial term). As this variable is a prediction it can have positive values (events in future), which usually happens when the client complete the loan before the initial term.
# 
# The 365243 value is almost exclusive for Revolving loans as they usually dont have a specific last due date and therefore if the credit line still active this value is probably 365243. For Consumer and Cash loans there are only a few 365243 values (dont know the reason).
# 
# DAYS_LAST_DUE
# 
# Relative to application date when the <b>last due actually happened</b>. If the last due was not payed yet the value will be 365243. 
# 
# For both columns most missing values are for refused/canceled or unused offers.
#     
# Missing values: 40% for both

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of DAYS_LAST_DUE_1ST_VERSION")
ax = sns.distplot(previous[previous["DAYS_LAST_DUE_1ST_VERSION"] < 365243]["DAYS_LAST_DUE_1ST_VERSION"].dropna())


# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of DAYS_LAST_DUE")
ax = sns.distplot(previous[previous["DAYS_LAST_DUE"] < 365243]["DAYS_LAST_DUE"].dropna(), color= 'gray')


# <h3>13. CNT_PAYMENT</h3>
# 
# Initial number of instalments to repay debt.

# In[ ]:


plt.figure(figsize=(12,5))
plt.title("Distribution of CNT_PAYMENT")
ax = sns.distplot(previous["CNT_PAYMENT"].dropna(), color= 'green')


# <h3> Work in progress....  next: categorical columns</h3>
