#!/usr/bin/env python
# coding: utf-8

# # Home Credit Default Risk - Kaggle Competition

# ## Table of Contents

# 1. [Loading Libraries](#lib)
# 2. [Loading Dataset Files](#files)
# 3. [Exploratory Data Analysis - EDA](#eda)  
#     3.1 [Evaluating Train Dataset](#train)   
#     3.2 [Evaluating Bureau Dataset](#bureau)  
#     3.3 [Evaluating Previous_Application Dataset](#previous_application)  
#     3.4 [Evaluating Instalment_Payment Dataset](#instalment)   
#     3.5 [Evaluating Credit_Card_Balance Dataset](#credit_card)   
#     3.6 [Evaluating Pos_Cash_Balance Dataset](#cash_balance)   
# 4. [Feature Engineering](#eng)
# 5. [Missing Values](#missing)
# 6. [Dimensionality Reduction - Primary Component Analysis - PCA](#pca)
# 7. [Conclusion and Next Steps](#conclusion)

# ## 1. Loading Libraries
# <a id="lib"></a>

# In[ ]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 2. Loading Dataset Files
# <a id="files"></a>

# In[ ]:


# Loading all the datasets provided on the competition
train = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')
bureau = pd.read_csv('../input/bureau.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
installments_payments = pd.read_csv('../input/installments_payments.csv')
pos_cash_balance = pd.read_csv('../input/POS_CASH_balance.csv')
previous_application = pd.read_csv('../input/previous_application.csv')


# ## 3. Exploratory Data Analysis - EDA
# <a id="eda"></a>

# ### 3.1 Evaluating TRAIN dataset
# <a id="train"></a>

# In[ ]:


train.head()


# In[ ]:


train.info()


# ### How many clients with payment difficulties?

# In[ ]:


# Ploting how many clients with payment difficulties on the train dataset
graph = sns.countplot(x="TARGET",data=train, palette='muted')


# In[ ]:


# Calculating how many clients with payment difficulties on the train dataset
pay_diff = train[train['TARGET'] == 1]['TARGET'].count()
pay_ok = train[train['TARGET'] == 0]['TARGET'].count()
print('Clients with payment difficulty: %s' % pay_diff)
print('Clients with payment ok: %s' % pay_ok)


# 8% of the clients are not trustable for a loan

# ### Correlation between numerical values in the TRAIN dataset

# In[ ]:


# Selecting only the numerical features in the Train dataset
numerical_features = train[['TARGET', 'SK_ID_CURR', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
                            'AMT_GOODS_PRICE','REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
                            'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 
                            'FLAG_WORK_PHONE','FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL','CNT_FAM_MEMBERS', 
                            'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START', 
                            'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
                            'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY', 
                            'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                            'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
                            'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
                            'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG',
                            'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
                            'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE',
                            'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE',
                            'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE',
                            'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
                            'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI',
                            'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI',
                            'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI',
                            'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
                            'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'TOTALAREA_MODE',
                            'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
                            'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
                            'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
                            'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
                            'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
                            'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
                            'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
                            'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
                            'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',
                            'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
                            'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
                            'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']]


# In[ ]:


# Calculating the corelation between numerical features and the TARGET on the training dataset
corelation = numerical_features.corr()['TARGET']


# In[ ]:


# Sorting the correlation to check wich feature is better correlated with the TARGET 
corelation.abs().sort_values(ascending = False)[0:10]


# Seems that only the 3 Ext_Source features have a high corelation with the target features and we don't know nothing about them. Better explore the other datasets.

# ### 3.2 Evaluating BUREAU dataset
# <a id="bureau"></a>

# In[ ]:


bureau.head()


# In[ ]:


bureau.info()


# ### How many previous credit loans the clients have in other institutions?

# In[ ]:


# Ploting the histogram of the previous client accounts in other institutions
bureau_account = bureau['SK_ID_BUREAU'].groupby(bureau['SK_ID_CURR'])
bureau_account_n = bureau_account.count()
graph = sns.distplot(bureau_account_n)
graph.set_xlabel("Number of Previous Credit Loans per Client")


# In[ ]:


# Describing the stats of the graph
bureau_account_n.describe()


# Most of the clients have 4 previous credit loans in other institutions, but some of them have more than 100.

# ### How many active previous credit loans the clients have in other institutions?

# In[ ]:


# Selecting only clients with active credit loans accounts on the Bureau dataset
active_account = bureau[['SK_ID_CURR', 'CREDIT_ACTIVE']]
bureau_active_account = active_account[active_account['CREDIT_ACTIVE'] == 'Active']
bureau_active_account_n = bureau_active_account['CREDIT_ACTIVE'].groupby(bureau_active_account['SK_ID_CURR']).count()


# In[ ]:


# Ploting the histogram of the clients with active credit loans accounts
graph = sns.distplot(bureau_active_account_n)
graph.set_xlabel("Number of Previous Active Credit Loans per Client")


# In[ ]:


# Describing the stats of the graph
bureau_active_account_n.describe()


# Most of the clients have 2 previous active credit loans in other institutions, but some of them have more than 30.

# ### What is the debt amount that clients have on previous credit loans in onther institutions?

# In[ ]:


# Checking for null values on the debt amount feature
print('There are %i accounts with null debt amount' %bureau['AMT_CREDIT_SUM_DEBT'].isnull().sum())


# In[ ]:


# Elimitating null values to better visualize the data and grouping by client SK_ID_CURR
bureau_client_debt = bureau[['AMT_CREDIT_SUM_DEBT', 'SK_ID_CURR']]
bureau_client_debt = bureau_client_debt[bureau_client_debt['AMT_CREDIT_SUM_DEBT'].notnull()]
bureau_client_debt_amount = bureau_client_debt.groupby(bureau_client_debt['SK_ID_CURR']).sum()


# In[ ]:


# Defining bins to better visualize the current debt per client
# Low debt is less than $100
# Medium debt is between $100 and $1.000
# Medium high debt is between $1.000 and $10.000
# High debt is bigger than $10.000
def amount_bins_debt(amount):
    if amount < 10000:
        amount = 'Low'
    else: 
        if amount > 10000 and amount < 100000:
            amount = 'Medium'
        else: 
            if amount > 100000 and amount < 1000000:
                amount = 'Medium High'
            else:
                amount = 'High'
    return amount


# In[ ]:


# Transforming the total debt amount in bin sizes
bureau_client_debt_bins = bureau_client_debt_amount['AMT_CREDIT_SUM_DEBT'].apply(amount_bins_debt)


# In[ ]:


# Ploting the current debt amount per client on all previous accounts
graph = sns.countplot(bureau_client_debt_bins, palette='muted')
graph.set_xlabel("Current Debt Amount")
graph.set_ylabel("Number of Clients")

# Everything below is to put a legend in the graph
class TextHandler(HandlerBase):
    def create_artists(self, legend, tup ,xdescent, ydescent,
                        width, height, fontsize,trans):
        tx = Text(width/2.,height/2,tup[0], fontsize=fontsize,
                  ha="center", va="center", color=tup[1], fontweight="bold")
        return [tx]
    
handltext = ["Low", "Medium", "Medium High", "High"]
labels = ["$ < 10k", "10k < $ < 100k", "100k< $ < 1M", "$ > $1M"]


t = graph.get_xticklabels()
labeldic = dict(zip(handltext, labels))
labels = [labeldic[h.get_text()]  for h in t]
handles = [(h.get_text(),c.get_fc()) for h,c in zip(t,graph.patches)]

graph.legend(handles, labels, handler_map={tuple : TextHandler()})


# 40% of the clients have a sum of debts on previous accounts on other institutions between \$100.000 and \$1.000.000 and 16% have more than \$1.000.000

# ### How many clients are overdue with previous credit loans in onther institutions?

# In[ ]:


# Selecting due and overdue clients
bureau_due = bureau[bureau['CREDIT_DAY_OVERDUE'] == 0]
bureau_due = bureau_due[['SK_ID_CURR', 'CREDIT_DAY_OVERDUE','AMT_CREDIT_SUM_OVERDUE']]
bureau_due_n = len(bureau_due['SK_ID_CURR'].unique())
bureau_overdue = bureau[bureau['CREDIT_DAY_OVERDUE'] != 0]
bureau_overdue = bureau_overdue[['SK_ID_CURR', 'CREDIT_DAY_OVERDUE', 'AMT_CREDIT_SUM_OVERDUE']]
bureau_overdue_n = len(bureau_overdue['SK_ID_CURR'].unique())

# Ploting the result
graph = sns.barplot(x=['Due', 'Overdue'],y=[bureau_due_n, bureau_overdue_n], palette= 'muted')
graph.set_ylabel("Number of Clients")
print('Due Clients: %s\nOverdue Clients: %s' %(bureau_due_n, bureau_overdue_n))


# Less than 2% of the clients are overdue with previous credit loans at other institution.
# There are 2.003 different clients in the Bureau dataset that are not in the Train dataset.

# ### What is the maximum days past due of clients with previous credit loans in onther institutions?

# In[ ]:


# Ploting the maximum days overdue of the clients
bureau_max_client_overdue = bureau_overdue['CREDIT_DAY_OVERDUE'].groupby(bureau_overdue['SK_ID_CURR']).max()
graph = sns.distplot(bureau_max_client_overdue)
graph.set_xlabel("Maximum Days Overdue per Client")


# In[ ]:


# Describing the stats of the graph
bureau_max_client_overdue.describe()


# Most of the 3.864 clients are 35 days overdue but the mean value is 351 days overdue. We can explain that because of a small number of clients that are a few thousand days overdue and move the average to a bigger value.

# ### What is the current amount past due of clients with previous credit loans in other institutions?

# In[ ]:


# Defining bins to better visualize the current amount past due per client
# Low overdue is less than $100
# Medium overdue is between $100 and $1.000
# Medium high overdue is between $1.000 and $10.000
# High overdue is bigger than $10.000
def amount_bins_overdue(overdue):
    if overdue < 100:
        overdue = 'Low'
    else: 
        if overdue > 100 and overdue < 1000:
            overdue = 'Medium'
        else: 
            if overdue > 1000 and overdue < 10000:
                overdue = 'Medium High'
            else:
                overdue = 'High'
    return overdue


# In[ ]:


# Calculating the total current amount overdue per client
bureau_amount_overdue = bureau_overdue['AMT_CREDIT_SUM_OVERDUE'].groupby(bureau_overdue['SK_ID_CURR'])
bureau_sum_amount_overdue = bureau_amount_overdue.sum()

# Transforming the total amount in bin sizes
bureau_sum_amount_overdue_bins = bureau_sum_amount_overdue.apply(amount_bins_overdue)


# In[ ]:


# Ploting the current amount overdue per client
graph = sns.countplot(bureau_sum_amount_overdue_bins, palette='muted')
graph.set_xlabel("Current Amount Overdue")
graph.set_ylabel("Number of Clients")

# Everything below is to put a legend in the graph
class TextHandler(HandlerBase):
    def create_artists(self, legend, tup ,xdescent, ydescent,
                        width, height, fontsize,trans):
        tx = Text(width/2.,height/2,tup[0], fontsize=fontsize,
                  ha="center", va="center", color=tup[1], fontweight="bold")
        return [tx]
    
handltext = ["Low", "Medium", "Medium High", "High"]
labels = ["$ < 100", "100 < $ < $1k", "1k< $ < 10k", "$ > $10k"]


t = graph.get_xticklabels()
labeldic = dict(zip(handltext, labels))
labels = [labeldic[h.get_text()]  for h in t]
handles = [(h.get_text(),c.get_fc()) for h,c in zip(t,graph.patches)]

graph.legend(handles, labels, handler_map={tuple : TextHandler()})


# Most of the clients are less than \$1.000 overdue, but at least 500 clients are more than \$10.000 overdue.

# ### How many times clients prolonged credit in previous credit loans in onther institutions?

# In[ ]:


# Ploting the number of times credit was prolonged per client in all previous loan credit accounts
credit_prolong=bureau[bureau['CNT_CREDIT_PROLONG'] > 0]
bureau_credit_prolong_client = credit_prolong['CNT_CREDIT_PROLONG'].groupby(credit_prolong['SK_ID_CURR'])
bureau_credit_prolong_client_n = bureau_credit_prolong_client.sum()
graph = sns.distplot(bureau_credit_prolong_client_n)
graph.set_xlabel("Number of Credit Prolongued per Client")


# In[ ]:


# Describing the stats of the graph
bureau_credit_prolong_client_n.describe()


# Only 3% of the clients have prolonged credit on previous accounts. Most of them have prolonged only once but there are clients that prolonged credit 9 times.

# ### 3.3 Evaluating PREVIOUS_APPLICATION dataset
# <a id="previous_application"></a>

# In[ ]:


previous_application.head()


# In[ ]:


previous_application.info()


# ### How many previous credit loans the clients have in Home Credit?

# In[ ]:


# Ploting the histogram of the number of previous client accounts in Home Credit
prev_app_acc_n = previous_application['SK_ID_PREV'].groupby(previous_application['SK_ID_CURR']).count()
graph = sns.distplot(prev_app_acc_n)
graph.set_xlabel("Number of Previous Credit Loans per Client")


# In[ ]:


# Describing the stats of the graph
prev_app_acc_n.describe()


# Most of the clients have 4 previous credit loans in Home Credit, but some of them have more than 100.

# ### Which are the top 5 purpose of previous credit on Home Credit?

# In[ ]:


# Grouping, sorting and selecting the top five credit purpose
most_common_purpose = previous_application['SK_ID_PREV'].groupby(previous_application['NAME_CASH_LOAN_PURPOSE'])
most_common_purpose = most_common_purpose.count().sort_values(ascending = False)[0:5]

# Ploting the top five purpose of previous credit
graph = sns.barplot(y=most_common_purpose.index,x=most_common_purpose.values, palette= 'muted', orient = 'h')
graph.set_ylabel("Number of Application")


# 96% of the previous application was for the XAP and XNA purpose. I don`t actually know what that mean.

# ### What is the status of previous applications on Home Credit?

# In[ ]:


application_status = previous_application['SK_ID_PREV'].groupby(previous_application['NAME_CONTRACT_STATUS']).count()

# Ploting the top five purpose of previous credit
graph = sns.barplot(y=application_status.index,x=application_status.values, palette= 'muted', orient = 'h')
graph.set_ylabel("Number of Application")


# 62% of previous applications on Home Credit are approved, 18% are cancelled and 17% are refused.

# ### How many previous refused applications clients have on Home Credit?

# In[ ]:


# Filtering the previous application with only refused previous applications
refused_application = previous_application[previous_application['NAME_CONTRACT_STATUS'] == 'Refused']


# In[ ]:


# Printing the number and percentage of clients with previous refused applications
client_refused_n = len(refused_application['SK_ID_CURR'].unique())
client_total_n = len(previous_application['SK_ID_CURR'].unique())
print('Total Clients With Previous Applications: %s\nTotal Clients With Refused Previous Application : %s'
      %(client_total_n, client_refused_n))
print('Percentage of Clients With Refuse Previous Applications: %2.2f %%'%((client_refused_n/client_total_n)*100))

# Ploting the histogram of the number of refused previous client accounts in Home Credit
prev_app_refused_n = refused_application['SK_ID_PREV'].groupby(refused_application['SK_ID_CURR']).count()
graph = sns.distplot(prev_app_refused_n)
graph.set_xlabel("Number of Previous Refused Credit Loans per Client")


# In[ ]:


# Describing the stats of the graph
prev_app_refused_n.describe()


# Most of the clients with refused application have 2 refused credit applications on Home Credit but some of them have more than 60.

# ### 3.4 Evaluating Instalment Payment dataset
# <a id="instalment"></a>

# In[ ]:


installments_payments.head()


# In[ ]:


installments_payments.info()


# ### How many clients delayed with installment payment on previous Home Credit applications?

# In[ ]:


# Filtering the installment days
day_payment = installments_payments[['SK_ID_PREV', 'SK_ID_CURR', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT']]

# Filling missing payment with 0 value
day_payment = day_payment.fillna(0)

# Creating a new column with the diference between payment days and installment days
day_payment['DELAYED'] = day_payment['DAYS_ENTRY_PAYMENT'] - day_payment['DAYS_INSTALMENT']
day_payment.head()


# In[ ]:


# Counting clients that delayed or not the payment
inst_paym_delayed_clients = day_payment['DELAYED'].groupby(day_payment['SK_ID_CURR']).sum()
inst_paym_delayed_clients_n = (inst_paym_delayed_clients.values > 0).sum()
inst_paym_not_delayed_clients_n = (inst_paym_delayed_clients.values <= 0).sum()

# Ploting the result
graph = sns.barplot(x=['Not_Delayed_Clients', 'Delayed_Clients'],y=[inst_paym_not_delayed_clients_n, 
                                                                    inst_paym_delayed_clients_n], palette= 'muted')
graph.set_ylabel("Number of Clients")
print('Not_Delayed_Clients: %s\nDelayed_Clients: %s' %(inst_paym_not_delayed_clients_n, inst_paym_delayed_clients_n))


# Only 3% of the clients have delayed at least one payment of instalments on previous application in Home Credit.

# ### How many clients are in debt with instalments in previous Home Credit applications?

# In[ ]:


# Filtering the installment payment
amt_payment = installments_payments[['SK_ID_PREV', 'SK_ID_CURR', 'NUM_INSTALMENT_NUMBER', 
                                        'AMT_INSTALMENT', 'AMT_PAYMENT']]

# Filling missing payment with 0 value
amt_payment = amt_payment.fillna(0)

# Calculating the debt in each installment
debt_instalment = amt_payment['AMT_PAYMENT'] - amt_payment['AMT_INSTALMENT']

# Creating a new debt colunm on amount_payment
amt_payment['AMT_DEBT'] = debt_instalment
amt_payment.head()


# In[ ]:


# Calculating the total debt per client
inst_paym_debt_client = amt_payment['AMT_DEBT'].groupby(amt_payment['SK_ID_CURR']).sum()
with_debt_client_n = (inst_paym_debt_client.values < 0).sum()
without_debt_client_n = (inst_paym_debt_client.values >= 0).sum()

# Ploting the result
graph = sns.barplot(x=['Without_Debt', 'With_Debt'],y=[without_debt_client_n, with_debt_client_n], palette= 'muted')
graph.set_ylabel("Number of Clients")
print('With Installment Debt Clients: %s\nWithout Installment Debt Clients: %s' %(with_debt_client_n, 
                                                                                   without_debt_client_n))


#    37% of the clients are in debt with intalments in previous application in Home Credit.

# ### 3.5 Evaluating Credit Card Balance dataset
# <a id="credit_card"></a>

# In[ ]:


credit_card_balance.head()


# In[ ]:


credit_card_balance.info()


# ### How much credit card debt clients have in previous Home Credit applications?

# In[ ]:


# selecting the credit card payment columns
amt_credit_payment = credit_card_balance[['SK_ID_PREV', 'SK_ID_CURR', 'MONTHS_BALANCE', 
                                          'AMT_BALANCE','AMT_RECEIVABLE_PRINCIPAL']] 
                                          
# Filling missing payment with 0 value
amt_credit_payment = amt_credit_payment.fillna(0)

# Calculating the last month balance for each previous application
last_balance = pd.DataFrame(columns = ['SK_ID_PREV', 'MONTHS_BALANCE'])
last_balance['SK_ID_PREV'] = amt_credit_payment['MONTHS_BALANCE'].groupby(amt_credit_payment['SK_ID_PREV']).max().index
last_balance['MONTHS_BALANCE'] = amt_credit_payment['MONTHS_BALANCE'].groupby(amt_credit_payment['SK_ID_PREV']).max().values
last_balance = last_balance[['SK_ID_PREV', 'MONTHS_BALANCE']].apply(tuple, 1)

# Filtering only the last month balance
amt_credit_payment = amt_credit_payment[amt_credit_payment[['SK_ID_PREV', 'MONTHS_BALANCE']].apply(tuple, 1).isin(last_balance)]

# Creating a new debt colunm on amount_credit_payment
amt_credit_payment['AMT_DEBT'] = amt_credit_payment['AMT_RECEIVABLE_PRINCIPAL'] - amt_credit_payment['AMT_BALANCE']
amt_credit_payment.head()


# In[ ]:


# Calculating the total debt per client
credit_card_debt_client = amt_credit_payment['AMT_DEBT'].groupby(amt_credit_payment['SK_ID_CURR']).sum()
with_credit_debt_client_n = (credit_card_debt_client.values < 0).sum()
without_credit_debt_client_n = (credit_card_debt_client.values >= 0).sum()

# Ploting the result
graph = sns.barplot(x=['Without_Debt', 'With_Debt'],y=[without_credit_debt_client_n, with_credit_debt_client_n], 
                    palette= 'muted')
graph.set_ylabel("Number of Clients")
print('With Credit Card Debt Clients: %s\nWithout Credit Card Debt Clients: %s' %(with_credit_debt_client_n, 
                                                                                   without_credit_debt_client_n))


#    34% of the clients are in debt with the credit card in previous application in Home Credit.

# In[ ]:


# Selecting only clients with credit card debit
debt = credit_card_debt_client.values
debt = debt[debt<0]
debt = debt * -1

# Ploting the amount of credit card debt 
graph = sns.distplot(debt)


# In[ ]:


# Descibing the stats of the graph
debt_df = pd.DataFrame(columns = ['Debt'])
debt_df['Debt'] = debt
debt_df.describe()


# Most of the clients have a 5936,74 debt with credit card in previous applications in Home Credit, but some of them have more than 200.000,00 in debt.

# ### How many clients were overdue with credit card payment on previous Home Credit applications?

# In[ ]:


# Counting clients that delayed or not the payment
dpd_credit_card = credit_card_balance['SK_DPD'].groupby(credit_card_balance['SK_ID_CURR']).sum()
credit_overdue_clients_n = (dpd_credit_card.values > 0).sum()
credit_due_clients_n = (dpd_credit_card.values == 0).sum()


# In[ ]:


# Ploting the result
graph = sns.barplot(x=['Due Credit Card Client', 'Overdue Credit Card Clients'], 
                    y=[credit_due_clients_n, credit_overdue_clients_n], palette= 'muted')
graph.set_ylabel("Number of Clients")
print('Due Credit Card Clients: %s\nOverdue Credit Card Clients: %s' %(credit_due_clients_n, credit_overdue_clients_n))


#    22% of the clients were overdue with credit card payment in previous application in Home Credit.

# ### 3.6 Evaluating Pos Cash Balance dataset
# <a id="cash_balance"></a>

# In[ ]:


pos_cash_balance.head()


# In[ ]:


pos_cash_balance.info()


# ### How many clients were overdue with previous cash loans applications in Home Credit?

# In[ ]:


# Counting clients that are due and overdue with the cash loan
dpd_cash_loan = pos_cash_balance['SK_DPD'].groupby(pos_cash_balance['SK_ID_CURR']).sum()
cash_loan_overdue_clients_n = (dpd_cash_loan.values > 0).sum()
cash_loan_due_clients_n = (dpd_cash_loan.values == 0).sum()


# In[ ]:


# Ploting the result
graph = sns.barplot(x=['Due Cash Loan Client', 'Overdue Cash Loan Clients'], 
                    y=[cash_loan_due_clients_n, cash_loan_overdue_clients_n], palette= 'muted')
graph.set_ylabel("Number of Clients")
print('Due Cash Loan Clients: %s\nOverdue Cash Loan Clients: %s' %(cash_loan_due_clients_n, cash_loan_overdue_clients_n))


#    19% of the clients were overdue with cash loan payment in previous application in Home Credit.

# ## 4. Features Engineering
# <a id="eng"></a>

# In[ ]:


# Creating a new train dataset with the new features from other datasets
new_train = numerical_features


# In[ ]:


# Joining with the new features from bureau dataset
# Number of previous loans in other institutions
new_train = new_train.join(bureau_account_n, on='SK_ID_CURR')
new_train.rename(columns={'SK_ID_BUREAU':'CNT_ACOUNTS_BUREAU'}, inplace=True)

# Number of previous active loans in other institutions
new_train = new_train.join(bureau_active_account_n, on='SK_ID_CURR')
new_train.rename(columns={'CREDIT_ACTIVE':'CNT_ACTIVE_ACOUNTS_BUREAU'}, inplace=True)

# Debt amount of previous loans in other institutions
new_train = new_train.join(bureau_client_debt_amount, on='SK_ID_CURR')
new_train.rename(columns={'AMT_CREDIT_SUM_DEBT':'AMT_CREDIT_DEBT_BUREAU'}, inplace=True)

# Days overdue of previous loans in other institutions
bureau_days_overdue_sum = bureau_overdue['CREDIT_DAY_OVERDUE'].groupby(bureau_overdue['SK_ID_CURR']).sum()
new_train = new_train.join(bureau_days_overdue_sum, on='SK_ID_CURR')
new_train.rename(columns={'CREDIT_DAY_OVERDUE':'CNT_DAYS_OVERDUE_BUREAU'}, inplace=True)

# Amount overdue of previous loans in other institutions
new_train = new_train.join(bureau_sum_amount_overdue, on='SK_ID_CURR')
new_train.rename(columns={'AMT_CREDIT_SUM_OVERDUE':'AMT_CREDIT_OVERDUE_BUREAU'}, inplace=True)

# Number of times credit was prolonged of previous loans in other institutions
new_train = new_train.join(bureau_credit_prolong_client_n, on='SK_ID_CURR')
new_train.rename(columns={'CREDIT_CREDIT_PROLONG':'CNT_DAYS_PROLONG_BUREAU'}, inplace=True)


# In[ ]:


# Joining with the new features from previous application dataset
# Number of previous loans in Home Credit
new_train = new_train.join(prev_app_acc_n, on='SK_ID_CURR')
new_train.rename(columns={'SK_ID_PREV':'CNT_ACOUNTS_HC'}, inplace=True)

# Number of refused previous loans in Home Credit
new_train = new_train.join(prev_app_refused_n, on='SK_ID_CURR')
new_train.rename(columns={'SK_ID_PREV':'CNT_REFUSED_HC'}, inplace=True)


# In[ ]:


# Joining with the new features from instalment payment dataset
# Number of delayed days of instalment payment in Home Credit
inst_paym_delayed_clients.values[inst_paym_delayed_clients.values < 0] = 0
new_train = new_train.join(inst_paym_delayed_clients, on='SK_ID_CURR')
new_train.rename(columns={'DELAYED':'CNT_DAYS_DELAYED_INSTALLMENT_HC'}, inplace=True)

# Amount of delayed instalment payment in Home Credit
inst_paym_debt_client.values[inst_paym_debt_client.values > 0] = 0
new_train = new_train.join(inst_paym_debt_client, on='SK_ID_CURR')
new_train.rename(columns={'AMT_DEBT':'AMT_DEBT_INSTALLMENT_HC'}, inplace=True)


# In[ ]:


# Joining with the new features from credit card balance dataset
# Amount of debt of credit card in Home Credit
credit_card_debt_client.values[credit_card_debt_client.values > 0] = 0
new_train = new_train.join(credit_card_debt_client, on='SK_ID_CURR')
new_train.rename(columns={'AMT_DEBT':'AMT_DEBT_CREDIT_CARD_HC'}, inplace=True)

# Number of days past due of credit card in Home Credit
new_train = new_train.join(dpd_credit_card, on='SK_ID_CURR')
new_train.rename(columns={'SK_DPD':'CNT_DPD_CREDIT_CARD_HC'}, inplace=True)


# In[ ]:


# Joining with the new features from pos cash balance dataset
# Number of days past due of pos cash balance in Home Credit
new_train = new_train.join(dpd_cash_loan, on='SK_ID_CURR')
new_train.rename(columns={'SK_DPD':'CNT_DPD_CASH_LOAN_HC'}, inplace=True)


# ## 5. Missing Values
# <a id="missing"></a>

# In[ ]:


# Dealing with missing values
# Filling the new engineered missing values with zero
new_train[new_train.columns[106 :]] = new_train[new_train.columns[106 :]].fillna(0)


# In[ ]:


# Filling the numerical missing values with mean value
new_train = new_train.fillna(new_train.mean())


# ## 6. Dimensonality Reduction - Primary Component Analysis - PCA
# <a id="pca"></a>

# In[ ]:


# Spliting the new_train into features X and target Y components
y_train = new_train['TARGET']
X_train = new_train.drop(['TARGET'], axis=1)


# In[ ]:


# Standardizing all the features to a normal distribution 
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)


# In[ ]:


# Using PCA from silkit learn
pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)


# In[ ]:


# Ploting the individual and cumulative explained variance of the features
plt.bar(range(1, (len(pca.explained_variance_ratio_) +1)), pca.explained_variance_ratio_, alpha=0.5, align='center', 
        label='individual explained variance')
plt.step(range(1, (len(pca.explained_variance_ratio_) +1)), np.cumsum(pca.explained_variance_ratio_), where='mid', 
        label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# The individual values of the explained variance are realy small, so we will need at least 40 features to explain 70% of the cumulative variance. 

# In[ ]:


# Ploting the result of the cumulative variance with 40 features
pca = PCA(n_components= 40)
X_train_pca = pca.fit_transform(X_train_std)

# Ploting the individual and cumulative explained variance of the features
plt.bar(range(1, 41), pca.explained_variance_ratio_, alpha=0.5, align='center', 
        label='individual explained variance')
plt.step(range(1, 41), np.cumsum(pca.explained_variance_ratio_), where='mid', 
        label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# ## 7. Conclusion and Next Steps
# <a id="conclusion"></a>

# My objective with this notebook was to show how to explore and analyze data to get ready for modeling. There are many other tools and procedures that can be used to get even more insights from the data, but a choose a few that are very common in data science projects.
# 
# The next step will be modeling and evaluating to check if more analysis should be done to get a better accuracy of the training predictions.
# 
# I would appreciate a collaborative approach of the community with comments and suggestion to this study. 
# 

# In[ ]:




