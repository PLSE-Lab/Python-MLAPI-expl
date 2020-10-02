#!/usr/bin/env python
# coding: utf-8

# # Home Credit Default Risk

# Can you predict how capable each applicant is of repaying a loan?

# 1. [Introduction](#introduction)
# 
# 2. [Loading the data](#Loading the data)
# 
# 3. [Checking for missing data](#Checking for missing data)
# 
# 4. [Explanatory Data Analysis](#EDA)

# <a id='introduction'></a>

# # 1. Introduction

# Founded in 1997, Home Credit Group is an international consumer finance provider with operations in 11 countries. They focus on providing responsible lending primarily to people with little or no credit history. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data, including telco and transactional information to predict their clients' repayment abilities. 
# 
# Home Credit is challenging Kagglers to help them unlock the full potential of their data through developing a good machine learning model that accurately predicts loan applicants' ability of repayment. Doing so will ensure that applicants capable of repayment are well served and that loans are given with a principal, maturity, and repayment calendar that will empower loan applicants to be successful.

# <a id='Loading the data'></a>

# # 2. Loading the data

# The datasets we have are:
# - application_{train|test}.csv: loan application data
# - bureau.csv: loan applicant's previous credits provided by other financial institutions that were reported to Credit Bureau
# - bureau_balance.csv: monthly balances of previous credits in Credit Bureau
# - POS_CASH_balance.csv: monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant has with Home Credit
# - credit_card_balance.csv: monthly balance snapshots of previous credit cards that the applicant has with Home Credit
# - previous_application.csv: previous applications for Home Credit loans
# - installments_payments.csv: repayment history for the previously disbursed credits in Home Credit

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from matplotlib_venn import venn3


# ## 2.1 Datasets sizes and snapshots:

# In[ ]:


df_train = pd.read_csv('../input/application_train.csv')
df_train.shape


# In[ ]:


df_train.head()


# In[ ]:


df_test = pd.read_csv('../input/application_test.csv')
df_test.shape


# In[ ]:


df_bu_balance = pd.read_csv('../input/bureau_balance.csv')
df_bu_balance.shape


# In[ ]:


df_bu_balance.head()


# In[ ]:


df_bureau = pd.read_csv('../input/bureau.csv')
df_bureau.shape


# In[ ]:


df_bureau.head()


# In[ ]:


df_cre_balance = pd.read_csv('../input/credit_card_balance.csv')
df_cre_balance.shape


# In[ ]:


df_cre_balance.head()


# In[ ]:


df_installment = pd.read_csv('../input/installments_payments.csv')
df_installment.shape


# In[ ]:


df_installment.head()


# In[ ]:


df_pos_cash = pd.read_csv('../input/POS_CASH_balance.csv')
df_pos_cash.shape


# In[ ]:


df_pos_cash.head()


# In[ ]:


df_prev_app = pd.read_csv('../input/previous_application.csv')
df_prev_app.shape


# In[ ]:


df_prev_app.head()


# <a id='Checking for missing data'></a>

# # 3. Checking for missing data

# In[ ]:


def missing_values(df):
    missing_value = df.isnull().sum()
    missing_percent = (df.isnull().sum()/len(df)*100).round(2)
    table = pd.concat([missing_value, missing_percent], axis=1).rename(
            columns={0:'Missing Values (Count)', 1: 'Missing Values (%)'})
    table = table[table['Missing Values (Count)']>0].sort_values(by='Missing Values (Count)', ascending=False)
    return table


# In[ ]:


missing_values(df_train).head(30)


# In[ ]:


missing_values(df_test).head(30)


# In[ ]:


missing_values(df_bu_balance)


# In[ ]:


missing_values(df_bureau)


# In[ ]:


missing_values(df_cre_balance)


# In[ ]:


missing_values(df_installment)


# In[ ]:


missing_values(df_pos_cash)


# In[ ]:


missing_values(df_prev_app)


# <a id='EDA'></a>

# # 4. Explanatory Data Analysis

# ## 4.1 Application_train dataset

# ### 4.1.1 Distribution of Default

# In[ ]:


df_viz = df_train["TARGET"].value_counts().rename(index={0: 'Not default', 1: 'Default'})
trace = go.Pie(labels=df_viz.index, values=df_viz.values)
data = [trace]
layout = go.Layout(title='Distribution of Default', titlefont=dict(size=22))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# Only 8% of loans are default and so we have a highly imbalance data.

# ### 4.1.2 Distribution of Loan Type

# In[ ]:


df_viz = df_train['NAME_CONTRACT_TYPE'].value_counts()
trace = go.Pie(labels=df_viz.index, values=df_viz.values)
data = [trace]
layout = go.Layout(title='Distribution of Loan Type', titlefont=dict(size=22))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# 90.5% of the loans are cash loans.

# ### 4.1.3 Real Estate/ Car Ownership

# In[ ]:


both = round(len(df_train[(df_train['FLAG_OWN_REALTY']=='Y') & 
                          (df_train['FLAG_OWN_CAR']=='Y')]) / len(df_train) * 100, 2)
realty_only = round(len(df_train[(df_train['FLAG_OWN_REALTY']=='Y') & 
                                 (df_train['FLAG_OWN_CAR']=='N')]) / len(df_train) * 100, 2)
car_only = round(len(df_train[(df_train['FLAG_OWN_REALTY']=='N') & 
                              (df_train['FLAG_OWN_CAR']=='Y')]) / len(df_train) * 100, 2)


# In[ ]:


plt.figure(figsize=(10, 10))
venn = venn2(subsets = (realty_only, car_only, both), set_labels = ('Real Estate', 'Car'))
for t in venn.set_labels: t.set_fontsize(16)
for t in venn.subset_labels: t.set_fontsize(16)
plt.title('Real Estate/ Car Ownership (%)', fontsize=22);


# 45.84% applicants own real estate only while 10.48% own car only. 23.53% own both. 

# ### 4.1.4 Distribution of Applicant's Total Income

# In[ ]:


plt.figure(figsize=(12, 5))
sns.distplot(df_train['AMT_INCOME_TOTAL'])
plt.title('Distribution of Applicant\'s Total Income', fontsize=22)
plt.xlabel('Income');


# ### 4.1.5 Distribution of Applicant's Credit Amount

# In[ ]:


plt.figure(figsize=(12, 5))
sns.distplot(df_train['AMT_CREDIT'])
plt.title('Distribution of Applicant\'s Credit Amount', fontsize=22)
plt.xlabel('Credit Amount');


# ### 4.1.6 Distribution of the Price of Goods for Consumer Loans

# In[ ]:


plt.figure(figsize=(12, 5))
sns.distplot(df_train['AMT_GOODS_PRICE'].dropna())
plt.title('Distribution of the Price of Goods for Consumer Loans', fontsize=22)
plt.xlabel('Price of Goods');


# ### 4.1.7 Applicant's Marital Status

# In[ ]:


temp = df_train['NAME_FAMILY_STATUS'].value_counts()
trace = go.Bar(x=temp.index, y=temp, 
               marker=dict(color='coral'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Distribution of Applicant\'s Marital Status', titlefont=dict(size=22),
                   xaxis=dict(title='Marital Status',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)),
                   yaxis=dict(title='Count',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ### 4.1.8 Applicant's Housing Type

# In[ ]:


temp = df_train['NAME_HOUSING_TYPE'].value_counts()
trace = go.Bar(x=temp.index, y=temp, 
               marker=dict(color='plum'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Distribution of Applicant\'s Housing Type', titlefont=dict(size=22),
                   xaxis=dict(title='Housing Type',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14),
                              tickangle=20),
                   yaxis=dict(title='Count',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ### 4.1.9 Applicant's Education

# In[ ]:


temp = df_train['NAME_EDUCATION_TYPE'].value_counts()
trace = go.Bar(x=temp.index, y=temp, 
               marker=dict(color='yellow'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Distribution of Applicant\'s Education', titlefont=dict(size=22),
                   xaxis=dict(title='Education',
                              titlefont=dict(size=16),
                              tickfont=dict(size=12),
                              tickangle=0),
                   yaxis=dict(title='Count',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ### 4.1.10 Applicant's Income Source

# In[ ]:


temp = df_train['NAME_INCOME_TYPE'].value_counts()
trace = go.Bar(x=temp.index, y=temp, 
               marker=dict(color='powderblue'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Distribution of Applicant\'s Income Source', titlefont=dict(size=22),
                   xaxis=dict(title='Income Source',
                              titlefont=dict(size=16),
                              tickfont=dict(size=12),
                              tickangle=0),
                   yaxis=dict(title='Count',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ### 4.1.11 Applicant's Occupation

# In[ ]:


temp = df_train['OCCUPATION_TYPE'].value_counts()
trace = go.Bar(x=temp.index, y=temp, 
               marker=dict(color='powderblue'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Distribution of Applicant\'s Occupation', titlefont=dict(size=22),
                   xaxis=dict(title='Occupation',
                              titlefont=dict(size=16),
                              tickfont=dict(size=12)),
                   yaxis=dict(title='Count',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# The top 3 occupations are:
# - laborer
# - sales staff
# - core staff

# ### 4.1.12 Applicant's Work Industry

# In[ ]:


temp = df_train['ORGANIZATION_TYPE'].value_counts().sort_values().tail(20)
trace = go.Bar(x=temp, y=temp.index, orientation='h',
               marker=dict(color='powderblue'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Distribution of Applicant\'s Work Industry', width = 900, titlefont=dict(size=22),
                   xaxis=dict(title='Count',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)),
                   yaxis=dict(title='Industry',
                              titlefont=dict(size=16),
                              tickfont=dict(size=12)),
                   margin=dict(l=200))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# The top 5 work industries are:
# - Business Entity Type 3
# - XNA
# - Self-employed
# - Other
# - Medicine

# ### 4.1.13 Applicant's Age

# In[ ]:


df_train['AGE'] = -(df_train['DAYS_BIRTH'])/365


# In[ ]:


plt.figure(figsize=(12, 5))
sns.distplot(df_train['AGE'].dropna())
plt.title('Distribution of Applicant\'s Age', fontsize=20)
plt.xlabel('Age');


# The distribution of applicant's age makes sense.

# ### 4.1.14 Applicant's Years of Employment

# In[ ]:


plt.figure(figsize=(12, 5))
sns.distplot(-(df_train['DAYS_EMPLOYED'].dropna())/365)
plt.title('Distribution of Applicant\'s Years of Employment', fontsize=20)
plt.xlabel('Year');


# Some observations have years of employment = -1000 and this doesn't make sense. We need to code these observations as missing values.

# ## 4.2 Default Rate Analysis

# ### 4.2.1 Default Rate by Applicant's Age

# In[ ]:


df_train['AGE_BIN'] = pd.cut(df_train['AGE'], np.linspace(20, 70, num = 11))


# In[ ]:


temp = df_train.groupby('AGE_BIN')['TARGET'].mean()
trace = go.Bar(x=temp.index.astype(str), y=temp, 
               marker=dict(color='pink'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Default Rate by Applicant\'s Age', titlefont=dict(size=22),
                   xaxis=dict(title='Age Bin',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)),
                   yaxis=dict(title='Default Rate (%)',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# Default rate is negatively correlated with applicant's age.

# ### 4.2.2 Default Rate by Loan Type

# In[ ]:


temp = df_train.groupby('NAME_CONTRACT_TYPE')['TARGET'].mean()
trace = go.Bar(x=temp.index, y=temp, 
               marker=dict(color='pink'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Default Rate by Loan Type', titlefont=dict(size=22),
                   xaxis=dict(title='Loan Type',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)),
                   yaxis=dict(title='Default Rate (%)',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# Cash loans have higher default rate.

# ### 4.2.3 Default Rate by Region

# In[ ]:


temp = df_train.groupby('REGION_RATING_CLIENT_W_CITY')['TARGET'].mean()
trace = go.Bar(x=['Region1', 'Region2', 'Region3'], y=temp, 
               marker=dict(color='pink'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Default Rate by Region', titlefont=dict(size=22),
                   xaxis=dict(title='Region',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)),
                   yaxis=dict(title='Default Rate (%)',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# Region 3 has the highest default rate.

# ### 4.2.4 No. of Default in Social Surroundings

# In[ ]:


temp = df_train.groupby('TARGET')['DEF_30_CNT_SOCIAL_CIRCLE'].mean()
trace = go.Bar(x=['Non-Default', 'Default'], y=temp, 
               marker=dict(color='pink'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Mean No. of People in Applicant\'s Social Surroundings Defaulted on 30 DPD', 
                   titlefont=dict(size=20),
                   xaxis=dict(titlefont=dict(size=16),
                              tickfont=dict(size=14)),
                   yaxis=dict(title='No. of Default',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


temp = df_train.groupby('TARGET')['DEF_60_CNT_SOCIAL_CIRCLE'].mean()
trace = go.Bar(x=['Non-Default', 'Default'], y=temp, 
               marker=dict(color='pink'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Mean No. of People in Applicant\'s Social Surroundings Defaulted on 60 DPD',
                   titlefont=dict(size=20),
                   xaxis=dict(titlefont=dict(size=16),
                              tickfont=dict(size=14)),
                   yaxis=dict(title='No. of Default',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# Applicants who are not able to repay tend to be surrounded by people who cannot repay too.

# # More to come!
