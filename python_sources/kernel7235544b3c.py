#!/usr/bin/env python
# coding: utf-8

# # Executive Summary

# This project aims to classify whether the applicant will default on the loan given applicantion information and past credit histories data. This project will approach it as a binary classification problem. 
# 
# Data Cleaning:
# 
# Each applicant past credit data are given if available. However, each applicant can have more than 2 installments or credit records thus need to summarise the past credit records per applicant. 
# Records like number of days past due for installment, percentage of payment made, max number of credit card days due etc are summarised for each applicant.
# Records like who accompanied the applicant to apply, if given area different from residential area, if email is given are some of the columns that are deem not important. These are removed.
# Categorical data are converted to labels.
# 
# Modeling
# 
# The model used for classification problem. The train and test data at 75% & 25% split. All the features are used as input. 
# 
# 1. Logistics Regression 
# 2. Decision Tree with Pruning 
# 3. Random Forest 
# 4. Neural Network 
# 

# # Problem Statement / Research Topic

# Data Source: https://www.kaggle.com/c/home-credit-default-risk/data
# 
# This was a data competition hosted at Kaggle.com. 
# 
# There are clients who are unable to get a loan due to their limited credit histories. In order not to penalize this group of clients, Home Credit wanted to explore other transactional data if it can help to determine whether the applicant is able to repay a loan.
# 
# This research can bring a win win situation for both Home Credit and the clients. Clients are able to loan and get their home and Home Credit can increase their loan take up rate at reduced risk.
# 
# This project is to predict whether the applicant will default on the loan. There will be 2 outcomes (Yes/No).

# # Data Cleaning
# 

# - previous_application.csv
# - installments_payments.csv
# - credit_card_balance.csv
# 
# Conslidate the payment histories for installments and credit card of the applicants 
# 
# #### Output
# - Merge both data files
# - "SK_ID_CURR" is the key
# - csv file to be merged with other data files

# In[ ]:


import csv
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Source

from scipy.stats import entropy
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[ ]:


DIMS=(16, 15)

# matrix bar chart 
def drawbar(n, df):
    fig = plt.figure(figsize=DIMS)
    drow = math.ceil(n/2)
    for i in range(n):
        s = df.iloc[:, i].groupby(df.iloc[:, i]).size()
        fig.tight_layout()
        ax = fig.add_subplot(drow, 2,i+1)
        s.plot.bar()
        ax.set_title(df.columns[i], fontsize = 12)
    plt.show()
        
# data proportion
def showpro(n, df):
    for i in range(n):
        s = df.iloc[:, i].groupby(df.iloc[:, i]).size()/len(df)*100
        print (s)
        print ("------")

        
# matrix box plot
def drawbox(n,df):
    fig = plt.figure(figsize=DIMS)
    drow = math.ceil(n/2)
    for i in range(n):
        fig.tight_layout()
        ax = fig.add_subplot(drow, 2,i+1)
        df.iloc[:, i].plot(kind='box', ax=ax)
        ax.set_title(df.columns[i], fontsize = 12)
    plt.show()
    
# matrix histogram    
def drawdistplot(n, df, bins):
    fig = plt.figure(figsize=DIMS)
    drow = math.ceil(n/2)
    for i in range(n):
        fig.tight_layout()
        ax = fig.add_subplot(drow, 2,i+1)
        sns.distplot(df.iloc[:, i],kde=False, bins = bins)
        ax.set_title(df.columns[i])
    plt.show()

    
# data proportion with respect to Y (TARGET)    
def crosst(n, df):
    for i in range(n-1):
        print (pd.crosstab(df.iloc[:, i+1], df.TARGET, normalize='index'))
        print ("--------")

# percentage of non-null value in each column        
def summary(df, p):
    n = len(df.columns)
    totalrow = df.shape[0]
    print (totalrow)
    for i in range(n):
        col = df.columns[i]
        cnt = df.iloc[:,i].count()
        percentage = (cnt/totalrow*100)
        if (percentage <= p):
            print ('column: ', col, ' - ', cnt, '@' , percentage)


# In[ ]:


# Read csv file

df_prevapp = pd.read_csv('../input/previous_application.csv')
df_install = pd.read_csv('../input/installments_payments.csv')
df_credit = pd.read_csv('../input/credit_card_balance.csv')


# # Clean Installments Payments

# In[ ]:


# Read the columns name in installments payments file
print (df_install.columns)
print (" ------- ")
print (df_install.describe())


# In[ ]:


# box plot visualization for'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT'

pic = df_install[['DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT',
       'AMT_INSTALMENT', 'AMT_PAYMENT']]        
n = len(pic.columns)
drawbox(n,pic)

# so many outliers for intallment payments


# In[ ]:


# histogram plot visualization for'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT'

pic = df_install[['DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT',
       'AMT_INSTALMENT', 'AMT_PAYMENT']]        
n = len(pic.columns)
drawdistplot(n,pic,100)


# ###  Calculate Days Past Due 
# 
# - 'DAYS_INSTALMENT' - 'DAYS_ENTRY_PAYMENT'
# - If > 0 means clients pay on time  
# - If < 0 means clients did not pay on time
#     
# ### Calculate Proportion of the Installment Paid
# 
# - 'AMT_PAYMENT' / 'AMT_INSTALMENT'
# - Proportion of the installment paid
# - If 1 means 100% paid
# - If 0.9 means 90% paid only

# In[ ]:



# Calculate Days Past Due

def cal_days_past(dfdueday,dfpayday):
    late = dfdueday-dfpayday
    if late >= 0:
        return np.NaN
    else:
        return abs(late)
    
# Calculate Proportion of the Installment Paid

def cal_percent_paid(dfpay,dfinstall):
    mis=dfpay-dfinstall
    if mis >=0:
        return 1.0
    else:
        return abs(dfpay)/dfinstall


# In[ ]:


# copy the dataframe into df_i
df_i = df_install.copy()


# In[ ]:


# check how many 0 installment amount
# there are 290 rows with 0 installment needed
# remove these 290 rows

len(df_i[df_i['AMT_INSTALMENT']==0])


# In[ ]:


# remove 'AMT_INSTALMENT' = 0

df_i = df_i[df_i.AMT_INSTALMENT != 0]


# In[ ]:


# New column : Calculate Days Past Due

df_i['Days_Past']=  df_i.apply(lambda x: cal_days_past(x["DAYS_INSTALMENT"],
                                                                           x["DAYS_ENTRY_PAYMENT"]),axis =1)


# In[ ]:


# New column : Calculate Proportion of the Installment Paid

df_i['Payment_Made']=  df_i.apply(lambda x: cal_percent_paid(x["AMT_PAYMENT"],
                                                                           x["AMT_INSTALMENT"]),axis =1)


# In[ ]:


# groupby SK_ID_PREV
# Get the mean of installment amout, mean number of days past due and mean of the proportion 
# 0 values are already in nan form so that it is not part of the mean calculation

df_i1 = df_i.groupby(["SK_ID_PREV", "SK_ID_CURR"]).agg({'AMT_INSTALMENT': 'mean' ,
                                                                    'Days_Past':'mean', 
                                                                    'Payment_Made':'mean'}).reset_index()


# In[ ]:


# replace all Nan to 0

df_i1 = df_i1.fillna(0)


# #### Cleaned Installment Data groupby SK_ID_PREV
# 

# In[ ]:


df_i1.head()


# # Clean Previous Application

# In[ ]:


# Read the columns name in previous application file
print (df_prevapp.columns)
print (" ------- ")
print (df_prevapp.describe())


# In[ ]:


# first draft of keeping columns that looks useful
# rate and usage of the loans are removed
# reduce the columns for analysis

tokeep =['SK_ID_PREV', 'SK_ID_CURR', 'AMT_CREDIT', 'FLAG_LAST_APPL_PER_CONTRACT', 
         'NFLAG_LAST_APPL_IN_DAY', 'NAME_CONTRACT_STATUS', 'CNT_PAYMENT', 
         'PRODUCT_COMBINATION','NFLAG_INSURED_ON_APPROVAL']


# In[ ]:


# reduced columns 
df_pa = df_prevapp[tokeep]


# # Merge installments and previous application

# In[ ]:


# merge installment and previous application data by SK_ID_PREV

df_pi = pd.merge(df_pa,df_i1, on="SK_ID_PREV", how= "left")


# In[ ]:


# Drop duplicate column 'SK_ID_CURR_y'

df_pi.drop(['SK_ID_CURR_y'], axis=1, inplace = True)


# In[ ]:


# Check 

print (df_pi["NAME_CONTRACT_STATUS"].unique())
print (df_pi["FLAG_LAST_APPL_PER_CONTRACT"].unique())

pic = df_pi[["NAME_CONTRACT_STATUS","FLAG_LAST_APPL_PER_CONTRACT" ]]        
n = len(pic.columns)
showpro(n,pic)

# Only keep approved and refused application
# Only keep FLAG_LAST_APPL_PER_CONTRACT == Y


# In[ ]:


# Keep only approved and refused application
# keep FLAG_LAST_APPL_PER_CONTRACT == Y

# approved and refused application
cond2 = df_pi["NAME_CONTRACT_STATUS"] == "Approved"
cond3 = df_pi["NAME_CONTRACT_STATUS"] == "Refused"

df_pi = df_pi[cond2|cond3]

# keep Y
cond4 = df_pi["FLAG_LAST_APPL_PER_CONTRACT"] == 'Y'
df_pi = df_pi[cond4]


# In[ ]:


# Groupby SK_ID_CURR_x
# MAX Credit Requested, SUM insurance bought
# MEAN of installment amount
# MEAN of days due past and proportion of payment made

df_pi_01 = df_pi.groupby("SK_ID_CURR_x").agg({'AMT_CREDIT':'max',
                                          'NFLAG_INSURED_ON_APPROVAL': 'sum',
                                          'AMT_INSTALMENT': 'mean',
                                          'Days_Past': 'mean',
                                          'Payment_Made': 'mean'}).reset_index()


# In[ ]:


df_pi_01.head()


# In[ ]:


# count number of installments per ID_CURR
# count number of approved and refused applications for each ID_CURR

df_pi_02 = pd.concat([df_pi[['SK_ID_CURR_x', 'SK_ID_PREV']].groupby('SK_ID_CURR_x').count().rename(columns={'SK_ID_PREV' :'count'}),
                   pd.crosstab(df_pi.SK_ID_CURR_x, df_pi.NAME_CONTRACT_STATUS)]
                  ,1).reset_index()


# In[ ]:


# Merge the 2 dataframes

df_prei = pd.merge(df_pi_01,df_pi_02, on="SK_ID_CURR_x", how= "left")


# In[ ]:


# rename the SK_ID_CURR_x to SK_ID_CURR_
df_prei = df_prei.rename(columns={'SK_ID_CURR_x': 'SK_ID_CURR_'})


# # Clean Credit Card Data

# In[ ]:


df_credit.columns


# * Remove columns that is not useful to reduce the complexity
# * Keep 'SK_ID_PREV', 'SK_ID_CURR',
# * Keep 'MONTHS_BALANCE', 'AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL'
# * Keep 'AMT_DRAWINGS_CURRENT'
# * Keep 'AMT_INST_MIN_REGULARITY' 'AMT_PAYMENT_TOTAL_CURRENT'
# * Keep 'AMT_RECEIVABLE_PRINCIPAL', 'AMT_TOTAL_RECEIVABLE'
# * Keep Total CNT_Drawings_Current
# * Remove 'CNT_INSTALMENT_MATURE_CUM'
# * Keep 'NAME_CONTRACT_STATUS'
# * Remove 'SK_DPD', Keep 'SK_DPD_DEF'

# In[ ]:


# Keep relevant columns 
keepcol =['SK_ID_PREV', 'SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_BALANCE',
       'AMT_CREDIT_LIMIT_ACTUAL', 
       'AMT_DRAWINGS_CURRENT', 
       'AMT_INST_MIN_REGULARITY',
       'AMT_PAYMENT_TOTAL_CURRENT',
       'AMT_RECEIVABLE_PRINCIPAL', 'AMT_TOTAL_RECEIVABLE',
       'CNT_DRAWINGS_CURRENT',
       'NAME_CONTRACT_STATUS','SK_DPD_DEF']


# In[ ]:


# Reduced columns dataframe 

df_c = df_credit[keepcol]


# In[ ]:


# remove rows with 0 'AMT_BALANCE' : no amt owed

df_c = df_c[df_c.AMT_BALANCE != 0]


# In[ ]:


# replace 0 with nan
# to exclude 0 from calculation

cols = ["AMT_DRAWINGS_CURRENT",'SK_DPD_DEF']
df_c[cols] = df_c.loc[:,cols].replace({0:np.nan})


# In[ ]:


# group by SK_ID_PREV
# extract frequency of credit card monthly use, and avg credit used 
# Max number of Day Due Past alert

df_c1 = (
    df_c.groupby(["SK_ID_PREV", "SK_ID_CURR"])
    .agg({'AMT_DRAWINGS_CURRENT': ['count' , 'mean'] , 'SK_DPD_DEF':'max'})
    .reset_index()
)

df_c1.columns = ["_".join(x) for x in df_c1.columns.ravel()]


# In[ ]:


# replace all nan to 0

df_c1.fillna(0, inplace = True)


# In[ ]:


print (len(df_c1['SK_ID_PREV_'].unique()))
print (len(df_c1['SK_ID_CURR_'].unique()))

# There are ID_CURR with more than 1 ID_PREV# group by SK_ID_CURR 


# In[ ]:


# group by SK_ID_CURR 
# avg amt credit used, total freq monthly usage, 
# to merge with installment and preapp data, max number of days past due alert

df_c2 = (
    df_c1.groupby('SK_ID_CURR_')
    .agg({'AMT_DRAWINGS_CURRENT_count' :'sum', 'AMT_DRAWINGS_CURRENT_mean' :'mean', 'SK_DPD_DEF_max': 'max' })
    .reset_index()
)


# # Merge Installment, Previous Application and Credit Card Data
# 

# In[ ]:


# merge credit , installment and pre app data into 1 dataframe

df_precredinstall = pd.merge(df_prei, df_c2, on="SK_ID_CURR_", how= "left")


# In[ ]:


df_precredinstall.to_csv("precredinstall.csv", index = False)


# # Clean Application File 

# In[ ]:


df = pd.read_csv('../input/application_train.csv')


# In[ ]:


print (df.shape)


# ## Features Selection

# In[ ]:


# pandas.core.series.Series
datatype = df.dtypes

print (datatype.unique())

cond1 = datatype == 'float64'
floatcol = datatype[cond1].index.tolist()

cond2 = datatype == 'int64'
intcol = datatype[cond2].index.tolist()

cond3 = datatype == 'O'
objcol = datatype[cond3].index.tolist()


# ## Remove strong correlated colummns (Float and Int Data)
# 
# 

# In[ ]:


## Float Data Columns

df_float = df[floatcol]


# In[ ]:


# correlated features
#Using Pearson Correlation

plt.figure(figsize=(16,16))
corf = df_float.corr()
sns.heatmap(corf, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[ ]:



upperf = corf.where(np.triu(np.ones(corf.shape), k=1)
                          .astype(np.bool))

#print (upper)
to_dropf = [column for column in upperf
           .columns if any(upperf[column] > 0.9)]

print (to_dropf)
print (len(to_dropf))


# In[ ]:


# Int Data Columns
df_int =df[intcol]


# In[ ]:


# correlated features
#Using Pearson Correlation


plt.figure(figsize=(16,16))
cori = df_int.corr()
sns.heatmap(cori, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[ ]:


upperi = cori.where(np.triu(np.ones(cori.shape), k=1)
                          .astype(np.bool))

#print (upper)
to_dropi = [column for column in upperi
           .columns if any(upperi[column] > 0.90)]

print (to_dropi)
print (len(to_dropi))


# In[ ]:


# where 1 is the axis number (0 for rows and 1 for columns.)
to_dropf.extend(to_dropi)

print (to_dropf)
df_xcorr = df.drop(to_dropf, 1)


# In[ ]:


df_xcorr.shape


# ## Remove features with many null values

# In[ ]:


summary(df_xcorr,35)

# remove columns of data with more than 65% null values

colnull = ['OWN_CAR_AGE', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 
           'FLOORSMIN_AVG', 'NONLIVINGAPARTMENTS_AVG', 'FONDKAPREMONT_MODE']


# In[ ]:


df_xcorr1 = df_xcorr.drop(colnull, 1)
print (df_xcorr1.shape)


# ## Go through remaining features

# In[ ]:


print (df_xcorr1.columns[:20])
print (df_xcorr1.columns[20:40])
print (df_xcorr1.columns[40:60])
print (df_xcorr1.columns[60:])


# * Remove (random behavior)
# * 'REGION_POPULATION_RELATIVE'
# * 'WEEKDAY_APPR_PROCESS_START' 
# * 'HOUR_APPR_PROCESS_START'  
# * 'DAYS_REGISTRATION'
# * 'DAYS_ID_PUBLISH'
# * 'NAME_TYPE_SUITE'
# * 'DAYS_LAST_PHONE_CHANGE'

# In[ ]:


col =['TARGET', 'CODE_GENDER', 'NAME_CONTRACT_TYPE', 'REGION_RATING_CLIENT']

pic = df_xcorr[col]
n = len(pic.columns)
drawbar(n,pic)

showpro(n,pic)

# remove XNA code_gender


# In[ ]:


col =['CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE']

pic = df_xcorr[col]
n = len(pic.columns)
drawbar(n,pic)

showpro(n,pic)


# keep 'CNT_FAM_MEMBERS'
# 70% have 0 children


# In[ ]:


col = [ 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']

pic =df_xcorr[col]
#pic.head()

n = len(pic.columns)
drawbar(n,pic)


# In[ ]:


col = [ 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']

pic = df_xcorr[col]        
n = len(pic.columns)
drawdistplot(n,pic,10)



# keep basic information


# In[ ]:


col = [ 'REGION_POPULATION_RELATIVE', 
        'DAYS_BIRTH',  'DAYS_EMPLOYED',  'DAYS_REGISTRATION','DAYS_ID_PUBLISH',
        'OWN_CAR_AGE']


pic = df_xcorr[col]        
n = len(pic.columns)
drawdistplot(n,pic,20)

# Remove 
# DAYS_REGISTRATION','DAYS_ID_PUBLISH',  -  It should not affect credibility


# In[ ]:


col =['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

pic = df_xcorr[col]        
n = len(pic.columns)
drawdistplot(n,pic,10)

# Keep all 3 columns,
# unknown data


# In[ ]:


pic = df_xcorr1[['REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']]

n = len(pic.columns)

showpro(n,pic)

pic = df_xcorr1[['TARGET', 'REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']]


crosst(n,pic)

plt.figure(figsize=(16,16))
corx = pic.corr()
sns.heatmap(corx, annot=True, cmap=plt.cm.Reds)
plt.show()


upperx = corx.where(np.triu(np.ones(corx.shape), k=1)
                          .astype(np.bool))

#print (upper)
to_dropx = [column for column in upperx
           .columns if any(upperx[column] > 0.80)]

print (to_dropx)
print (len(to_dropx))


# In[ ]:


pic = df_xcorr[['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
       'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']]        

n = len(pic.columns)
drawbar(n,pic)


showpro(n,pic)

# Since 100% provided mobile
# All columns not needed


# In[ ]:


pic = df_xcorr1[['AMT_REQ_CREDIT_BUREAU_HOUR',
       'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
       'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
       'AMT_REQ_CREDIT_BUREAU_YEAR']]

n = len(pic.columns)

drawbar(n,pic)


#keep 'AMT_REQ_CREDIT_BUREAU_YEAR'


# In[ ]:


pic = df_xcorr1[['FLAG_DOCUMENT_2',
       'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
       'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
       'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',
       'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
       'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
       'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
       'FLAG_DOCUMENT_21']]

n = len(pic.columns)

showpro(n,pic)



# Remove 99% are 0, 'FLAG_DOCUMENT_2','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_9' , 'FLAG_DOCUMENT_10',
#  'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
#'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 
#'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'


#Keep'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', , 'FLAG_DOCUMENT_8',


# In[ ]:


col = ['APARTMENTS_AVG', 'BASEMENTAREA_AVG',
'YEARS_BEGINEXPLUATATION_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
'FLOORSMAX_AVG', 'LANDAREA_AVG', 'NONLIVINGAREA_AVG']
       
pic = df_xcorr1[col]        
n = len(pic.columns)
drawdistplot(n,pic,10)

col = ['HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']
pic = df_xcorr1[col]        
n = len(pic.columns)
drawbar(n,pic)
showpro(n,pic)

#remove 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'


# In[ ]:


col = ['OBS_30_CNT_SOCIAL_CIRCLE',
       'DEF_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']

pic = df_xcorr1[col]

n = len(pic.columns)
drawbar(n,pic)


# Keep 'DEF_30_CNT_SOCIAL_CIRCLE'
# interested in defaulted behaviour


# ### Columns kept for modelling 

# In[ ]:


finalcol = ['SK_ID_CURR', 'TARGET', 'CODE_GENDER', 'NAME_CONTRACT_TYPE', 'REGION_RATING_CLIENT', 
            'CNT_FAM_MEMBERS', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE','NAME_INCOME_TYPE', 
            'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
            'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
            'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 
            'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'AMT_REQ_CREDIT_BUREAU_YEAR', 
            'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_8', 
            'APARTMENTS_AVG', 'BASEMENTAREA_AVG','YEARS_BEGINEXPLUATATION_AVG', 'ELEVATORS_AVG', 
            'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'LANDAREA_AVG', 'NONLIVINGAREA_AVG', 'DEF_30_CNT_SOCIAL_CIRCLE']

df_app = df[finalcol]


# In[ ]:


col = df_app['CODE_GENDER'].isin(['F', 'M'])
df_app = df_app[col]


# ### AMT ANNUITY
# 
# - There are null amt annuity 
# - To be removed
# 
# ### Replace Nan to 0

# In[ ]:


# drop nan AMT_ANNUITY
df_app.dropna(subset=['AMT_ANNUITY'], inplace = True)

df_app.fillna(0,inplace=True)


# In[ ]:


df_app.shape


# ## Replace categorical data into labels

# In[ ]:


df_app.NAME_CONTRACT_TYPE.replace(['Cash loans', 'Revolving loans'],[0,1], inplace = True)
df_app.CODE_GENDER.replace(['M', 'F'], [0, 1], inplace = True)
df_app.FLAG_OWN_CAR.replace(['Y', 'N'], [1,0], inplace = True)
df_app.FLAG_OWN_REALTY.replace(['Y', 'N'], [1,0], inplace = True)
df_app.NAME_INCOME_TYPE.replace(['Businessman', 'State servant', 'Commercial associate'
                                 , 'Working', 'Maternity leave', 'Pensioner', 'Unemployed', 'Student'],
                               [0,1,2,3,4,5,6,7], inplace = True)

df_app.NAME_EDUCATION_TYPE.replace(['Academic degree', 'Higher education', 'Incomplete higher',
                            'Secondary / secondary special', 'Lower secondary'],
                           [0,1,2,3,4], inplace = True)

df_app.NAME_FAMILY_STATUS.replace(['Married','Civil marriage', 'Single / not married', 'Separated', 
                                   'Widow', 'Unknown'],[0,1,2,3,4,5], inplace = True)

df_app.NAME_HOUSING_TYPE.replace(['House / apartment',  'Municipal apartment', 'Office apartment', 
                          'Co-op apartment', 'Rented apartment', 'With parents'],
                        [0,1,2,3,4,5], inplace = True)

df_app.OCCUPATION_TYPE.replace(['Laborers', 'Core staff', 'Accountants', 'Managers', 
                                 'Drivers', 'Sales staff', 'Cleaning staff', 'Cooking staff',
                                 'Private service staff', 'Medicine staff', 'Security staff',
                                 'High skill tech staff', 'Waiters/barmen staff', 
                                 'Low-skill Laborers', 'Realty agents', 'Secretaries', 'IT staff','HR staff'],
                              [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], inplace = True)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df_app['ORGANIZATION_TYPE'] = labelencoder.fit_transform(df_app['ORGANIZATION_TYPE'])


# # Clean POS Cash Balance

# In[ ]:


df_pos = pd.read_csv('../input/POS_CASH_balance.csv')


# In[ ]:


df_pos.columns


# In[ ]:


# Retrieve the max value of days due past

df_pos1 = (
    df_pos.groupby(["SK_ID_PREV", "SK_ID_CURR"])
    .agg({'SK_DPD_DEF':'max'})
    .reset_index()
)


# In[ ]:


# Groupby ID_CURR

df_pos2 = (
    df_pos1.groupby('SK_ID_CURR')
    .agg({'SK_DPD_DEF': 'max' })
    .reset_index()
)


# In[ ]:


# write POS data to csv

df_pos2.to_csv("POS.csv", index = False)


# # Merge All Data to Application Data

# In[ ]:


# read application data
df_01 = pd.read_csv('precredinstall.csv')


# In[ ]:


df_01 = df_01.rename(columns={'SK_ID_CURR_': 'SK_ID_CURR'})
df_01.fillna(0,inplace=True)


# In[ ]:


# read POS Cash data
df_02 =  pd.read_csv('POS.csv')


# In[ ]:


df_02 = df_02.rename(columns={'SK_DPD_DEF': 'DPD_DEF_POS'})


# ## Merging of all csv files

# In[ ]:


df_f1 = pd.merge(df_app,df_01, on="SK_ID_CURR", how= "left")


# In[ ]:


df_f2 = pd.merge(df_f1,df_02, on="SK_ID_CURR", how= "left")


# ## After merging
# 
# - Replace Nan with 0

# In[ ]:


df_f2.fillna(0,inplace=True)


# In[ ]:


df_f2.to_csv("final.csv", index =False)


# In[ ]:


df_f2.info()


# # Generate Predictive Models

# In[ ]:


df = pd.read_csv('final.csv')


# In[ ]:


df.columns


# In[ ]:


s = df['TARGET'].groupby(df['TARGET']).size()
print (s)

fig = plt.figure(figsize=(10,6))
s.plot.bar()

plt.show()

# data Imbalance


# 
# ### Data Imbalance
# 
# Target 1 & 0
# However, 92% are 0 and 8% are 1 as shown in the graph
# 
# #### To balance the data for training and testing
# 
# - Target 1 : Select All (24,825 rows)
# - Target 0 : Random pick 25,000 rows
# 

# In[ ]:


# extract target = 1
cond1 = df['TARGET'] == 1
df_1 = df[cond1]

# extract target = 0
cond2 = df['TARGET'] == 0
df_0 = df[cond2]

# random pick 25,000 rows of target = 0
df_0 = df_0.sample(n = 25000, random_state = 1)

print (len(df_1), len(df_0))

# concat both target = 1 and 0 into dataframe
df_c = pd.concat([df_1,df_0])


# ###  Assign X & Y

# In[ ]:


X = df_c[['CODE_GENDER', 'NAME_CONTRACT_TYPE',
       'REGION_RATING_CLIENT', 'CNT_FAM_MEMBERS', 'OCCUPATION_TYPE',
       'ORGANIZATION_TYPE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_OWN_CAR',
       'FLAG_OWN_REALTY', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL',
       'AMT_CREDIT_x', 'AMT_ANNUITY', 'EXT_SOURCE_1', 'EXT_SOURCE_2',
       'EXT_SOURCE_3', 'REG_REGION_NOT_LIVE_REGION',
       'REG_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
       'REG_CITY_NOT_WORK_CITY', 'AMT_REQ_CREDIT_BUREAU_YEAR',
       'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
       'FLAG_DOCUMENT_8', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG',
       'YEARS_BEGINEXPLUATATION_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
       'FLOORSMAX_AVG', 'LANDAREA_AVG', 'NONLIVINGAREA_AVG',
       'DEF_30_CNT_SOCIAL_CIRCLE', 'AMT_CREDIT_y', 'NFLAG_INSURED_ON_APPROVAL',
       'AMT_INSTALMENT', 'Days_Past', 'Payment_Made', 'count', 'Approved',
       'Refused', 'AMT_DRAWINGS_CURRENT_count', 'AMT_DRAWINGS_CURRENT_mean',
       'SK_DPD_DEF_max', 'DPD_DEF_POS']]

y = df_c['TARGET']


# ### Split data into train (75%) and test data (25%)

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)


# In[ ]:


print (len(X_train), len(X_test))
print (len(y_train), len(y_test))


# ## Modeling with all the columns as Input

# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_train)
print('Train accuracy score:',accuracy_score(y_train,y_pred))
print('Test accuracy score:', accuracy_score(y_test,logreg.predict(X_test)))


# ### Decision Tree - No Pruning

# In[ ]:


dtree = DecisionTreeClassifier(criterion="entropy")
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_train)
print('Train accuracy score:',accuracy_score(y_train,y_pred))
print('Test accuracy score:', accuracy_score(y_test,dtree.predict(X_test)))


# ### Decision Tree - Pruning

# In[ ]:


dptree = DecisionTreeClassifier(max_depth=5, criterion="entropy")
dptree.fit(X_train, y_train)
y_pred = dptree.predict(X_train)
print('Train accuracy score:',accuracy_score(y_train,y_pred))
print('Test accuracy score:', accuracy_score(y_test,dptree.predict(X_test)))


# ### Random Forest 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier as RFC
rfc_b = RFC(n_estimators=100)
rfc_b.fit(X_train,y_train)
y_pred = rfc_b.predict(X_train)
print('Train accuracy score:',accuracy_score(y_train,y_pred))
print('Test accuracy score:', accuracy_score(y_test,rfc_b.predict(X_test)))


# overfit on the training data


# ### NN Model

# In[ ]:


# standardize the data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_trainS = scaler.transform(X_train)
X_testS = scaler.transform(X_test)


# In[ ]:


# run the model

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=3000)
mlp.fit(X_trainS,y_train)
y_pred = mlp.predict(X_trainS)

print('Train accuracy score:',accuracy_score(y_train,y_pred))
print('Test accuracy score:', accuracy_score(y_test,mlp.predict(X_testS)))


# ## Modeling with selected 20 features

# ## Feature selection

# In[ ]:


features = [x for i,x in enumerate(X_train.columns)]
# Find the full list of features in the dataset
print(features)


# ### Use Random Forest for Feature Selection

# In[ ]:



n_features = len(features)

plt.figure(figsize=(8,15))
plt.barh(range(n_features), rfc_b.feature_importances_, align='center')
plt.yticks(np.arange(n_features), features)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.title("Feature Importance Plot")
plt.ylim(-1, n_features)
plt.show()


# ### Selected Features

# In[ ]:


# sort the more important features from random forest model
# select the top 20 features for modeling again


fi = rfc_b.feature_importances_
col = np.array(features)

df_features = pd.DataFrame({'c': col , 'f':fi})

selected = df_features.sort_values('f', ascending = False)[:20]['c'].tolist()
print (selected)


# In[ ]:


X_trainA = X_train[selected]
X_testA  = X_test[selected]


# ### Rerun Random Forest Model with Reduced 20 Features

# In[ ]:


rfc_bA = RFC(n_estimators=100)
rfc_bA.fit(X_trainA,y_train)
y_pred = rfc_bA.predict(X_trainA)
print('Train accuracy score:',accuracy_score(y_train,y_pred))
print('Test accuracy score:', accuracy_score(y_test,rfc_bA.predict(X_testA)))


# ### Rerun Neural Networ with Reduced 20 Features

# In[ ]:


scaler = StandardScaler()
scaler.fit(X_trainA)

X_trainS = scaler.transform(X_trainA)
X_testS = scaler.transform(X_testA)


from sklearn.neural_network import MLPClassifier
mlpf = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=3000)
mlpf.fit(X_trainS,y_train)
y_pred = mlpf.predict(X_trainS)

print('Train accuracy score:',accuracy_score(y_train,y_pred))
print('Test accuracy score:', accuracy_score(y_test,mlpf.predict(X_testS)))

