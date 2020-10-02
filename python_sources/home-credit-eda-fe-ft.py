#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#  importing required libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

#import os
#print(os.listdir("../input"))

get_ipython().run_line_magic('matplotlib', 'inline')


# **Functions**
# 
# missing data

# In[ ]:


# missing data
def missing_data(df):
    featureList = list(df)
    for removeColumn in ['SK_ID_PREV','SK_ID_CURR','isTrain','isTest','TARGET','SK_ID_BUREAU']:
        if removeColumn in featureList: 
            featureList.remove(removeColumn)
    train_total = df.loc[df.isTrain,featureList].isnull().sum()
    test_total = df.loc[df.isTest,featureList].isnull().sum()
    train_percent = (df.loc[df.isTrain,featureList].isnull().sum()/df.loc[df.isTrain,featureList].isnull().count()*100)
    test_percent = (df.loc[df.isTest,featureList].isnull().sum()/df.loc[df.isTest,featureList].isnull().count()*100)
    df = pd.concat([train_total, train_percent,test_total,test_percent], axis=1, keys=['train_Total', 'train_Percent','test_Total','trest_Percent'])
    return(df.query('train_Total >0 or test_Total >0'))

def missing_info(data, feature):
    return((data.loc[data.isTrain,feature].isnull().sum(),
            round((data.loc[data.isTrain,feature].isnull().sum()*100/data[data.isTrain].shape[0]),2),
            data.loc[data.isTest,feature].isnull().sum(),
            round((data.loc[data.isTest,feature].isnull().sum()*100/data[data.isTest].shape[0]),2)))


# Plots

# In[ ]:


# all in one plot
def plot_df(df,withTarget = True):
    featureList = list(df)
    for removeColumn in ['SK_ID_PREV','SK_ID_CURR','isTrain','isTest','TARGET','SK_ID_BUREAU']:
        if removeColumn in featureList: 
            featureList.remove(removeColumn)
    for feature in featureList:
        print(feature)
        if (df[feature].dtypes == np.object) | (len(df[feature].unique())<15):
            plot_stats(df,feature,withTarget)
        else:
            plot_distribution(df,feature,withTarget)

# distribution
def plot_distribution(df,feature,withTarget = True):
    df = df.dropna(axis=0, subset=[feature])
    if withTarget :
        fig, axes = plt.subplots(ncols=4,nrows=2, figsize=(20,10))
        s = sns.distplot(ax=axes[0,0], a = df[feature],kde=False)
        s = sns.distplot(ax=axes[0,1], a = df.loc[df.isTrain,feature].values,kde=False)
        s = sns.distplot(ax=axes[0,2], a = df.loc[df.isTest,feature].values,kde=False)
        s = sns.boxplot(ax=axes[0,3], x = 'TARGET', y=feature,data=df[df.isTrain])
        axes[0,0].set_title('Total')
        axes[0,1].set_title('Train')
        axes[0,2].set_title('Test')
        axes[0,3].set_title('TARGET=1')
        df = df.query('{0} !=0'.format(feature))
        df = df[~is_outlier(df[feature])]
        s = sns.distplot(ax=axes[1,0], a = df[feature],kde=False)
        s = sns.distplot(ax=axes[1,1], a = df.loc[df.isTrain,feature].values,kde=False)
        s = sns.distplot(ax=axes[1,2], a = df.loc[df.isTest,feature].values,kde=False)
        s = sns.boxplot(ax=axes[1,3], x = 'TARGET', y=feature,data=df[df.isTrain])
    else:
        fig, axes = plt.subplots(ncols=3,nrows=1, figsize=(20,4))
        s = sns.distplot(ax=axes[0], a = df[feature],kde=False)
        df = df.query('{0} !=0'.format(feature))
        s = sns.distplot(ax=axes[1], a = df[feature],kde=False)
        df = df[~is_outlier(df[feature])]
        s = sns.distplot(ax=axes[2], a = df[feature],kde=False)
        axes[0].set_title('with outliers')
        axes[1].set_title('without zero')
        axes[2].set_title('without outliers')
    for ax in fig.axes:
        matplotlib.pyplot.sca(ax)
        plt.xticks(rotation=45)
        ax.set(xlabel='')
    plt.show()
    
# bar plot
def plot_stats(df,feature,withTarget = True):
    sns.set_color_codes("pastel")
    if(withTarget):
        print(chisq_of_df_cols(df, feature))
        if(len(df[feature].unique()) <8):
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20,4))
        else:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(20,25))

        s = sns.countplot(ax=ax1,x=feature, data=df, orient="h")
        s = sns.countplot(ax=ax2,x=feature, data=df.query('isTrain == True'))
        s = sns.countplot(ax=ax3,x=feature, data=df.query('isTrain == False'))
        s = sns.countplot(ax=ax4,x=feature, data=df.query('isTrain == True and TARGET == 1'))
        ax1.set_title('Total')
        ax2.set_title('Train')
        ax3.set_title('Test')
        ax4.set_title('TARGET=1')
    else :
        if(len(df[feature].unique()) >6):
            plt.subplots(figsize=(20,5))
        plt.xticks(rotation=45)
        s = sns.countplot(x=feature, data=df, orient="h")
    plt.show();


# chi squared test

# In[ ]:


#chi squared test
def chisq_of_df_cols(df, feature):
    groupsizes = df.groupby([feature, 'TARGET']).size()
    ctsum = groupsizes.unstack(feature)
    # fillna(0) is necessary to remove any NAs which will cause exceptions
    chi2, p, ddof, expected = chi2_contingency(ctsum.fillna(0))
    return(p)


# Outlier remover

# In[ ]:


# Outlier remover
# Source: https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting (see references)

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


#  ***APPLICATION_TRAIN***
# 
# This is the main file. 
# * SK_ID_CURR is the ID of loan in our sample
# * Dimensions : Train : [ 307511 , 122 ] , test : [ 48744 , 121 ]

# Combining train and test

# In[ ]:


# combining dataset
def get_combined_dataset() :
    application_train = pd.read_csv('../input/application_train.csv')
    application_train['isTrain'] = True
    application_train['isTest'] = False
    application_test = pd.read_csv('../input/application_test.csv')
    application_test['isTest'] = True
    application_test['isTrain'] = False
    application=application_train.append(application_test, ignore_index=True,sort=False)
    application.set_index('SK_ID_CURR')
    return (application)


# Target variable 

# In[ ]:


application = get_combined_dataset()
sns.set(style="darkgrid")
ax = sns.countplot(x="TARGET", data=application.query('isTrain == True'))
for p in ax.patches:
    ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))


# What application_train/test look like

# In[ ]:


plot_df(application)


# Filtering /one hote encoding/ capping

# In[ ]:


def get_application_dataset():
    df = get_combined_dataset()
    filteredColList =['NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE',
                      'WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE','FONDKAPREMONT_MODE'] 
    df = df[[x for x in list(df) if x not in filteredColList]]
    oheCols = ['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','HOUSETYPE_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE']
    df.loc[df.CODE_GENDER == 'XNA' ,'CODE_GENDER'] = 'F'

    df.loc[(df.DAYS_EMPLOYED > 0),'DAYS_EMPLOYED'] = np.nan
    df.loc[(df.REGION_RATING_CLIENT_W_CITY < 0),'REGION_RATING_CLIENT_W_CITY'] = np.nan
    df.loc[(df.OBS_30_CNT_SOCIAL_CIRCLE > 10),'OBS_30_CNT_SOCIAL_CIRCLE'] = 10
    df.loc[(df.DEF_30_CNT_SOCIAL_CIRCLE > 10),'DEF_30_CNT_SOCIAL_CIRCLE'] = 10
    df.loc[(df.OBS_60_CNT_SOCIAL_CIRCLE > 10),'OBS_60_CNT_SOCIAL_CIRCLE'] = 10
    df.loc[(df.DEF_60_CNT_SOCIAL_CIRCLE > 10),'DEF_60_CNT_SOCIAL_CIRCLE'] = 10
    df.loc[(df.AMT_REQ_CREDIT_BUREAU_QRT > 10),'AMT_REQ_CREDIT_BUREAU_QRT'] = 10
    df = pd.get_dummies(df,columns=oheCols)
    
    df['NEW_INCOME2Credit']=df['AMT_CREDIT']/df['AMT_INCOME_TOTAL']
    df['NEW_Credit2ANNUITY']=df['AMT_ANNUITY']/df['AMT_CREDIT']
    df['NEW_INCOME2ANNUITY']=df['AMT_ANNUITY']/df['AMT_INCOME_TOTAL']
    df['NEW_DAYS_EMPLOYED2DAYS_BIRTH'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_AMT_INCOME_TOTAL2CNT_FAM_MEMBERS'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    
    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT2GOODS'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_OWN_CAR_AGE2DAYS_BIRTH'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_OWN_CAR_AGE2DAYS_EMPLOYED'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_DAYS_LAST_PHONE_CHANGE2DAYS_BIRTH'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_DAYS_LAST_PHONE_CHANGE2DAYS_EMPLOYED'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    
    return(df)


# What application_train/test look like after aggregation

# In[ ]:


application = get_application_dataset()
plot_df(application)


# Transform APPLICATION

# In[ ]:


def transform_application(df):
    
    logTransformation = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']
    df[logTransformation] = df[logTransformation].apply(lambda x : np.log(x+1),axis=1)
    
    sqrtTransformation = ['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','OWN_CAR_AGE','DAYS_LAST_PHONE_CHANGE']
    df[sqrtTransformation] = df[sqrtTransformation].apply(lambda x: np.sqrt(np.abs(x)),axis=1)

    return(df)


# ***BUREAU & BUREAU_BALANCE***
# 
# All the information about an applicant in the BUREAU is available in this table. It is required to do the aggregation in order to use them as features. There are two type of records. (Closed, Bad debt, Sold) or Active. So we need to aggregate accordinly.
# 
# What does BUREAU look like 

# In[ ]:


bureau = pd.read_csv('../input/bureau.csv')
plot_df(bureau,withTarget=False)


# What does bureau_balance look like

# In[ ]:


bureau_balance = pd.read_csv('../input/bureau_balance.csv')
plot_df(bureau_balance,withTarget=False)


# bureau_balance aggregation

# In[ ]:


def bureau_balance():
    df = pd.read_csv('../input/bureau_balance.csv')
    # getting the furthest date attached to bureau_id
    df1 = df.groupby(['SK_ID_BUREAU']).agg(
            {'MONTHS_BALANCE': min,
            })
    # Status of bureau_id as per freshest month
    df2 = df.groupby(['SK_ID_BUREAU']).agg(
                {'MONTHS_BALANCE': max,
                }).reset_index()
    df2 = pd.merge(df2,df,on=['SK_ID_BUREAU','MONTHS_BALANCE'],how='inner')
    df2 = pd.crosstab(df2['SK_ID_BUREAU'], df2['STATUS'])

    df = pd.merge(df1,df2,on=['SK_ID_BUREAU'],how='left').reset_index()
    df.columns = ['SK_ID_BUREAU','MONTHS_BALANCE','BB_S_0','BB_S_1','BB_S_2','BB_S_3','BB_S_4','BB_S_5','BB_S_C','BB_S_X']
    return(df)


# BUREAU & BUREAU_BALANCE aggregation

# In[ ]:


def get_bureau_dataset():
    b = pd.read_csv('../input/bureau.csv')
    bb = bureau_balance()
    df = pd.merge(b,bb,on='SK_ID_BUREAU',how='left')
    df.loc[(df.DAYS_CREDIT_ENDDATE < 0) | (df.DAYS_CREDIT_ENDDATE > 5000),'DAYS_CREDIT_ENDDATE'] = np.nan
    df.loc[(df.DAYS_ENDDATE_FACT < -5000),'DAYS_ENDDATE_FACT'] = np.nan
    df.loc[(df.AMT_CREDIT_MAX_OVERDUE > 40000),'AMT_CREDIT_MAX_OVERDUE'] = 40000
    df.loc[(df.DAYS_CREDIT_UPDATE < -3000),'DAYS_CREDIT_UPDATE'] = np.nan
    df.loc[(df.AMT_CREDIT_SUM_DEBT < 0),'AMT_CREDIT_SUM_DEBT'] = np.nan
    df.loc[(df.AMT_CREDIT_SUM_LIMIT < 0),'AMT_CREDIT_SUM_LIMIT'] = np.nan

    All = df.groupby(['SK_ID_CURR']).agg(
            {'DAYS_CREDIT': [min, max],
             'CREDIT_DAY_OVERDUE':max,
             'DAYS_CREDIT_ENDDATE':max,
             'DAYS_ENDDATE_FACT':[min,max],
             'AMT_CREDIT_MAX_OVERDUE':max,
             'CNT_CREDIT_PROLONG':max,
             'AMT_CREDIT_SUM':max,
             'AMT_CREDIT_SUM_DEBT':max,
             'AMT_CREDIT_SUM_LIMIT':max,
             'DAYS_CREDIT_UPDATE':min,
             'AMT_ANNUITY':max,
             'MONTHS_BALANCE':min,
             'BB_S_0':sum,
             'BB_S_1':sum,
             'BB_S_2':sum,
             'BB_S_3':sum,
             'BB_S_4':sum,
             'BB_S_5':sum,
             'BB_S_C':sum,
             'BB_S_X':sum
            })
    All.columns = ["_all_".join(x) for x in All.columns.ravel()]
    Active = df.query('CREDIT_ACTIVE == "Active"').groupby(['SK_ID_CURR']).agg(
            {'CREDIT_DAY_OVERDUE':max,
             'AMT_CREDIT_MAX_OVERDUE': max,
             'CNT_CREDIT_PROLONG':[max,sum],
             'AMT_CREDIT_SUM':sum,
             'AMT_CREDIT_SUM_DEBT':sum,
             'AMT_CREDIT_SUM_LIMIT':sum,
             'AMT_CREDIT_SUM_OVERDUE':sum,
             'DAYS_CREDIT_UPDATE':min,
             'AMT_ANNUITY':sum,
             'MONTHS_BALANCE':min,
             'BB_S_0':sum,
             'BB_S_1':sum,
             'BB_S_2':sum,
             'BB_S_3':sum,
             'BB_S_4':sum,
             'BB_S_5':sum,
             'BB_S_C':sum,
             'BB_S_X':sum
            })
    Active.columns = ["_act_".join(x) for x in Active.columns.ravel()]
    
    CREDIT_ACTIVE_ctab = pd.crosstab(df['SK_ID_CURR'], df['CREDIT_ACTIVE']).rename_axis(None, axis=1)
    from functools import reduce
    dfs = [All,Active,CREDIT_ACTIVE_ctab]

    df_final = reduce(lambda left,right: pd.merge(left,right,on='SK_ID_CURR',how='outer'), dfs)
    df_final.reset_index(inplace=True)
    return(df_final)


# BUREAU after aggregation

# In[ ]:


bureau = get_bureau_dataset()
bureau = application.loc[:,['SK_ID_CURR','isTrain','isTest','TARGET']].merge(bureau,how='left',on='SK_ID_CURR')
plot_df(bureau,withTarget=True)


# In[ ]:


bureau.head()


#  Transfrom bureau

# In[ ]:


def bureau_newFeature(df):
    df['AMT_CREDIT_SUM_sum2AMT_CREDIT_SUM_DEBT_sum'] = df['AMT_CREDIT_SUM_DEBT_act_sum']/df['AMT_CREDIT_SUM_act_sum']
    df['AMT_CREDIT_SUM_sum2AMT_ANNUITY_sum'] = df['AMT_CREDIT_SUM_act_sum']/df['AMT_ANNUITY_act_sum']
    df['AMT_CREDIT_SUM_DEBT_sum2AMT_ANNUITY_sum'] = df['AMT_CREDIT_SUM_DEBT_act_sum']/df['AMT_ANNUITY_act_sum']
    df.replace([np.inf, -np.inf], np.nan,inplace=True)
    df.loc[df.AMT_CREDIT_SUM_sum2AMT_CREDIT_SUM_DEBT_sum>2,'AMT_CREDIT_SUM_sum2AMT_CREDIT_SUM_DEBT_sum'] = np.nan
    df.loc[df.AMT_CREDIT_SUM_sum2AMT_ANNUITY_sum>120,'AMT_CREDIT_SUM_sum2AMT_ANNUITY_sum'] = np.nan
    df.loc[df.AMT_CREDIT_SUM_DEBT_sum2AMT_ANNUITY_sum>80,'AMT_CREDIT_SUM_DEBT_sum2AMT_ANNUITY_sum'] = np.nan
    return(df)
    
def transform_bureau(df):
    logTransformation = ['CREDIT_DAY_OVERDUE_all_max','AMT_CREDIT_MAX_OVERDUE_all_max','AMT_CREDIT_SUM_all_max','AMT_CREDIT_SUM_DEBT_all_max',
                         'AMT_CREDIT_SUM_LIMIT_all_max','AMT_ANNUITY_all_max','AMT_CREDIT_MAX_OVERDUE_act_max','AMT_CREDIT_SUM_act_sum',
                         'AMT_CREDIT_SUM_DEBT_act_sum','AMT_CREDIT_SUM_LIMIT_act_sum','AMT_CREDIT_SUM_OVERDUE_act_sum','AMT_ANNUITY_act_sum']
    df[logTransformation] = df[logTransformation].apply(lambda x : np.log(x+1),axis=1)
    
    sartLogTransformation = ['CREDIT_DAY_OVERDUE_act_max','DAYS_CREDIT_UPDATE_act_min']
    df[logTransformation] = df[logTransformation].apply(lambda x : np.sqrt(np.log(np.abs(x+1))),axis=1)
    
    sqrtTransformation = ['DAYS_CREDIT_all_min','DAYS_CREDIT_all_max','DAYS_CREDIT_ENDDATE_all_max','DAYS_ENDDATE_FACT_all_min','DAYS_ENDDATE_FACT_all_max',
                         'DAYS_CREDIT_UPDATE_all_min','MONTHS_BALANCE_all_min','MONTHS_BALANCE_act_min']
    df[sqrtTransformation] = df[sqrtTransformation].apply(lambda x: np.sqrt(np.abs(x)),axis=1)

    return(df)


# In[ ]:


df = get_bureau_dataset()
df = bureau_newFeature(df)
df = transform_bureau(df)


# ***PREVIOUS_APPLICATION***

# PREVIOUS_APPLICATION before aggregation

# In[ ]:


previous_application = pd.read_csv('../input/previous_application.csv')
plot_df(previous_application,False)


# PREVIOUS_APPLICATION aggregation

# In[ ]:


def get_previous_application():
    df = pd.read_csv('../input/previous_application.csv')
    df.loc[df.DAYS_FIRST_DRAWING >0,'DAYS_FIRST_DRAWING'] = np.nan
    df.loc[df.DAYS_FIRST_DUE >0,'DAYS_FIRST_DUE'] = np.nan
    df.loc[df.DAYS_LAST_DUE_1ST_VERSION >2000,'DAYS_LAST_DUE_1ST_VERSION'] = np.nan
    df.loc[df.DAYS_LAST_DUE >3000,'DAYS_LAST_DUE'] = np.nan
    df.loc[df.DAYS_TERMINATION >3000,'DAYS_TERMINATION'] = np.nan

    NAME_CONTRACT_STATUS_ctab = pd.crosstab(df['SK_ID_CURR'], df['NAME_CONTRACT_STATUS'])
    df_grouped = df.query('NAME_CONTRACT_STATUS != "Refused" and FLAG_LAST_APPL_PER_CONTRACT == "Y" and NFLAG_LAST_APPL_IN_DAY == 1')                                                    .groupby(['SK_ID_CURR'])                                                    .agg(
                                                        {'AMT_ANNUITY':max,
                                                         'AMT_APPLICATION':max,
                                                         'AMT_CREDIT':max,
                                                         'AMT_DOWN_PAYMENT':max,
                                                         'AMT_GOODS_PRICE':max,
                                                         'RATE_DOWN_PAYMENT':[min, max],
                                                         'RATE_INTEREST_PRIMARY':[min, max],
                                                         'RATE_INTEREST_PRIVILEGED':[min, max],
                                                         'DAYS_DECISION':[min, max],
                                                         'CNT_PAYMENT':[min, max],
                                                         'DAYS_FIRST_DRAWING':min,
                                                         'DAYS_FIRST_DUE':[min, max],
                                                         'DAYS_LAST_DUE_1ST_VERSION':[min, max],
                                                         'DAYS_LAST_DUE':[min, max],
                                                         'DAYS_TERMINATION':[min, max],
                                                         'NFLAG_INSURED_ON_APPROVAL':sum
                                                        })
    df_final = pd.merge(df_grouped,NAME_CONTRACT_STATUS_ctab,on='SK_ID_CURR',how='outer')
    df_final.reset_index(inplace=True)
    df_final.columns = ['SK_ID_CURR','AMT_ANNUITY_max','AMT_APPLICATION_max','AMT_CREDIT_max','AMT_DOWN_PAYMENT_max','AMT_GOODS_PRICE_max',
                        'RATE_DOWN_PAYMENT_min','RATE_DOWN_PAYMENT_max','RATE_INTEREST_PRIMARY_min','RATE_INTEREST_PRIMARY_max',
                        'RATE_INTEREST_PRIVILEGED_min','RATE_INTEREST_PRIVILEGED_max','DAYS_DECISION_min','DAYS_DECISION_max','CNT_PAYMENT_min',
                        'CNT_PAYMENT_max','DAYS_FIRST_DRAWING_min','DAYS_FIRST_DUE_min','DAYS_FIRST_DUE_max',
                        'DAYS_LAST_DUE_1ST_VERSION_min','DAYS_LAST_DUE_1ST_VERSION_max','DAYS_LAST_DUE_min','DAYS_LAST_DUE_max','DAYS_TERMINATION_min',
                        'DAYS_TERMINATION_max','NFLAG_INSURED_ON_APPROVAL_sum','Approved','Canceled','Refused','Unused_offer']
    df_final.head()
    return(df_final)


# PREVIOUS_APPLICATION after aggregation

# In[ ]:


previous_application = get_previous_application()
previous_application = application.loc[:,['SK_ID_CURR','isTrain','isTest','TARGET']].merge(previous_application,how='left',on='SK_ID_CURR')
plot_df(previous_application,withTarget=True)


# PREVIOUS_APPLICATION transfromation

# In[ ]:


def transform_previous_application(df):
    logTransformation = ['AMT_ANNUITY_max','AMT_APPLICATION_max','AMT_CREDIT_max', 'AMT_DOWN_PAYMENT_max','AMT_GOODS_PRICE_max']
    df[logTransformation] = df[logTransformation].apply(lambda x : np.log(x+1),axis=1)
    
    sqrtTransformation = ['DAYS_DECISION_min','DAYS_DECISION_max','DAYS_FIRST_DRAWING_min','DAYS_FIRST_DUE_min','DAYS_FIRST_DUE_max','DAYS_LAST_DUE_min',
                          'DAYS_LAST_DUE_max','DAYS_TERMINATION_min','DAYS_TERMINATION_max']
    df[sqrtTransformation] = df[sqrtTransformation].apply(lambda x: np.sqrt(np.abs(x)),axis=1)
    return(df)


# ***POS_CASH_balance***

# POS_CASH_balance before aggregation

# In[ ]:


POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
plot_df(POS_CASH_balance,False)


# POS_CASH_balance  aggregation

# In[ ]:


def get_POS_CASH_balance():
    POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
    Closed_Loans = POS_CASH_balance[POS_CASH_balance['SK_ID_PREV'].isin(POS_CASH_balance.query('NAME_CONTRACT_STATUS == "Completed"').SK_ID_PREV)]
    Active_Loans = POS_CASH_balance[~POS_CASH_balance['SK_ID_PREV'].isin(POS_CASH_balance.query('NAME_CONTRACT_STATUS == "Active" and MONTHS_BALANCE == -1').SK_ID_PREV)]

    Active = Active_Loans.groupby(['SK_ID_CURR']).agg(
                    {  'MONTHS_BALANCE':min,
                       'CNT_INSTALMENT':[min,max],
                       'CNT_INSTALMENT_FUTURE':[min,max]
                    })
    Closed = Closed_Loans.groupby(['SK_ID_CURR']).agg(
                    {  'MONTHS_BALANCE':[min,max],
                       'CNT_INSTALMENT':max
                    })
    NAME_CONTRACT_STATUS = POS_CASH_balance.query('(NAME_CONTRACT_STATUS == "Completed") or (NAME_CONTRACT_STATUS == "Active" and MONTHS_BALANCE == -1) ')[['SK_ID_PREV','SK_ID_CURR','NAME_CONTRACT_STATUS']].drop_duplicates()
    NAME_CONTRACT_STATUS_ctab = pd.crosstab(NAME_CONTRACT_STATUS['SK_ID_CURR'], NAME_CONTRACT_STATUS['NAME_CONTRACT_STATUS'])

    from functools import reduce
    dfs = [NAME_CONTRACT_STATUS_ctab,Active,Closed]
    df_final = reduce(lambda left,right: pd.merge(left,right,on='SK_ID_CURR',how='outer'), dfs)
    df_final.reset_index(inplace=True)
    df_final.columns = ['SK_ID_CURR','Active','Completed','MONTHS_BALANCE_A_min','CNT_INSTALMENT_A_min','CNT_INSTALMENT_A_max','CNT_INSTALMENT_FUTURE_A_min',
                        'CNT_INSTALMENT_FUTURE_max','MONTHS_BALANCE_C_min','MONTHS_BALANCE_C_max','CNT_INSTALMENT_C_max']
    return(df_final)


# POS_CASH_balance after aggregation

# In[ ]:


POS_CASH_balance = get_POS_CASH_balance()
POS_CASH_balance = application.loc[:,['SK_ID_CURR','isTrain','isTest','TARGET']].merge(POS_CASH_balance,how='left',on='SK_ID_CURR')
plot_df(POS_CASH_balance,withTarget=True)


# POS_CASH_balance transformation

# In[ ]:


def transform_POS_CASH_balance(df):
    sqrtTransformation = ['CNT_INSTALMENT_A_min','CNT_INSTALMENT_A_max','CNT_INSTALMENT_FUTURE_A_min','CNT_INSTALMENT_FUTURE_max','CNT_INSTALMENT_C_max']
    df[sqrtTransformation] = df[sqrtTransformation].apply(lambda x: np.sqrt(np.abs(x)),axis=1)
    return(df)


# ***INSTALLMETS_PAYMENTS***

# INSTALLMETS_PAYMENTS before aggregation

# In[ ]:


instalment_payments = pd.read_csv('../input/installments_payments.csv')
plot_df(instalment_payments,withTarget=False)


# INSTALLMETS_PAYMENTS aggregation

# In[ ]:


def get_installment_payments():
    instalment_payments = pd.read_csv('../input/installments_payments.csv')
    instalment_payments['MONTH']=(instalment_payments['DAYS_INSTALMENT']/30).astype(int)
    # features for last month active loans
    Active = instalment_payments.query('MONTH == -1').groupby('SK_ID_CURR').agg({
        'NUM_INSTALMENT_VERSION':max,
        'NUM_INSTALMENT_NUMBER':max,
        'AMT_INSTALMENT':sum,
        'AMT_PAYMENT':sum
    })
    Closed = instalment_payments.groupby('SK_ID_CURR').agg({
        'NUM_INSTALMENT_VERSION':max,
        'NUM_INSTALMENT_NUMBER':max,
        'DAYS_INSTALMENT':min,
        'AMT_INSTALMENT':[max,min]
    })
    from functools import reduce
    df_final = pd.merge(Active,Closed,on='SK_ID_CURR',how='outer')
    df_final.reset_index(inplace=True)
    df_final.columns=['SK_ID_CURR','NUM_INSTALMENT_VERSION_A_max','NUM_INSTALMENT_NUMBER_A_max','AMT_INSTALMENT_A_sum','AMT_PAYMENT_A_sum',
                      'NUM_INSTALMENT_VERSION_C_max','NUM_INSTALMENT_NUMBER_C_max','DAYS_INSTALMENT_C_min','AMT_INSTALMENT_C_max','AMT_INSTALMENT_c_min']
    return(df_final)


# INSTALLMETS_PAYMENTS after aggregation

# In[ ]:


instalment_payments = get_installment_payments()
instalment_payments = application.loc[:,['SK_ID_CURR','isTrain','isTest','TARGET']].merge(instalment_payments,how='left',on='SK_ID_CURR')
plot_df(instalment_payments)


# INSTALLMETS_PAYMENTS transformation

# In[ ]:


def transform_installment_payments(df):
    logTransformation = ['AMT_INSTALMENT_A_sum','AMT_PAYMENT_A_sum','AMT_INSTALMENT_C_max','AMT_INSTALMENT_c_min']
    df[logTransformation] = df[logTransformation].apply(lambda x : np.log(x+1),axis=1)
    
    sqrtTransformation = ['NUM_INSTALMENT_VERSION_A_max','NUM_INSTALMENT_NUMBER_A_max','NUM_INSTALMENT_VERSION_C_max','NUM_INSTALMENT_NUMBER_C_max',
                          'DAYS_INSTALMENT_C_min']
    df[sqrtTransformation] = df[sqrtTransformation].apply(lambda x: np.sqrt(np.abs(x)),axis=1)
    return(df)


# ***CREDIT_CARD_BALANCE***

# credit_card_balance before aggregation

# In[ ]:


credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
plot_df(credit_card_balance,withTarget=False)


# CREDIT_CARD_BALANCE aggregation

# In[ ]:


def get_credit_card_balance():
    df = pd.read_csv('../input/credit_card_balance.csv')
    df.loc[df.AMT_BALANCE <0,'AMT_BALANCE'] = np.nan
    df.loc[df.AMT_RECEIVABLE_PRINCIPAL <0,'AMT_RECEIVABLE_PRINCIPAL'] = np.nan
    df.loc[df.AMT_RECIVABLE <0,'AMT_RECIVABLE'] = np.nan
    df.loc[df.AMT_TOTAL_RECEIVABLE <0,'AMT_TOTAL_RECEIVABLE'] = np.nan
    # features for last month active loans
    last_month = df.query('MONTHS_BALANCE == -1').groupby('SK_ID_CURR').agg({
        'AMT_BALANCE':lambda x: sum(x[x>0]),
        'AMT_CREDIT_LIMIT_ACTUAL':sum,
        'AMT_DRAWINGS_ATM_CURRENT':sum,
        'AMT_DRAWINGS_CURRENT':sum,
        'AMT_DRAWINGS_OTHER_CURRENT':sum,
        'AMT_DRAWINGS_POS_CURRENT':sum,
        'AMT_INST_MIN_REGULARITY':sum,
        'AMT_PAYMENT_CURRENT':sum,
        'AMT_PAYMENT_TOTAL_CURRENT':sum,
        'AMT_RECEIVABLE_PRINCIPAL':lambda x: sum(x[x>0]),
        'AMT_RECIVABLE':lambda x: sum(x[x>0]),
        'AMT_TOTAL_RECEIVABLE':lambda x: sum(x[x>0]),
        'CNT_DRAWINGS_ATM_CURRENT':sum,
        'CNT_DRAWINGS_CURRENT':sum,
        'CNT_DRAWINGS_POS_CURRENT':sum,
        'CNT_DRAWINGS_OTHER_CURRENT':sum,
        'CNT_INSTALMENT_MATURE_CUM':sum,
    })
    all_month = df.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE':[max,min],
        'AMT_BALANCE':max,
        'AMT_CREDIT_LIMIT_ACTUAL':[max,min],
        'AMT_DRAWINGS_ATM_CURRENT':max,
        'AMT_DRAWINGS_CURRENT':max,
        'AMT_DRAWINGS_OTHER_CURRENT':max,
        'AMT_DRAWINGS_POS_CURRENT':max,
        'AMT_INST_MIN_REGULARITY':max,
        'AMT_PAYMENT_CURRENT':max,
        'AMT_PAYMENT_TOTAL_CURRENT':max,
        'AMT_RECEIVABLE_PRINCIPAL':max,
        'AMT_RECIVABLE':max,
        'AMT_TOTAL_RECEIVABLE':max,
        'CNT_DRAWINGS_ATM_CURRENT':max,
        'CNT_DRAWINGS_CURRENT':max,
        'CNT_DRAWINGS_POS_CURRENT':max,
        'CNT_DRAWINGS_OTHER_CURRENT':max,
        'CNT_INSTALMENT_MATURE_CUM':max,
        'SK_DPD':max,
        'SK_DPD_DEF':max,
    })
    
    #from functools import reduce
    df_final = pd.merge(last_month,all_month,on='SK_ID_CURR',how='outer')
    df_final.columns=['AMT_BALANCE_P_sum',
        'AMT_CREDIT_LIMIT_ACTUAL_p_sum',
        'AMT_DRAWINGS_ATM_CURRENT_p_sum',
        'AMT_DRAWINGS_CURRENT_p_sum',
        'AMT_DRAWINGS_OTHER_CURRENT_p_sum',
        'AMT_DRAWINGS_POS_CURRENT_p_sum',
        'AMT_INST_MIN_REGULARITY_p_sum',
        'AMT_PAYMENT_CURRENT_p_sum',
        'AMT_PAYMENT_TOTAL_CURRENT_p_sum',
        'AMT_RECEIVABLE_PRINCIPAL_p_sum',
        'AMT_RECIVABLE_p_sum',
        'AMT_TOTAL_RECEIVABLE_p_sum',
        'CNT_DRAWINGS_ATM_CURRENT_p_sum',
        'CNT_DRAWINGS_CURRENT_p_sum',
        'CNT_DRAWINGS_POS_CURRENT_p_sum',
        'CNT_DRAWINGS_OTHER_CURRENT_p_sum',
        'CNT_INSTALMENT_MATURE_CUM_p_sum',
        'MONTHS_BALANCE_A_max',
        'MONTHS_BALANCE_A_min',
        'AMT_BALANCE_A_max',
        'AMT_CREDIT_LIMIT_ACTUAL_A_max',
        'AMT_CREDIT_LIMIT_ACTUAL_A_min',
        'AMT_DRAWINGS_ATM_CURRENT_A_max',
        'AMT_DRAWINGS_CURRENT_A_max',
        'AMT_DRAWINGS_OTHER_CURRENT_A_max',
        'AMT_DRAWINGS_POS_CURRENT_A_max',
        'AMT_INST_MIN_REGULARITY_A_max',
        'AMT_PAYMENT_CURRENT_A_max',
        'AMT_PAYMENT_TOTAL_CURRENT_A_max',
        'AMT_RECEIVABLE_PRINCIPAL_A_max',
        'AMT_RECIVABLE_A_max',
        'AMT_TOTAL_RECEIVABLE_A_max',
        'CNT_DRAWINGS_ATM_CURRENT_A_max',
        'CNT_DRAWINGS_CURRENT_A_max',
        'CNT_DRAWINGS_POS_CURRENT_A_max',
        'CNT_DRAWINGS_OTHER_CURRENT_A_max',
        'CNT_INSTALMENT_MATURE_CUM_A_max',
        'SK_DPD_A_max',
        'SK_DPD_DEF_A_max',]
    df_final.reset_index(inplace=True)
    return(df_final)


# CREDIT_CARD_BALANCE aggregation

# In[ ]:


credit_card_balance = get_credit_card_balance()
credit_card_balance = application.loc[:,['SK_ID_CURR','isTrain','isTest','TARGET']].merge(credit_card_balance,how='left',on='SK_ID_CURR')
plot_df(credit_card_balance)


# CREDIT_CARD_BALANCE transformation

# In[ ]:


def transform_credit_card_balance(df):
    logTransformation = ['AMT_BALANCE_P_sum','AMT_CREDIT_LIMIT_ACTUAL_p_sum','AMT_DRAWINGS_ATM_CURRENT_p_sum','AMT_DRAWINGS_CURRENT_p_sum',
                         'AMT_DRAWINGS_OTHER_CURRENT_p_sum','AMT_DRAWINGS_POS_CURRENT_p_sum','AMT_INST_MIN_REGULARITY_p_sum','AMT_PAYMENT_CURRENT_p_sum',
                         'AMT_PAYMENT_TOTAL_CURRENT_p_sum','AMT_RECEIVABLE_PRINCIPAL_p_sum','AMT_RECIVABLE_p_sum','AMT_TOTAL_RECEIVABLE_p_sum',
                         'CNT_DRAWINGS_POS_CURRENT_p_sum','AMT_BALANCE_A_max','AMT_CREDIT_LIMIT_ACTUAL_A_max','AMT_CREDIT_LIMIT_ACTUAL_A_min',
                         'AMT_DRAWINGS_ATM_CURRENT_A_max','AMT_DRAWINGS_CURRENT_A_max','AMT_DRAWINGS_OTHER_CURRENT_A_max','AMT_DRAWINGS_POS_CURRENT_A_max',
                         'AMT_INST_MIN_REGULARITY_A_max','AMT_PAYMENT_CURRENT_A_max','AMT_PAYMENT_TOTAL_CURRENT_A_max','AMT_RECEIVABLE_PRINCIPAL_A_max',
                         'AMT_RECIVABLE_A_max','SK_DPD_A_max','SK_DPD_DEF_A_max']
    df[logTransformation] = df[logTransformation].apply(lambda x : np.log(x+1),axis=1)
    
    sqrtTransformation = ['CNT_DRAWINGS_CURRENT_p_sum','CNT_INSTALMENT_MATURE_CUM_p_sum','CNT_DRAWINGS_ATM_CURRENT_A_max',
                          'CNT_DRAWINGS_CURRENT_A_max','CNT_DRAWINGS_POS_CURRENT_A_max','CNT_INSTALMENT_MATURE_CUM_A_max']
    df[sqrtTransformation] = df[sqrtTransformation].apply(lambda x: np.sqrt(np.abs(x)),axis=1)
    return(df)


# **Final DF**

# In[ ]:


#def getFinalDataSet():
#application = transform_application(get_application_dataset())
#bureau = transform_bureau(get_bureau_dataset())
#previous_application = transform_previous_application(get_previous_application())
#POS_CASH_balance = transform_POS_CASH_balance(get_POS_CASH_balance())
#installment_payments = transform_installment_payments(get_installment_payments()) 
#credit_card_balance = transform_credit_card_balance(get_credit_card_balance())
#dfs = [application, bureau, previous_application, POS_CASH_balance, installment_payments, credit_card_balance]
#from functools import reduce
#df = reduce(lambda left,right: pd.merge(left,right,on='SK_ID_CURR',how='left'), dfs)

#feature = ['AMT_ANNUITY']
#plotList = ['isTrain','isTest','TARGET']+feature
#df = bureau.copy()
#df[feature] = np.log(df[feature]+1)
#df[feature] = np.log(np.abs(df[feature])+1)
#df[feature] = np.sqrt(np.log(np.abs(df[feature])+1))
#df[feature] = np.sqrt(np.abs(df[feature]))
#plot_df(df.loc[:,plotList])

#df = bureau.copy()
#feature = ['AMT_CREDIT_SUM_DEBT_sum2AMT_ANNUITY_sum']
#df['AMT_CREDIT_SUM_DEBT_sum2AMT_ANNUITY_sum'] = df['AMT_CREDIT_SUM_DEBT_sum']/df['AMT_ANNUITY_sum']
#df.replace([np.inf, -np.inf], np.nan,inplace=True)
#plotList = ['isTrain','isTest','TARGET']+feature
#df[feature] = np.log(df[feature]+1)
#df[feature] = np.log(np.abs(df[feature])+1)
#df[feature] = np.sqrt(np.log(np.abs(df[feature])+1))
#df[feature] = np.sqrt(np.abs(df[feature]))
#plot_df(df.loc[:,plotList])

