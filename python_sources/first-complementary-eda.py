#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)




pd.set_option("display.max_columns",100)
pd.set_option('display.max_colwidth', -1)


# In[ ]:


print('Importing data...')
data = {
    'train': pd.read_csv('../input/application_train.csv'),
    'test': pd.read_csv('../input/application_test.csv'),
    'bb': pd.read_csv('../input/bureau_balance.csv'),
    'b': pd.read_csv('../input/bureau.csv'),
    'ccb': pd.read_csv('../input/credit_card_balance.csv'),
    'ip': pd.read_csv('../input/installments_payments.csv'),
    'POSb': pd.read_csv('../input/POS_CASH_balance.csv'),
    'previous': pd.read_csv('../input/previous_application.csv')
    }


# In[ ]:


def cat_features(df):
    cat_f = df.select_dtypes(include = ['object']).apply(lambda x: x.nunique(dropna=False), axis = 0)
    return cat_f

def cat_levels(df,cat_info):
    cat_f = df[cat_info.index]
    levels = {}
    for c in cat_f:
        level = list(df[c].replace(np.nan,'NaN').unique())
        levels.update({c:level})
    return pd.DataFrame.from_dict(levels,orient='index').fillna('')

def bin_num(df):
    binary_f = []
    for c in df.columns:
        if len(df[c].unique())==2 and c not in cat_features(df).index:
            binary_f.append(c)
    return binary_f

def cat_plot(df,cols,r,c,figsize):
    fig, ax = plt.subplots(r,c,figsize=figsize)
    for i in range(len(cols)):
        colname = cols[i]
        row = i//c
        col = i%c
        axa = sns.countplot(x=colname, data=df,ax = ax[row,col])
        plt.setp(axa.xaxis.get_majorticklabels(), rotation=-45)
        plt.tight_layout()
    plt.show()

def cat_plot_target(df,cols,r,c,figsize):
    plt.figure(figsize=figsize)
    for i in range(len(cols)):
        colname = cols[i]
        df_plot = df[['TARGET',colname]].dropna().melt(['TARGET'],value_name=colname)
        df_group = df_plot.groupby(['TARGET',colname],as_index = False).count()
        sums = df_group.groupby('TARGET',as_index=False).sum()['variable']
        df_group['sum'] = df_group['TARGET'].apply(lambda x: sums[0] if x==0 else sums[1])
        df_group['percent'] = df_group['variable']/df_group['sum']
        plt.subplot(r,c,i+1)
        axa = sns.barplot(x = colname, y = 'percent', hue = 'TARGET',data = df_group)
        plt.setp(axa.xaxis.get_majorticklabels(), rotation=-45)
    plt.tight_layout()
    plt.show()

def num_abberrant(df):
    num_df = df.describe()
    abb = []
    for n in num_df.columns:
        low = num_df[n]['mean']-3*num_df[n]['std']
        up = num_df[n]['mean']+3*num_df[n]['std']
        if num_df[n]['min'] < low or num_df[n]['max'] > up:
            abb.append(n)
    return num_df[abb]

# all values in normalized data should be in [0,1]
def norm_abb(df):
    norm_abb = []
    for c in df.columns:
        if df[c].dropna().between(0,1).all() == False:
            print('Values of column ' + c + ' not in range [0,1].')
            norm_abb.append(c)
    return norm_abb

# all values in 'time only relative to the application' data should be negative
def time_abb(df):
    time_abb = []
    for c in df.columns:
        if df[c].dropna().le(0).all() == False:
            print('Values of column ' + c + ' has value greater than 0.')
            time_abb.append(c)
    return time_abb

# all values in 'rounded' data should be integer
def round_abb(df):
    round_abb = []
    for c in df.columns:
        if df[c].dropna().dtype != int:
            print('Values of column ' + c + ' has value greater than 0.')
            round_abb.append(c)
    return round_abb

def sns_distplot(df,cols,r,c):
    plt.figure(figsize = (24,12))
    for i in range(len(cols)):
        plt.subplot(r,c,i+1)
        sns.distplot(df[cols[i]].dropna())
    plt.tight_layout()
    plt.show()

def sns_distplot_target(df,cols,r,c,figsize):
    plt.figure(figsize = figsize)
    for i in range(len(cols)):
        plt.subplot(r,c,i+1)
        df_plot = df[['TARGET',cols[i]]].melt(['TARGET'],value_name=cols[i])
        sns.distplot(df_plot[df_plot['TARGET']==0][cols[i]].dropna())
        sns.distplot(df_plot[df_plot['TARGET']==1][cols[i]].dropna())
    plt.tight_layout()
    plt.show()

def NA_finder(df):
    NA_f = df.isnull().sum() 
    NA_f = NA_f[NA_f != 0].sort_values(ascending=False)
    NA_f_percent = NA_f.sort_values(ascending=False)/len(df)*100.0
    plt.figure(figsize=(20,20))
    NA_f_percent.plot.bar()
    plt.title('NA percentage distribution for NA containing features.')
    plt.ylabel('Percentage (%)')
    plt.show()
    return NA_f_percent

def cat_plot_bureau(df,cols,r,c,figsize):
    plt.figure(figsize=figsize)
    for i in range(len(cols)):
        colname = cols[i]
        df_plot = df[['CREDIT_ACTIVE',colname]].dropna().melt(['CREDIT_ACTIVE'],value_name=colname)
        df_group = df_plot.groupby(['CREDIT_ACTIVE',colname],as_index = False).count()
        df_group['counts'] = df_group['variable']
        plt.subplot(r,c,i+1)
        axa = sns.barplot(x = colname, y = 'counts', hue = 'CREDIT_ACTIVE',data = df_group)
        plt.setp(axa.xaxis.get_majorticklabels(), rotation=-45)
    plt.tight_layout()
    plt.show()
    
def sns_distplot_bureau(df,cols,r,c,figsize):
    plt.figure(figsize = figsize)
    for i in range(len(cols)):
        plt.subplot(r,c,i+1)
        df_plot = df[['CREDIT_ACTIVE',cols[i]]].melt(['CREDIT_ACTIVE'],value_name=cols[i])
        sns.distplot(df_plot[df_plot['CREDIT_ACTIVE']=='Active'][cols[i]].dropna())
        sns.distplot(df_plot[df_plot['CREDIT_ACTIVE']=='Closed'][cols[i]].dropna())
    plt.tight_layout()
    plt.show()


# # Training and testing dataset

# To make good assumption about the features and target value, I will use the training set data for correlation analysis.

# In[ ]:


train = data['train']
train.head()


# __1. target distribution__

# In[ ]:


plt.hist(train['TARGET'])
plt.show()


# The label looks highly imbalanced. Use ROC AUC score as performance metric will be helpful for model validation. To perform feature engineering on training and testing data together, I concatened the two datasets.

# In[ ]:


df = pd.concat([data['train'],data['test']])
train_row = data['train'].shape[0]


# __2. categorical features__<br/>
# The nunique function has dropna=True as default, which is not what I want. I want to find the binary features for label encoding, if it contains NAN values, then OneHotEncoder will be more useful here.

# In[ ]:


# summary of categorical features
cats = cat_features(train)
cat_levels = cat_levels(df,cats)
cat_levels


# By examing the categorical features, I found some categorical data use 'XNA' instead of np.nan as NA values. So next I will replace 'XNA' as np.nan values in both train and test set. I also found 'WEEKDAY_APPR_PROCESS_START' is a time feature that may be encoded by LabelEncoder, but I still chosed to use OneHotEncoding to avoid possible missing feature characteristics.

# In[ ]:


# find 'XNA'
df.replace('XNA',np.nan,inplace = True)
train.replace('XNA',np.nan,inplace = True)
# find binary features
bin_cats = cats[cats == 2].index
print('binary categorical features are: ' + str(bin_cats))
bin_nums = bin_num(df)
print('binary numerical features are: ' + str(bin_nums))


# To further investigate the categorical features' effect on target, I plotted the percentage distribution on different TARGET conditions for all categorical features.

# In[ ]:


cat_plot_target(train,cats.index,4,4,(15,15))


# From the above paired plots, I find 'HOUSETYPE_MODE' feature value of 'block of flats' and 'EMERGENCYSTATE_MODE' feature do not affect the TARGET value. So I will clean them from my dataset for modeling.<br/>
# <br/>
# __3. numerical features__<br/>
# I first look at description table and try to group different numerical features.

# In[ ]:


print(df['DAYS_EMPLOYED'][df['DAYS_EMPLOYED']>0].unique())


# In[ ]:


df['DAYS_EMPLOYED'].replace(365243,np.nan,inplace=True)
train['DAYS_EMPLOYED'].replace(365243,np.nan,inplace=True)


# In[ ]:


num_f = [f for f in df.columns if (f not in cats.index)]
num_f.remove('TARGET')
num_f.remove('SK_ID_CURR')
dist_plot_f = [f for f in num_f if df[f].nunique()>50]
print(str(len(dist_plot_f)) + ' numerical features will be compared in the distribution plots.')
sns_distplot_target(train,dist_plot_f,10,5,(24,24))


# In[ ]:


bar_plot_f = [f for f in num_f if f not in dist_plot_f]
print(str(len(bar_plot_f)) + ' numerical features will be compared in the percentage bar plots.')
cat_plot_target(train,bar_plot_f,11,5,(24,48))


# From the distribution and bar plots, I found the following features have almost the same distribution for both target values and will be cleaned in my dataset for modeling:
#             ['COMMONAREA_MEDI','COMMONAREA_MODE','LANDAREA_MEDI','LANDAREA_MODE','LIVINGAPARTMENTS_MEDI',
#               'LIVINGAPARTMENTS_MODE','LIVINGAREA_MEDI','LIVINGAREA_MODE','NONLIVINGAPARTMENTS_MEDI',
#               'NONLIVINGAPARTMENTS_MODE','NONLIVINGAREA_MEDI','NONLIVINGAREA_MODE','YEARSBEGINEXPLUATATION_MEDI',
#                 'YEARSBEGINEXPLUATATION_MODE','YEARSBUILD_MEDI','YEARSBUILD_MODE', 'AMT_REQ_CREDIT_BUREAU_HOUR', 
#               'AMT_REQ_CREDIT_BUREAU_WEEK','DEF_60_CNT_SOCIAL_CIRCLE','ELEVATORS_MODE',
#               'FLAG_CONT_MOBILE', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 
#               'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 
#               'FLAG_DOCUMENT_21', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_9', 
#               'FLAG_EMAIL', 'FLAG_MOBIL', 'FLOORSMAX_MODE','FLOORSMIN_MODE','OBS_60_CNT_SOCIAL_CIRCLE',
#               'LIVE_REGION_NOT_WORK_REGION', 'REG_REGION_NOT_LIVE_REGION']

# In[ ]:


'EMERGENCYSTATE_MODE' in bar_plot_f


# In[ ]:


drop_f = ['NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAPARTMENTS_MODE', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_WEEK','FLAG_CONT_MOBILE', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_9', 'FLAG_EMAIL', 'FLAG_MOBIL', 'LIVE_REGION_NOT_WORK_REGION', 'REG_REGION_NOT_LIVE_REGION']
df = df.drop(drop_f,axis=1)
train = train.drop(drop_f,axis=1)


# I find for some feature contains a lot of NA values, a good way to impute these data is important since the goal for this project is to help people with few credit history.

# __3. NA values__

# In[ ]:


NA_f = NA_finder(df)

######### Summary feature engineering for train/test dataset

def traintest_processing(df):
    df.replace('XNA',np.nan,inplace = True)
    df['DAYS_EMPLOYED'].replace(365243,np.nan,inplace=True)
    bin_f = bin_finder(df)
    for b in bin_f:
        le.fit(df[b])
        df[b] = le.transform(df[b])
    def df['EMERGENCYSTATE_MODE']
    drop_f = ['NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAPARTMENTS_MODE', 'AMT_REQ_CREDIT_BUREAU_HOUR', 
              'AMT_REQ_CREDIT_BUREAU_WEEK','FLAG_CONT_MOBILE', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 
              'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 
              'FLAG_DOCUMENT_21', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_9', 
              'FLAG_EMAIL', 'FLAG_MOBIL', 'LIVE_REGION_NOT_WORK_REGION', 'REG_REGION_NOT_LIVE_REGION']
    df = df.drop(drop_f,axis=1)
    df_dum = pd.get_dummies(df)
    ### new features learnt from other kernels
    df_dum['DAYS_EMPLOYED_PERC'] = df_dum['DAYS_EMPLOYED'] / df_dum['DAYS_BIRTH']
    df_dum['INCOME_CREDIT_PERC'] = df_dum['AMT_INCOME_TOTAL'] / df_dum['AMT_CREDIT']
    df_dum['INCOME_PER_PERSON'] = df_dum['AMT_INCOME_TOTAL'] / df_dum['CNT_FAM_MEMBERS']
    df_dum['ANNUITY_INCOME_PERC'] = df_dum['AMT_ANNUITY'] / df_dum['AMT_INCOME_TOTAL']
    del df_dum['HOUSETYPE_MODE_block of flats']
    return df_dum
# # Bureau balance data

# In[ ]:


bb = data['bb']
bb.head()


# The bureau balance data contains one numerical data and one categorical data, and this two data seems to be related. According to the description table, 'MONTHS_BALANCE' == -1 means the freshest balance date.

# In[ ]:


bb['STATUS'].unique()


# In[ ]:


bb_fresh = bb[bb['MONTHS_BALANCE'] == -1]
sns.countplot(x="STATUS", data=bb_fresh)


# In[ ]:


# length of balance history is a good feature
# scoring of previous status is also a good one
count = bb[['SK_ID_BUREAU','MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count().rename(columns = {'MONTHS_BALANCE':'HIST_LEN'}).reset_index()
status = bb[bb['MONTHS_BALANCE'] == -1][['SK_ID_BUREAU','STATUS']]
bb_join = pd.merge(count,status,on='SK_ID_BUREAU')
bb_dum = pd.get_dummies(bb)
bb_dum_sum = bb_dum.groupby('SK_ID_BUREAU',as_index=False).sum()
weights = np.array([1,2,3,4,5])
# score1 is the weighted sum of DPD status
bb_dum_sum['BUREAU_SCORE1'] = (bb_dum_sum.iloc[:,2:7]*weights).sum(axis=1)
bb_score = pd.merge(bb_dum_sum[['SK_ID_BUREAU','BUREAU_SCORE1']], bb_join, on='SK_ID_BUREAU')
# score2 is the ratio of score1 and history length
bb_score['BUREAU_SCORE2'] = bb_score['BUREAU_SCORE1']/bb_score['HIST_LEN']
# score3 is the reward score of not have any DPD
bb_score['BUREAU_SCORE3'] = bb_score['BUREAU_SCORE2'].apply(lambda x: 1 if x==0 else 0)
bb_score.head()


# # Bureau data

# __1. Overview__ <br/>
# Bureau balance and bureau data are connected tightly, I will merge the two tables together to perform EDA. Before merging, I found like bureau balance data, bureau data has a column showing the status of account, I want to see if they show the same information.

# In[ ]:


b = data['b']
credit_status_check = pd.merge(bb_fresh,b[['SK_ID_BUREAU','CREDIT_ACTIVE']],on = 'SK_ID_BUREAU',how = 'inner')
credit_status_check.head()


# So these two columns are actually different information, I cannot combine the two information together.

# In[ ]:


bureau = pd.merge(b,bb_score,on='SK_ID_BUREAU')
bureau.head()


# __2. Categorical variables__

# In[ ]:


bureau_cats = cat_features(bureau)
bureau_cats


# __3. data split into two sets based on 'CREDIT_ACTIVE' feature__<br/>
# The 'CREDIT_ACTIVE' is an important feature when we consider historical data. The number of bureau records that are still active or already closed can affect other features as well. So I want to examine if 'CREDIT_ACTIVE' condition can affect a lot of other feature distributions.

# In[ ]:


cat_plot_bureau(bureau,bureau_cats.index.drop('CREDIT_ACTIVE'),1,3,(15,5))


# The categroical features countplots look very different for active and closed accounts. So I think it's necessary to analyze them seperately when apply feature enigneering.

# In[ ]:


bureau_active = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].drop('CREDIT_ACTIVE',axis=1)
bureau_closed = bureau[bureau['CREDIT_ACTIVE'] == 'Closed'].drop('CREDIT_ACTIVE',axis=1)


# __4. Single value column exploration__<br/>
# The 'DAYS_ENDDATE_FACT' feature only has value in closed bureau credit account (in feature description). I need to find features like this and delete them.

# In[ ]:


bureau_active = bureau_active.drop([col for col in bureau_active.columns 
                                    if bureau_active[col].nunique(dropna=False) == 1],axis=1)
print(str(bureau.shape[1]-1-bureau_active.shape[1]) + ' columns dropped due to columns contain only 1 value.')
bureau_closed = bureau_closed.drop([col for col in bureau_closed.columns 
                                    if bureau_closed[col].nunique(dropna=False) == 1],axis=1)
print(str(bureau.shape[1]-1-bureau_closed.shape[1]) + ' columns dropped due to columns contain only 1 value.')


# Unfortunately, the 'DAYS_ENDDATE_FACT' column is not found by nunique method. I have to mannually remove this column. In case of the samething happen, I first looked at the NA value distribution in all variable.

# In[ ]:


bureau_active.isnull().sum()/len(bureau_active)


# In[ ]:


bureau_closed.isnull().sum()/len(bureau_closed)


# From the above information, I find only 'DAYS_ENDDATE_FACT' should be removed from bureau_active.

# In[ ]:


bureau_active.drop('DAYS_ENDDATE_FACT',axis=1,inplace=True)


# __5. 'ground truth' validation__<br/>
# Like when exploring the training and test set data, I will check for some ground truth restrictions.

# The 'DAYS_CREDIT_ENDDATE' feature has 46039 values above 0, the description of this feature was clear about that it was the date decided by the loan company. I will still keep the positive values.

# __6. Aggregation function exploration__

# Since each SK_ID_CURR has multiple SK_BUREAU_ID, I will need aggregation function for different features. I first examine the characteristics of different features by looking at the description table.<br/>
# To apply aggregation function, I should first encode the categorical variables.

# In[ ]:


# no binary categorical variable, use one-hot-encoding
bureau_active_dum = pd.get_dummies(bureau_active)
bureau_closed_dum = pd.get_dummies(bureau_closed)
bureau_active_dum.head()


# In[ ]:


bureau_f = bureau.drop(['SK_ID_CURR','SK_ID_BUREAU'],axis=1).columns
bureau_num_f = [f for f in bureau_f if f not in bureau_cats.index]
sns_distplot_bureau(bureau,bureau_num_f,3,6,(24,12))


# From the distribution plots, I think all above features should be retained in the dataset. Here are the aggregation functions I decided after observing each feature:

# In[ ]:


bureau_agg = {
    'SK_ID_BUREAU': ['count'],
    'DAYS_CREDIT': ['min','max','mean','median','var'],
    'CREDIT_DAY_OVERDUE': ['min','max'],
    'DAYS_CREDIT_ENDDATE': ['min','max','mean'],
    'AMT_CREDIT_MAX_OVERDUE': ['min','max','mean'],
    'CNT_CREDIT_PROLONG': ['mean','sum'],
    'AMT_CREDIT_SUM': ['sum','mean'],
    'AMT_CREDIT_SUM_DEBT': ['sum','mean'],
    'AMT_CREDIT_SUM_LIMIT': ['max','mean'],
    'AMT_CREDIT_SUM_OVERDUE': ['sum'],
    'DAYS_CREDIT_UPDATE': ['min','max'],
    'AMT_ANNUITY': ['min','max','mean'],
    'BUREAU_SCORE1': ['mean','min','max'],
    'HIST_LEN': ['max','min','mean'],
    'BUREAU_SCORE2': ['mean'],
    'BUREAU_SCORE3': ['mean','max'],
}


# For the features created by one-hot-encodeing, I think taking the mean is reasonable.

# In[ ]:


bureau_active_agg = {}
bureau_active_agg.update(bureau_agg)
for col in bureau_active_dum.columns:
    if col not in bureau_active_agg.keys():
        bureau_active_agg.update({col:['mean']})

bureau_closed_agg = {}
bureau_closed_agg.update(bureau_agg)
for col in bureau_closed_dum.columns:
    if col not in bureau_closed_agg.keys():
        bureau_closed_agg.update({col:['mean']})

del bureau_agg


# In[ ]:


bureau_active_grouped = bureau_active_dum.groupby('SK_ID_CURR').agg(bureau_active_agg)
bureau_active_grouped.columns = pd.Index(['BUREAU_ACTIVE_' + e[0] + '_' + e[1].upper() for e in bureau_active_grouped.columns.tolist()])
bureau_closed_grouped = bureau_closed_dum.groupby('SK_ID_CURR').agg(bureau_closed_agg)
bureau_closed_grouped.columns = pd.Index(['BUREAU_CLOSED_' + e[0] + '_' + e[1].upper() for e in bureau_closed_grouped.columns.tolist()])
bureau_grouped = pd.merge(bureau_active_grouped,bureau_closed_grouped, left_index = True, right_index = True).reset_index()
bureau_df = pd.merge(train[['SK_ID_CURR','TARGET']],bureau_grouped,on='SK_ID_CURR',how='left')


# In[ ]:


bureau_df.iloc[:,:4].head()


# In[ ]:


plot_cols = bureau_df.columns[2:]
cat_plot_f = [f for f in plot_cols if bureau_df[f].nunique()<50]
cat_plot_target(bureau_df,cat_plot_f,11,5,(24,48))


# In[ ]:


dist_plot_f = [f for f in plot_cols if f not in cat_plot_f]
sns_distplot_target(bureau_df,dist_plot_f,15,5,(24,50))


# In[ ]:


drop_f = ['CREDIT_CURRENCY']
drop_dum_f = ['BUREAU_ACTIVE_CREDIT_TYPE_Another type of loan_MEAN',
              'BUREAU_ACTIVE_CREDIT_TYPE_Cash loan (non-earmarked)_MEAN',
              'BUREAU_ACTIVE_CREDIT_TYPE_Loan for business development_MEAN',
              'BUREAU_ACTIVE_CREDIT_TYPE_Loan for the purchase of equipment_MEAN',
              'BUREAU_ACTIVE_CREDIT_TYPE_Mobile operator loan_MEAN',
              'BUREAU_ACTIVE_CREDIT_TYPE_Real estate loan_MEAN',
              'BUREAU_ACTIVE_CREDIT_TYPE_Unknown type of loan_MEAN',
              'BUREAU_ACTIVE_STATUS_2_MEAN',
              'BUREAU_ACTIVE_STATUS_3_MEAN',
              'BUREAU_ACTIVE_STATUS_4_MEAN',
              'BUREAU_ACTIVE_STATUS_5_MEAN',
              'BUREAU_CLOSED_CREDIT_DAY_OVERDUE_MIN',
             'BUREAU_CLOSED_CREDIT_DAY_OVERDUE_MAX',
             'BUREAU_CLOSED_CNT_CREDIT_PROLONG_MEAN',
             'BUREAU_CLOSED_CNT_CREDIT_PROLONG_SUM',
             'BUREAU_CLOSED_AMT_CREDIT_SUM_OVERDUE_SUM',
             'BUREAU_CLOSED_CREDIT_TYPE_Another type of loan_MEAN',
             'BUREAU_CLOSED_CREDIT_TYPE_Cash loan (non-earmarked)_MEAN',
             'BUREAU_CLOSED_CREDIT_TYPE_Loan for business development_MEAN',
             'BUREAU_CLOSED_CREDIT_TYPE_Loan for the purchase of equipment_MEAN',
             'BUREAU_CLOSED_CREDIT_TYPE_Loan for working capital replenishment_MEAN',
             'BUREAU_CLOSED_CREDIT_TYPE_Real estate loan_MEAN',
             'BUREAU_CLOSED_CREDIT_TYPE_Unknown type of loan_MEAN',
             'BUREAU_CLOSED_STATUS_1_MEAN',
             'BUREAU_CLOSED_STATUS_2_MEAN',
             'BUREAU_CLOSED_STATUS_3_MEAN',
             'BUREAU_CLOSED_STATUS_4_MEAN',
             'BUREAU_CLOSED_STATUS_5_MEAN',
              'BUREAU_CLOSED_DAYS_CREDIT_ENDDATE_MAX'
             ]


# # Credit card balance

# In[ ]:


credit = data['ccb']
credit.head()


# In[ ]:


credit_cats = cat_features(credit)
credit_num_f = credit.iloc[:,2:].drop(credit_cats.index,axis=1).columns


# In[ ]:


sns_distplot(credit,credit_num_f,4,5)


# Like bureau data, 1 SK_ID_CURR has multiple SK_ID_PREV, aggregation functions need to be assigned. From the above exploration, I have the following aggregation functions:

# In[ ]:





# In[ ]:




