#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import OrderedDict
from sklearn.preprocessing import Imputer
pd.set_option('display.max_columns',None)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score,GridSearchCV,RandomizedSearchCV
# Any results you write to the current directory are saved as output.


# In[ ]:


color_dict = OrderedDict({1: 'red', 0: 'green'})
loan_dict = OrderedDict({1: 'Cant repay',0:'Repaid'})
drop_cols = []


# In[ ]:


def plot_continuos(data,var):
    plt.figure(figsize=(8,8))
    for key,clr in color_dict.items():
        sns.kdeplot(data[train.TARGET==key][var].dropna(),color=clr,label = loan_dict[key])
    plt.xlabel(var)
    plt.ylabel('Density')
    
def plot_bar(data,x,y):
    tempdf = data.groupby(x)[y].mean().reset_index()
    sns.barplot(x = x,y = y,data = tempdf)
    plt.xticks(rotation=90)
    plt.show()
    
def plot_count(data,x,hue):
    sns.countplot(x = x,data = data,hue = hue)
    plt.show()
    
def create_missing_columns_report(data):    
    prop = data.isnull().sum()/len(data)
    missing_df = pd.DataFrame(prop).reset_index().rename(columns = {'index':'columns',0:'%missing'})
    missing_df.sort_index(by = '%missing',ascending=False,inplace=True)
    
    dict_column_datatype = data.dtypes.to_dict()
    missing_df['datatype'] = missing_df['columns'].map(dict_column_datatype)
    return missing_df


def convert_categoricals(data):
    ohe_frame = pd.DataFrame()
    ohe_cols= []
    for col in categorical_cols:
        if(data[col].nunique()==2):            
            data.loc[:,col] = np.where(((data[col]=='no')|(data[col]=='N')|(data[col]=='F')),0,1)
            data[col] = data[col].astype(np.int8)         
    data = pd.get_dummies(data)       
    return data

def drop_rows(data,column,value,**args):
    
    index = data.index[(data[data[column]==value])&(data[AMT_INCOME_TOTAL]>1000000)]
    train_filter = train.drop(index,axis=0)

def check_column_consistency(df1, df2):
        """ Check if columns of train and test data are in same order or not. Should be called after train, valid
        and test has been transformed. If certain columns are missing or are not in order then they are added or ordered
        accordingly
        :param df1: train data frame
        :param df2: test or valid data frame
        :return: consistent data frames
        """
        df1_columns = df1.columns.tolist()
        df2_columns = df2.columns.tolist()

        for df1_col in df1_columns:
            if df1_col not in df2_columns:
                df2[df1_col] = 0
        df2 = df2[df1_columns]
        df1 = df1[df1_columns]
        return df1, df2
    
    
def days_age(data):

    #cols_days = [x for x in data if x.startswith('DAYS_')]
    #for col in cols_days:
    data.loc[:,'Age(years)'] = data['DAYS_BIRTH']*-1/365
    return data


def days_employ_flag(data):
    quart90 = np.percentile(data.DAYS_EMPLOYED, 90)
    index = data[data.DAYS_EMPLOYED>=quart90].index
    data.loc[:,'days_employ_flag'] = np.where(data.DAYS_EMPLOYED>=quart90,1,0)
    days_mean = np.mean(data.loc[~(data.DAYS_EMPLOYED>=quart90),'DAYS_EMPLOYED'].values)
    data.loc[index,'DAYS_EMPLOYED'] = days_mean
    
    return data



def train_eval(feature_train,feature_test,target_train,nfolds,test_ids,return_preds=False):
      
    sfold = StratifiedKFold(n_splits= nfolds,shuffle=True,random_state=100)
    valid_scores_list = []
    test_predictions_df = pd.DataFrame()
    feature_columns = feature_train.columns
    feature_importance = np.zeros(len(feature_columns))
    featuresNames = []
    featureImps =[]

    feature_train_arr = feature_train.values
    feature_test_arr = feature_test.values
    target_train_arr = target_train.values
    
    clf_lgb=lgb.LGBMClassifier(  n_estimators=10000,
                                 n_jobs = -1,
                                 metric = 'None',
                                 random_state=100,
                                 class_weight='balanced')
    for i, (train_index,valid_index) in enumerate(sfold.split(feature_train,target_train)):
        fold_predictions_df = pd.DataFrame()        
        # Training and validation data
        X_train = feature_train_arr[train_index]
        X_valid = feature_train_arr[valid_index]
        y_train = target_train_arr[train_index]
        y_valid = target_train_arr[valid_index]
        
        
        fit_params={"early_stopping_rounds":100,
            "eval_metric" : 'auc', 
            "eval_set" : [(X_train,y_train), (X_valid,y_valid)],
            'eval_names': ['train', 'valid'],
            'verbose': 100,
            'categorical_feature': 'auto'}
        
        clf_lgb.fit(X_train,y_train,**fit_params)
        best_iteration = clf_lgb.best_iteration_
        valid_scores_list.append(clf_lgb.best_score_['valid']['auc'])
        display(f'Fold {i + 1}, Validation Score: {round(valid_scores_list[i], 5)}, Estimators Trained: {clf_lgb.best_iteration_}')
        fold_probabilitites = clf_lgb.predict_proba(feature_test_arr,num_iteration = best_iteration)[:,1]      
        fold_predictions_df['Score'] = fold_probabilitites     
        fold_predictions_df['SK_ID_CURR'] = test_ids
        fold_predictions_df['fold'] = (i+1)
        
        test_predictions_df = test_predictions_df.append(fold_predictions_df)
        valid_scores = np.array(valid_scores_list)
        #print(test_predictions_df.shape)
        fold_feature_importance = clf_lgb.feature_importances_
        fold_feature_importance = 100.0 * (fold_feature_importance / fold_feature_importance.max())
        feature_importance = (feature_importance+fold_feature_importance)/nfolds
        sorted_idx = np.argsort(feature_importance)
        for item in sorted_idx[::-1][:]:
            featuresNames.append(np.asarray(feature_columns)[item])
            featureImps.append(feature_importance[item])
            featureImportance = pd.DataFrame([featuresNames, featureImps]).transpose()
            featureImportance.columns = ['FeatureName', 'Importance']
        
    
    # Average the predictions over folds    
    test_predictions_df = test_predictions_df.groupby('SK_ID_CURR', as_index = False).mean()
    #test_predictions_df['Target'] = test_predictions_df[[0,1]].idxmax(axis = 1)
    #test_predictions_df['Score'] = test_predictions_df[1]   
    test_predictions_df.drop('fold',axis=1,inplace=True)   
        
    
    return test_predictions_df,featureImportance,valid_scores


# In[ ]:


train = pd.read_csv("../input/application_train.csv")
test = pd.read_csv("../input/application_test.csv")
prev_appl = pd.read_csv("../input/previous_application.csv")
bureau = pd.read_csv("../input/bureau.csv")
bureau_bal = pd.read_csv("../input/bureau_balance.csv")
#pos_cash = pd.read_csv("../input/POS_CASH_balance.csv")
credit_card_bal = pd.read_csv("../input/credit_card_balance.csv")


# In[ ]:


print('Shape of train:{}'.format(train.shape))
train.head(3)


# In[ ]:


print('Shape of test:{}'.format(test.shape))
test.head(3)


# **Extracting features from Previous application , POS balance and Credit card balance **

# In[ ]:


prev_appl.loc[:,'FLAG_LAST_APPL_PER_CONTRACT'] = prev_appl['FLAG_LAST_APPL_PER_CONTRACT'].map({'Y':1,'N':0})


# In[ ]:


prev_appl.head(3)


# In[ ]:


pos_cash.head(3)


# In[ ]:


'''
pos_cash_agg = pos_cash.groupby(['SK_ID_PREV','SK_ID_CURR'],as_index=False)[['MONTHS_BALANCE','CNT_INSTALMENT','CNT_INSTALMENT_FUTURE','SK_DPD']].agg(['mean','sum','std'])
pos_cash_agg.columns = [' _'.join(col).strip() for col in pos_cash_agg.columns.values]
pos_cash_agg.reset_index(inplace=True)
pos_cash_agg.head(3)
'''


# In[ ]:


credit_card_bal.head(3)


# In[ ]:


credit_card_bal.columns = ['Credit_Card_'+col for col in credit_card_bal.columns.values]
credit_card_bal.head(2)


# In[ ]:


credit_agg = credit_card_bal.groupby(['Credit_Card_SK_ID_PREV','Credit_Card_SK_ID_CURR'],as_index=False)[['Credit_Card_MONTHS_BALANCE','Credit_Card_AMT_BALANCE','Credit_Card_AMT_PAYMENT_CURRENT','Credit_Card_CNT_DRAWINGS_CURRENT']].agg(['mean','sum','std'])
credit_agg.columns = [' _'.join(col).strip() for col in credit_agg.columns.values]
credit_agg.reset_index(inplace=True)
credit_agg.head(3)


# In[ ]:


#del pos_cash
del credit_card_bal


# In[ ]:


agg_data_prev_appl1 = prev_appl.groupby(['SK_ID_PREV','SK_ID_CURR'],as_index=False)[['AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','AMT_DOWN_PAYMENT','AMT_GOODS_PRICE',                                                                      'DAYS_DECISION','CNT_PAYMENT','DAYS_FIRST_DUE']].agg(['mean','sum','std'])
agg_data_prev_appl2 = prev_appl.groupby(['SK_ID_PREV','SK_ID_CURR'],as_index=False)[['FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY']].agg(['sum'])
agg_data_prev_appl3 = prev_appl.groupby(['SK_ID_PREV','SK_ID_CURR'],as_index=False).size().reset_index().rename(columns = {0:'Count'}).set_index(['SK_ID_PREV','SK_ID_CURR'])                                                               
agg_data_prev_appl1.columns = [' _'.join(col).strip() for col in agg_data_prev_appl1.columns.values]
agg_data_prev_appl2.columns = [' _'.join(col).strip() for col in agg_data_prev_appl2.columns.values]
#print(agg_data_prev_appl.columns)
agg_data_prev_appl = pd.concat([agg_data_prev_appl1,agg_data_prev_appl2,agg_data_prev_appl3],axis=1)                                                                     
agg_data_prev_appl.reset_index(inplace=True)
agg_data_prev_appl.columns = ['Previous_Appl_'+col for col in agg_data_prev_appl.columns.values]
#feat_data_prev_appl = pd.merge(agg_data_prev_appl.copy(),pos_cash_agg,left_on  =['Previous_Appl_SK_ID_PREV','Previous_Appl_SK_ID_CURR'], right_on = ['SK_ID_PREV','SK_ID_CURR'],how='left')
feat_data_prev_appl = pd.merge(agg_data_prev_appl.copy(),credit_agg,left_on = ['Previous_Appl_SK_ID_PREV','Previous_Appl_SK_ID_CURR'],right_on = ['Credit_Card_SK_ID_PREV','Credit_Card_SK_ID_CURR'],how='left')
feat_data_prev_appl.drop('Previous_Appl_SK_ID_PREV',axis=1,inplace=True)
del prev_appl
del agg_data_prev_appl1
del agg_data_prev_appl2
del agg_data_prev_appl3

#agg_data_prev_appl.rename(columns = {'Previous_Appl_SK_ID_CURR':'SK_ID_CURR'},inplace=True)


# In[ ]:


feat_data_prev_appl.head(3)


# In[ ]:


train = pd.merge(train,feat_data_prev_appl,left_on='SK_ID_CURR',right_on='Previous_Appl_SK_ID_CURR',how='left')
test = pd.merge(test,feat_data_prev_appl,left_on='SK_ID_CURR',right_on='Previous_Appl_SK_ID_CURR',how='left')
del feat_data_prev_appl
print(train.shape,test.shape)


# **Extracting features from Bureau**

# In[ ]:


bureau.head(4)


# In[ ]:


agg_data_bureau1 = bureau.groupby('SK_ID_CURR',as_index=False)[['DAYS_CREDIT','DAYS_CREDIT_ENDDATE','CREDIT_DAY_OVERDUE','AMT_CREDIT_MAX_OVERDUE','AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_OVERDUE','DAYS_CREDIT_UPDATE','AMT_ANNUITY']].agg(['mean','sum','std'])
agg_data_bureau2 = bureau.groupby('SK_ID_CURR',as_index=False)[['CNT_CREDIT_PROLONG']].agg(['sum'])
agg_data_bureau3 = bureau.groupby('SK_ID_CURR',as_index=False).size().reset_index().rename(columns = {0:'Bureau_Count'}).set_index('SK_ID_CURR')                                                               
agg_data_bureau1.columns = [' _'.join(col).strip() for col in agg_data_bureau1.columns.values]
agg_data_bureau2.columns = [' _'.join(col).strip() for col in agg_data_bureau2.columns.values]
#print(agg_data_prev_appl.columns)
agg_data_bureau = pd.concat([agg_data_bureau1,agg_data_bureau2,agg_data_bureau3],axis=1)                                                                     
agg_data_bureau.reset_index(inplace=True)
agg_data_bureau.columns = ['Bureau_'+col for col in agg_data_bureau.columns.values]
del agg_data_bureau1
del agg_data_bureau2
del agg_data_bureau3
del bureau


# In[ ]:


train = pd.merge(train,agg_data_bureau,left_on='SK_ID_CURR',right_on='Bureau_SK_ID_CURR',how='left')
test = pd.merge(test,agg_data_bureau,left_on='SK_ID_CURR',right_on='Bureau_SK_ID_CURR',how='left')
del agg_data_bureau
print(train.shape,test.shape)


# **The train data has one extra column than test which is the target column**

# In[ ]:


train.info()


# In[ ]:


train.select_dtypes('object').nunique()


# In[ ]:


#Show a head of object columns
train[list(train.select_dtypes('object').columns)].head(2)


# In[ ]:


#Show a head of float columns
train[list(train.select_dtypes(np.float64).columns)].head(2)


# 

# 1.  **EDA**

# * 1. 1. **Distribution of target variable**

# In[ ]:


sns.countplot(train.TARGET)
plt.xlabel('Target')
plt.ylabel('Frequency')
train.TARGET.value_counts(normalize=True)


# **Distribution of target variable is imbalanced as shown in the above plot which signifies that  number of instances where the loan was paid in time  outweighs significantly the number of instances where the loan payment was not done or done after due  dateb**

# 1. 2. **Missing values**

# In[ ]:


missing_df = create_missing_columns_report(train)
missing_df.head(20)


# In[ ]:


missing_df[missing_df.datatype == 'object']


# `**So majority of float columns have the largest missing values and a few categorical columns have missing values**

# In[ ]:


train.describe(percentiles=[0.9,0.92,0.94,0.96,0.98,0.99,0.995])


# In[ ]:


test.describe(percentiles=[0.9,0.92,0.94,0.96,0.98,0.99,0.995])


# **From the above stats, some columns have suspicious values**
# 
# **1.AMT_INCOME_TOTAL:Till 99 percentile, it has values to the order of 5,but the maximum value is of the order 8.Should be an outlier.**
# 
# **2.DAYS_EMPLOYED: Between 90 percentile and 100 perentile, days employed translates to 100 years.Something odd!!.**
# 
# **3.DAYS_BIRTH: It is with reference to the loan application date,so it is a negative number. Working on years will be easier. So I will create another column 'Age' which will be a postive number in  years**
# 
# **4.OBS_30_CNT_SOCIAL_CIRCLE &OBS_60_CNT_SOCIAL_CIRCLE:At 9 percentile, the value is 10, but the maximum value is close to 3o times the 99 percentile value.**   

# In[ ]:


train.query('AMT_INCOME_TOTAL>100000000')


# In[ ]:


train.OCCUPATION_TYPE.unique()


# In[ ]:


plot_bar(train,'OCCUPATION_TYPE','AMT_INCOME_TOTAL')


# In[ ]:


train_lab = train.query('OCCUPATION_TYPE=="Laborers"')
train_lab.describe(percentiles= [0.9,0.94,0.98,0.99,0.995,0.999])


# **From the above stats and  plot  it is clear that the average  income of laborers is around 170k,but the customer with SK_ID_CURR:114967 has an income of 117 million which is clearly an outlier**
# **I will cap the  income column for laborers at 1 million  and in the second plot you can see the distribution after capping**

# In[ ]:


train_lab = train.query('OCCUPATION_TYPE=="Laborers"')
train_lab.drop((train_lab.index[train_lab['AMT_INCOME_TOTAL']>1000000]),axis=0,inplace=True)
plot_continuos(train_lab,'AMT_INCOME_TOTAL')


# In[ ]:


index = train.index[(train.OCCUPATION_TYPE=="Laborers")&(train['AMT_INCOME_TOTAL']>1000000)]
train_filter = train.drop(index,axis=0)


# In[ ]:


train_filter.shape


# In[ ]:


train_filter.describe(percentiles=[0.9,0.92,0.94,0.96,0.98,0.99,0.995])


# In[ ]:


plot_continuos(train_filter,'AMT_INCOME_TOTAL')


# **Handling days employed**

# In[ ]:


plot_continuos(train_filter,'DAYS_EMPLOYED')


# In[ ]:


print('Number of samples:{}'.format(len(train_filter.query('DAYS_EMPLOYED==365243'))))
print('Percentage of samples:{}'.format(len(train_filter.query('DAYS_EMPLOYED==365243'))*100/len(train_filter)))


# **From 90th percentile onwards the column has a constant value of 365243 which is close to 100 years. Thats weird.**
# 
# **Close to 18% of the data have days employed  approx 100 years which is surprising and cant get rid of such a huge chunk of data.**
# 
# **One way to handle this is to replace this value with the mean of the values of this column between 0 to 90th percentile.**
# 
# **Secondly ,create an extra column which will act as a flag  with value 1 where the outlier value exists in the days_employed column.**

# In[ ]:


train_filter2 = days_employ_flag(train_filter.copy())
print('Shape of data before handling DAYS_EMPLOYED:{}'.format(train_filter.shape))
print('Shape of data after handling DAYS_EMPLOYED:{}'.format(train_filter2.shape))
#quart3 = np.percentile(train_filter.DAYS_EMPLOYED, 75)
#iqr = quart3 - quart1

#outlier = train_filter[train_filter['DAYS_EMPLOYED'] > quart3 + 1.5 * iqr].DAYS_EMPLOYED.max()


# In[ ]:


plot_continuos(train_filter2,'DAYS_EMPLOYED')


# **Handling days birth**

# In[ ]:



train_filter3 = days_age(train_filter2.copy())
print('Shape of data before handling DAYS_BIRTH:{}'.format(train_filter2.shape))
print('Shape of data after handling DAYS_BIRTH:{}'.format(train_filter3.shape))
plot_continuos(train_filter3,'Age(years)')


# In[ ]:


drop_cols.append('DAYS_BIRTH')


# **From the above plot, it is clear that as the age decreases, the propensity to pay the credit decreases
# as shown by the  tilt of the red curve towards left**

# In[ ]:


train_filter3[[col for col in train_filter3 if col.startswith('OBS')]].describe(percentiles=[0.9,0.92,0.94,0.96,0.98,0.99,0.995])


# In[ ]:


train_filter3.OBS_30_CNT_SOCIAL_CIRCLE.isnull().sum()


# In[ ]:


outlier_30 = np.nanpercentile(train_filter3.OBS_30_CNT_SOCIAL_CIRCLE,100)
outlier_60 = np.nanpercentile(train_filter3.OBS_60_CNT_SOCIAL_CIRCLE,100)
SK_ID_CURR_30 = train_filter3[train_filter3['OBS_30_CNT_SOCIAL_CIRCLE']>=outlier_30]['SK_ID_CURR'].values[0]
SK_ID_CURR_60=train_filter3[train_filter3['OBS_60_CNT_SOCIAL_CIRCLE']>=outlier_60]['SK_ID_CURR'].values[0]

print(SK_ID_CURR_30 ,',',SK_ID_CURR_60)


# In[ ]:


train_filter3.drop(train_filter3.index[train_filter3.SK_ID_CURR==SK_ID_CURR_30],axis=0,inplace=True)
print('Shape of data after handling outlier row:{}'.format(train_filter3.shape))


# In[ ]:


plot_continuos(train_filter3,'OBS_30_CNT_SOCIAL_CIRCLE')


# In[ ]:


plot_continuos(train_filter3,'OBS_60_CNT_SOCIAL_CIRCLE')


# In[ ]:


train_filter3.head(3)


# In[ ]:


plot_count(train_filter3,'FLAG_MOBIL',hue='TARGET')


# In[ ]:


train_filter3[train_filter3.FLAG_MOBIL==0]


# In[ ]:


plot_count(train_filter3,'FLAG_EMP_PHONE',hue='TARGET')


# In[ ]:


plot_count(train_filter3,'FLAG_WORK_PHONE',hue='TARGET')


# In[ ]:


plot_count(train_filter3,'FLAG_CONT_MOBILE',hue='TARGET')


# In[ ]:


plot_count(train_filter3,'FLAG_PHONE',hue='TARGET')


# In[ ]:


train_filter3['flag_mob'] = train_filter3['FLAG_MOBIL']+train_filter3['FLAG_EMP_PHONE']+                            train_filter3['FLAG_WORK_PHONE']+train_filter3['FLAG_CONT_MOBILE']+train_filter3['FLAG_PHONE']


# In[ ]:


train_filter3.head(3)


# In[ ]:


plot_count(train_filter3,'flag_mob',hue='TARGET')


# In[ ]:


train_filter3['flag_mob'] = train_filter3['flag_mob'].astype('object')


# In[ ]:


drop_cols.extend(['FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE'])
drop_cols


# In[ ]:


print('Shape of data after handling flag_mobile:{}'.format(train_filter3.shape))


# In[ ]:


train_filter3.head(3)


# In[ ]:


test_filter2 = days_employ_flag(test.copy())
print('Shape of data before handling DAYS_EMPLOYED:{}'.format(test.shape))
print('Shape of data after handling DAYS_EMPLOYED:{}'.format(test_filter2.shape))


test_filter3 = days_age(test_filter2.copy())
print('Shape of data before handling DAYS_BIRTH:{}'.format(test_filter2.shape))
print('Shape of data after handling DAYS_BIRTH:{}'.format(test_filter3.shape))



test_filter3['flag_mob'] = test_filter3['FLAG_MOBIL']+test_filter3['FLAG_EMP_PHONE']+                            test_filter3['FLAG_WORK_PHONE']+test_filter3['FLAG_CONT_MOBILE']+test_filter3['FLAG_PHONE']
    
#print('Shape of data before handling flag_mobile:{}'.format(test_filter2.shape))
print('Shape of data after handling flag_mobile:{}'.format(test_filter3.shape))


# In[ ]:


test_filter3['flag_mob'] = test_filter3['flag_mob'].astype('object')


# 1. 3. **How many levels in categorical columns?**

# In[ ]:



categorical_cols = list(train_filter3.select_dtypes('object').columns)
unique_levels = train_filter3[categorical_cols].apply(lambda x: x.nunique())
print('Total levels in categorical columns:{}'.format(unique_levels.sum()))
unique_levels


# 1. 4. **Encoding of categoricals**

# **Converting two levels of categorical columns  to values 0(no or N or F) and 1(yes or Y or M) and more than two levels to be one-hot encoded**

# In[ ]:


train_copy = convert_categoricals(train_filter3.copy())
test_copy = convert_categoricals(test_filter3.copy())

print('Train shape:{},Test shape:{}'.format(train_copy.shape,test_copy.shape))


# In[ ]:


train_copy.head(2)


# In[ ]:


test_copy.head(2)


# In[ ]:


train_copy.info()


# In[ ]:


test_copy.info()


# In[ ]:


train_labels = train_copy['TARGET']
test_ids = test.SK_ID_CURR.values
drop_cols.append('SK_ID_CURR')
print(drop_cols)
train_copy.drop(drop_cols+['TARGET'],axis=1,inplace=True)
test_copy.drop(drop_cols,axis=1,inplace=True)
train_copy, test_copy= check_column_consistency(train_copy, test_copy)
#train_copy,test_copy = train_copy.align(test_copy,axis=1,join='inner')
print('Train shape:{},Test shape:{}'.format(train_copy.shape,test_copy.shape))


# **Train and test data  columns are aligned**

# In[ ]:


train_copy.info()


# In[ ]:


train_copy.head(3)


# In[ ]:


test_copy.head(3)


# In[ ]:


test_copy.info()


# **Modelling**

# In[ ]:



test_predictions_df,featureImportance,valid_scores = train_eval(train_copy,test_copy,train_labels,10,test_ids,return_preds=False)


# In[ ]:


test_predictions_df.head(2)


# In[ ]:


#submission = test_predictions_df[['SK_ID_CURR','Score']]
test_predictions_df.rename(columns = {'Score':'TARGET'},inplace=True)


# In[ ]:


test_predictions_df.head(2)


# In[ ]:


test_predictions_df.to_csv('baseline_lgb.csv', index = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




