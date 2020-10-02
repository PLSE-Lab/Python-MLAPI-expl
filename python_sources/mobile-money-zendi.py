#!/usr/bin/env python
# coding: utf-8

# In[50]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datetime import timedelta
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import os
print(os.listdir("../input/fixedagain/"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df=pd.read_csv('../input/mobile-money-after-merging-withfstd-no-mobile/new_train.csv')#preprocceced files
test_df=pd.read_csv('../input/fixedagain/new_test_fixed_final.csv')#preprocceced files
train_df.drop(columns=['mobile_nearby'],inplace=True)# not yet calculated
test_df['Age']=test_df['Age'].apply(lambda x : 0 if x<25 else 1 if x<65 else 2)
submission=pd.read_csv('../input/mobile-money-and-financial-inclusion-in-tanzania/sample_submission.csv')
### -------------------------------reading FSTD data files ------------------------
geo_space_cols = ['region', 'district','ward','latitude','longitude'] # reading only thoose cols for optimisation
third_party=pd.read_csv('../input/mobile-money-and-financial-inclusion-in-tanzania/3rd_ppp_for_upload_win.csv',encoding='ISO-8859-1', usecols=geo_space_cols)
atm=pd.read_csv('../input/mobile-money-and-financial-inclusion-in-tanzania/atm_for_upload_win.csv', usecols=geo_space_cols)
banks=pd.read_csv('../input/mobile-money-and-financial-inclusion-in-tanzania/banks_for_upload_win.csv', usecols=geo_space_cols)
bus=pd.read_csv('../input/mobile-money-and-financial-inclusion-in-tanzania/bus_stands_for_upload_win.csv', usecols=geo_space_cols)
micro_finance=pd.read_csv('../input/mobile-money-and-financial-inclusion-in-tanzania/microfinance_for_upload_win.csv', usecols=geo_space_cols)
mobile=pd.read_csv('../input/mobile-money-and-financial-inclusion-in-tanzania/mobilemoney_agents_for_upload_win.csv',encoding='ISO-8859-1', usecols=geo_space_cols)
pos=pd.read_csv('../input/mobile-money-and-financial-inclusion-in-tanzania/pos_for_upload_win.csv', usecols=geo_space_cols)
post_office=pd.read_csv('../input/mobile-money-and-financial-inclusion-in-tanzania/post_office_for_upload_win.csv', usecols=geo_space_cols)
sacco=pd.read_csv('../input/mobile-money-and-financial-inclusion-in-tanzania/sacco_for_upload_win.csv', usecols=geo_space_cols)
sacco.head()


# In[ ]:


#### updatng col name
'''train_col_names=['ID','Age','gender','Marital_status','education','owning_house_land','owning_other_lands','own_mobile','getting_money_salaries',
 'getting_money_trading','getting_money_sevice_provider','getting_money_Occasional_jobs','getting_money_Rental_income','getting_money_savings',
 'getting_money_Pension','getting_money_Government','getting_money_someone_else','getting_money_someone_else_pay','Other',
 'who_you_work_for','type_of_your_sells','type_of_sevice_you_provide','have_sent_money_last_year','when_last_send','have_recieved_money_last_year',
 'when_last_recieve','mobile_usage_for_purshace_last_year', 'mobile_usage_for_bills_last_year','Literacy_Kiswhahili','Literacy_english'
,'latitude','longitude','mobile_money','savings','borrowing','insurance','mobile_money_classification']
test_col_names=['ID','Age','gender','Marital_status','education','owning_house_land','owning_other_lands','own_mobile','getting_money_salaries',
 'getting_money_trading','getting_money_sevice_provider','getting_money_Occasional_jobs','getting_money_Rental_income','getting_money_savings',
 'getting_money_Pension','getting_money_Government','getting_money_someone_else','getting_money_someone_else_pay','Other',
 'who_you_work_for','type_of_your_sells','type_of_sevice_you_provide','have_sent_money_last_year','when_last_send','have_recieved_money_last_year',
 'when_last_recieve','mobile_usage_for_purshace_last_year', 'mobile_usage_for_bills_last_year','Literacy_Kiswhahili','Literacy_english'
,'latitude','longitude']'''


# In[ ]:


#transforming age into bins
train_df['Age']=train_df['Age'].apply(lambda x : 0 if x<25 else 1 if x<65 else 2)
test_df['Age']=test_df['Age'].apply(lambda x : 0 if x<25 else 1 if x<65 else 2)
train_df.head()


# **See Train people distribution**

# In[ ]:


## for ploting !!
'''import plotly.plotly as py
import plotly.graph_objs as go

scl = [ [1,"rgb(5, 10, 172)"],[1,"rgb(40, 60, 190)"],[1,"rgb(70, 100, 245)"],\
    [1,"rgb(90, 120, 245)"]]
data = [ go.Scattergeo(
    
        lon = mobile['longitude'],
        lat = mobile['latitude'],
        mode = 'markers',
        marker = dict( 
            size = 4, 
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'circle',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorbar=dict(
                title="sacco"
            )
            

        ))]

layout = dict(
        title = 'train geo space distribution', 
    width=1000,
    height=1000,
        geo = dict(
            scope='africa',
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5        
        ),
    )

fig = go.Figure(data=data, layout=layout )
iplot(fig) '''


# In[ ]:


# adding region stuffs using knn from mobile data 
#region	district	ward	latitude	longitude

def imputate_region(train,test,df_reference): # df ref is the data frame to use for imputation (ex: mobile)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(df_reference['region'].values.tolist())
    neigh = KNeighborsClassifier()
    neigh.fit(df_reference[['latitude','longitude']].values.tolist(), le.transform(df_reference['region'].values.tolist()))
    train['region']=le.inverse_transform(list(train.apply(lambda x : neigh.predict([x[['latitude','longitude']].values])[0],axis=1)))
    test['region']=le.inverse_transform(list(test.apply(lambda x : neigh.predict([x[['latitude','longitude']].values])[0],axis=1)))

    return train,test

def imputate_district(train,test,df_reference): # df ref is the data frame to use for imputation (ex: mobile)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import preprocessing
    neigh = KNeighborsClassifier()
    le = preprocessing.LabelEncoder()
    le.fit(df_reference['district'].tolist())
    neigh.fit(df_reference[['latitude','longitude']].values.tolist(), le.transform(df_reference['district'].values.tolist()))
    train['district']=le.inverse_transform(list(train.apply(lambda x : neigh.predict([x[['latitude','longitude']].values])[0],axis=1)))
    test['district']=le.inverse_transform(list(test.apply(lambda x : neigh.predict([x[['latitude','longitude']].values])[0],axis=1)))
    return train,test

def imputate_ward(train,test,df_reference): # df ref is the data frame to use for imputation (ex: mobile)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import preprocessing
    neigh = KNeighborsClassifier()
    le = preprocessing.LabelEncoder()
    le.fit(df_reference['ward'].tolist())
    neigh.fit(df_reference[['latitude','longitude']].values, le.transform(df_reference['ward'].values.tolist()))
    train['ward']=le.inverse_transform(list(train.apply(lambda x : neigh.predict([x[['latitude','longitude']].values])[0],axis=1)))
    test['ward']=le.inverse_transform(list(test.apply(lambda x : neigh.predict([x[['latitude','longitude']].values])[0],axis=1)))
    return train,test


# In[ ]:


#already done!
'''train_df,test_df=imputate_region(train_df,test_df,mobile)
train_df,test_df=imputate_district(train_df,test_df,mobile)
train_df,test_df=imputate_ward(train_df,test_df,mobile)
train_df.head()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(train_df['region'].tolist()+test_df['region'].tolist())
train_df['region']=le.transform(train_df['region'].values)
test_df['region']=le.transform(test_df['region'].values)
le.fit(train_df['district'].tolist()+test_df['district'].tolist())
train_df['district']=le.transform(train_df['district'].values)
test_df['district']=le.transform(test_df['district'].values)
le.fit(train_df['ward'].tolist()+test_df['ward'].tolist())
train_df['ward']=le.transform(train_df['ward'].values)
test_df['ward']=le.transform(test_df['ward'].values)

train_df.head()
'''


# In[ ]:


'''#add features from 
# general literacy: english + Kiswhahili
train_df["general_literacy"]=train_df.Literacy_Kiswhahili + train_df.Literacy_english
test_df["general_literacy"]=test_df.Literacy_Kiswhahili + test_df.Literacy_english

# send and receive money
train_df["send_and_received_money"]=train_df.have_sent_money_last_year + train_df.have_recieved_money_last_year
test_df["send_and_received_money"]=test_df.have_sent_money_last_year + test_df.have_recieved_money_last_year

# people who don't own a mobile and rely on others
train_df["no_mobile_and_rely_on_others"]=train_df.getting_money_Pension + train_df.getting_money_someone_else_pay
test_df["no_mobile_and_rely_on_others"]=test_df.getting_money_Pension + test_df.getting_money_someone_else_pay

# frequency of using mobile money
train_df["mobile_money_use_freq"]=train_df.mobile_usage_for_purshace_last_year + train_df.mobile_usage_for_bills_last_year
test_df["mobile_money_use_freq"]=test_df.mobile_usage_for_purshace_last_year + test_df.mobile_usage_for_bills_last_year

# source of income
train_df["source_of_income"]=0
test_df["source_of_income"]=0
for x in ['getting_money_salaries','getting_money_trading','getting_money_sevice_provider','getting_money_Occasional_jobs','getting_money_Rental_income','getting_money_savings','getting_money_Pension',
          'getting_money_Government','getting_money_someone_else','getting_money_someone_else_pay','Other']:
    train_df["source_of_income"]=train_df["source_of_income"] + train_df[x]
    test_df["source_of_income"]=test_df["source_of_income"] + test_df[x]
train_df.head()'''


# In[ ]:


# return haversine distance in mile
def distance_less_than_5km(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return int(3.107>(0.6213712 * 12742 * np.arcsin(np.sqrt(a)))) # 2*R*asin
# return haversine distance in mile
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return (0.6213712 * 12742 * np.arcsin(np.sqrt(a))) # 2*R*asin


# In[ ]:


get_ipython().run_cell_magic('time', '', '# running only this to test if it improve the results or no\n\'\'\'train_df["mobile_nearby"]=0\ntest_df["mobile_nearby"]=0\ntrain_df["mobile_nearby"]=train_df.apply(lambda y : np.sum(mobile.apply(lambda x : distance_less_than_5km(y[\'latitude\'],y[\'longitude\'],x[\'latitude\'],x[\'longitude\']), axis=1)),axis=1)\ntest_df["mobile_nearby"]=test_df.apply(lambda y : np.sum(mobile.apply(lambda x : distance_less_than_5km(\n    y[\'latitude\'],y[\'longitude\'],x[\'latitude\'],x[\'longitude\']), axis=1)),axis=1)\n\'\'\'')


# In[52]:


from IPython.display import FileLink
def create_submission(submission_file, submission_name):
    submission_file.to_csv(submission_name+".csv",index=False)
    return FileLink(submission_name+".csv")
create_submission(submission_file=test_df, submission_name="new_test_fixed_with_mobile.csv")


# In[53]:


create_submission(submission_file=train_df, submission_name="new_train_fixed_with_mobile.csv")


# In[56]:


train_df.head()


# In[ ]:


'''%%time

## Add Fstd data  It take time

train_df["3rd_party_nearby"]=train_df.apply(lambda y : np.sum(third_party.apply(lambda x : distance_less_than_5km(y['latitude'],y['longitude'],x['latitude'],x['longitude']), axis=1)),axis=1)
train_df["atm_nearby"]=train_df.apply(lambda y : np.sum(atm.apply(lambda x : distance_less_than_5km(y['latitude'],y['longitude'],x['latitude'],x['longitude']), axis=1)),axis=1)
train_df["banks_nearby"]=train_df.apply(lambda y : np.sum(banks.apply(lambda x : distance_less_than_5km(y['latitude'],y['longitude'],x['latitude'],x['longitude']), axis=1)),axis=1)
#train_df["mobile_nearby"]=train_df.apply(lambda y : np.sum(mobile.apply(lambda x : distance_less_than_5km(y['latitude'],y['longitude'],x['latitude'],x['longitude']), axis=1)),axis=1)
train_df["bus_nearby"]=train_df.apply(lambda y : np.sum(bus.apply(lambda x : distance_less_than_5km(y['latitude'],y['longitude'],x['latitude'],x['longitude']), axis=1)),axis=1)
train_df["micro_finance_nearby"]=train_df.apply(lambda y : np.sum(micro_finance.apply(lambda x : distance_less_than_5km(y['latitude'],y['longitude'],x['latitude'],x['longitude']), axis=1)),axis=1)
train_df["pos_nearby"]=train_df.apply(lambda y : np.sum(pos.apply(lambda x : distance_less_than_5km(y['latitude'],y['longitude'],x['latitude'],x['longitude']), axis=1)),axis=1)
train_df["post_office_nearby"]=train_df.apply(lambda y : np.sum(post_office.apply(lambda x : distance_less_than_5km(y['latitude'],y['longitude'],x['latitude'],x['longitude']), axis=1)),axis=1)
train_df["sacco_nearby"]=train_df.apply(lambda y : np.sum(sacco.apply(lambda x : distance_less_than_5km(y['latitude'],y['longitude'],x['latitude'],x['longitude']), axis=1)),axis=1)



'''


# In[ ]:


'''%%time
test_df["atm_nearby"]=test_df.apply(lambda y : np.sum(atm.apply(lambda x : distance_less_than_5km(y['latitude'],y['longitude'],x['latitude'],x['longitude']), axis=1)),axis=1)
'''


# In[ ]:


'''%%time
test_df["banks_nearby"]=test_df.apply(lambda y : np.sum(banks.apply(lambda x : distance_less_than_5km(y['latitude'],y['longitude'],x['latitude'],x['longitude']), axis=1)),axis=1)
#test_df["mobile_nearby"]=train_df.apply(lambda y : np.sum(mobile.apply(lambda x : distance_less_than_5km(y['latitude'],y['longitude'],x['latitude'],x['longitude']), axis=1)),axis=1)
'''


# In[ ]:


'''%%time
test_df["bus_nearby"]=test_df.apply(lambda y : np.sum(bus.apply(lambda x : distance_less_than_5km(y['latitude'],y['longitude'],x['latitude'],x['longitude']), axis=1)),axis=1)
test_df["micro_finance_nearby"]=test_df.apply(lambda y : np.sum(micro_finance.apply(lambda x : distance_less_than_5km(y['latitude'],y['longitude'],x['latitude'],x['longitude']), axis=1)),axis=1)
test_df["pos_nearby"]=test_df.apply(lambda y : np.sum(pos.apply(lambda x : distance_less_than_5km(y['latitude'],y['longitude'],x['latitude'],x['longitude']), axis=1)),axis=1)
test_df["post_office_nearby"]=test_df.apply(lambda y : np.sum(post_office.apply(lambda x : distance_less_than_5km(y['latitude'],y['longitude'],x['latitude'],x['longitude']), axis=1)),axis=1)
test_df["sacco_nearby"]=test_df.apply(lambda y : np.sum(sacco.apply(lambda x : distance_less_than_5km(y['latitude'],y['longitude'],x['latitude'],x['longitude']), axis=1)),axis=1)
'''


# In[ ]:


'''%%time
test_df["3rd_party_nearby"]=test_df.apply(lambda y : np.sum(third_party.apply(lambda x : distance_less_than_5km(y['latitude'],y['longitude'],x['latitude'],x['longitude']), axis=1)),axis=1)
'''


# In[ ]:





# In[71]:


def train_model_xgb(X_train,Y_train,X_val,Y_val,X_test,parmaters,features_name): 
    d_train = xgb.DMatrix(X_train, Y_train,feature_names=features_name)
    d_valid = xgb.DMatrix(X_val, Y_val,feature_names=features_name)
    d_test = xgb.DMatrix(X_test,feature_names=features_name)
    list_track = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(parmaters, d_train, 99999,  list_track, maximize=False, verbose_eval=50,
                              early_stopping_rounds=100)
    train_pred =model.predict(d_train)            
    valid_pred =model.predict(d_valid)
    test_pred = model.predict(d_test)
    return train_pred ,valid_pred,test_pred,model


# In[72]:


def train_kfold_xgb(X_train,Y_train,X_test,parmaters,features_name,split=5):
    final_train_pred=np.zeros((X_train.shape[0],4))
    final_test_pred=np.zeros((X_test.shape[0],4))
    from sklearn.model_selection import StratifiedKFold
    kf =  StratifiedKFold(n_splits=split)
    i=1
    
    for train_index, val_index in kf.split(X_train, Y_train):
        print("fold:"+str(i))
        train_fold_features, val_fold_features = X_train.loc[train_index], X_train.loc[val_index]
        train_fold_target, val_fold_target = Y_train.loc[train_index], Y_train.loc[val_index] 
        train_pred ,valid_pred,test_pred,model=train_model_xgb( 
                                                        X_train=train_fold_features,
                                                        Y_train= train_fold_target,
                                                        X_val= val_fold_features,
                                                        Y_val= val_fold_target,
                                                        X_test= X_test,
                                                        parmaters=parmaters,
                                                        features_name=features_name
                                                    )
        
        final_train_pred[val_index]=valid_pred
        final_test_pred=final_test_pred+test_pred
        i=i+1
    return final_train_pred,final_test_pred,model


# In[60]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'inta':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# In[61]:


train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)
#submission = reduce_mem_usage(submission)


# In[62]:


train_df.drop(columns=['mobile_money','savings','borrowing','insurance'],inplace=True)
train_df.head() 


# In[63]:



target_train=train_df['mobile_money_classification']
train_df.drop(columns=['mobile_money_classification'],inplace=True)


# In[64]:


def train_Xgboost(train_df,test_df,Y,boosting_type='gbdt',metric='mlogloss') :
    
    import gc # garbej collector for mempry optimisation
    gc.enable()
    from sklearn.metrics import accuracy_score # to be changed in case of AUC,...
    from sklearn.metrics import roc_auc_score,mean_squared_error
    from sklearn.model_selection import train_test_split    
    from sklearn.metrics import mean_absolute_error

    X=train_df.copy()
    Y=Y.copy()
    #use this in case classification

    
    dtrain = xgb.DMatrix(X, label=Y)
    dtest = xgb.DMatrix(test_df)

  
    
    params = {
        # Parameters that we are going to tune.        
        'max_depth':3,
        'min_child_weight': 1,
        'eta':.3,
        'num_class': 4,
        'subsample': 1,
        'colsample_bytree': 1,
        # Other parameters
        'objective':'multi:softprob'
    }
    params['eval_metric'] = metric
    num_boost_round = 5000
    
    
    
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics=metric,
        early_stopping_rounds=10
    )
    print('Best MAE with cv : '+str(cv_results['test-'+str(metric)+'-mean'].min()))
    
    
    
    print('--Tunning Parameters max_depth and min_child_weight--')
    # Define initial best params and MAE
    min_mae = float("Inf")
    best_params = None
    
    gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(3,12)
    for min_child_weight in range(5,8)
    ]

    for max_depth, min_child_weight in gridsearch_params:
        print("CV with max_depth={}, min_child_weight={}".format(
                                 max_depth,
                                 min_child_weight))
        # Update our parameters
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=metric,
            early_stopping_rounds=10
        )
        # Update best MAE
        mean_mae = cv_results['test-'+str(metric)+'-mean'].min()
        boost_rounds = cv_results['test-'+str(metric)+'-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (max_depth,min_child_weight)
        
    params['max_depth'] = best_params[0]
    params['min_child_weight'] = best_params[1]
    print('--Tunning Parameters subsample and colsample_bytree--')
    gridsearch_params = [
        (subsample, colsample)
        for subsample in [i/10. for i in range(7,11)]
        for colsample in [i/10. for i in range(7,11)]
    ]
    min_mae = float("Inf")
    best_params = None
    # We start by the largest values and go down to the smallest
    for subsample, colsample in reversed(gridsearch_params):
        print("CV with subsample={}, colsample={}".format(
                                 subsample,
                                 colsample))
        # We update our parameters
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=metric,
            early_stopping_rounds=10
        )
        # Update best score
        mean_mae = cv_results['test-'+str(metric)+'-mean'].min()
        boost_rounds = cv_results['test-'+str(metric)+'-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (subsample,colsample)
    print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
    params['subsample'] = best_params[0]
    params['colsample_bytree'] = best_params[1]
    best_params=0.1
    for eta in [.3, .2, .1, .05, .01, .005]:
        print("CV with eta={}".format(eta))
        # We update our parameters
        params['eta'] = eta
        # Run and time CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=metric,
            early_stopping_rounds=10
              )
        # Update best score
        mean_mae = cv_results['test-'+str(metric)+'-mean'].min()
        boost_rounds = cv_results['test-'+str(metric)+'-mean'].argmin()
        print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = eta
    print("Best params: {}, MAE: {}".format(best_params, min_mae))
    params['eta'] = best_params
    print("Final Best params: {}".format(params))
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, "Test")],
        early_stopping_rounds=10
        )
    print("Best MAE: {:.2f} in {} rounds".format(model.best_score, model.best_iteration+1))



    final_train_pred,final_test_pred,model=train_kfold_xgb(X_train=train_df,Y_train=Y,X_test=test_df,parmaters=params,features_name=X.columns,split=10)

    return final_train_pred,final_test_pred,model


# In[68]:


best_params=   {'max_depth': 3, 'min_child_weight': 7, 'eta': 0.005,
                'num_class': 4, 'subsample': 0.7, 'colsample_bytree': 1.0, 'objective': 'multi:softprob', 'eval_metric': 'mlogloss'}
zaid_best_params={'colsample_bytree': 0.5,
                'eta': 0.1,
                'eval_metric': 'mlogloss',
                'max_depth': 3,
                'min_child_weight': 8,
                'num_class': 4,
                'objective': 'multi:softprob',
                'subsample': 0.6} 


# In[ ]:


best_features=['Age','gender','Marital_status','education','owning_house_land',
               'owning_other_lands','own_mobile','getting_money_Occasional_jobs',
               'type_of_your_sells','have_sent_money_last_year','when_last_send','have_recieved_money_last_year',
               'when_last_recieve','mobile_usage_for_purshace_last_year','mobile_usage_for_bills_last_year',
               'Literacy_Kiswhahili','Literacy_english','latitude','longitude','region',
                'general_literacy','send_and_received_money','mobile_money_use_freq',
                'source_of_income','3rd_party_nearby','atm_nearby','banks_nearby',
                'sacco_nearby']


# In[ ]:





# In[ ]:


test_df.head()


# In[76]:


#final_train_pred,final_test_pred,model=train_kfold_xgb(train_df,Y_train=target_train,
                                                       X_test=test_df,parmaters=zaid_best_params,
                                                       features_name=train_df.columns,
                                                       split=5)


# In[67]:


#final_train_pred,final_test_pred,model= train_Xgboost(train_df,test_df,target_train,boosting_type='gbdt',metric='mlogloss')


# In[77]:


submission[['no_financial_services','other_only','mm_only','mm_plus']]=final_test_pred/5
submission.head()


# In[ ]:


from IPython.display import FileLink
def create_submission(submission_file, submission_name):
    submission_file.to_csv(submission_name+".csv",index=False)
    return FileLink(submission_name+".csv")


# In[ ]:


#calcul_score(final_train_pred,target_train)


# In[ ]:


def calcul_score(probabilities,target):# predicted proba for 4 classed+taget col
    from sklearn.metrics import log_loss
    res=0
    for i in range(4):
        for j in range(len(target)):
            if(target[j]==i):
                res+=(log_loss([1],[probabilities[i,j]]))
            else:
                res+=(log_loss([0],[probabilities[i,j]]))
    print('logloss :' +str(res))
    return res
                
    


# In[78]:


create_submission(submission_file=submission, submission_name="xgb_with_geo_features_with_mobile_5folds_zaid_params")


# In[ ]:





# In[ ]:





# ### Light gbm part 

# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
#import xgboost as xgb
import pickle
import os
import gc
gc.enable()


# In[ ]:


train_df.head()


# In[ ]:


import seaborn as sns
sns.scatterplot(x='latitude',y='longitude',data=train_df)


# In[ ]:


def mean_encoding(train_data,test_data,columns,target_col):# updated version
    train_new=train_data.copy()
    test_new=test_data.copy()
    for column in columns:
        train_new[column + "_mean_target"] = None
    y = train_data[target_col].values
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    X=train_data.drop(columns=[target_col]).values

    for training_index, validation_index in  skf.split(X,y):
        x_train = train_data.iloc[training_index]
        x_validation = train_data.iloc[validation_index]
        # 'columns' is a list of columns to encode
        for column in columns:
            means = x_validation[column].map(x_train.groupby(column)[target_col].mean())
            x_validation[column + "_mean_target"] = means
        # train_new is a dataframe copy we made of the training data
        
        train_new.iloc[validation_index] = x_validation

    global_mean = train_data[target_col].mean()

    # replace nans with the global mean
    train_new.fillna(global_mean, inplace=True)
    for col in columns:
            grouped_p=train_new.groupby(col,as_index=False).agg({col+"_mean_target":"mean"})
            print(grouped_p)
            test_new=test_new.merge(grouped_p,how='left')
    return train_new,test_new


# In[ ]:


def fit_lgb(X_fit, y_fit, X_val, y_val, counter,lgb_path,test, name):
    
    model = lgb.LGBMClassifier(objective = "multiclass", 
    boosting = "gbdt",
    boost_from_average=False,
    num_threads=8,
    learning_rate =0.01,
    max_depth=3,
   
    num_iterations =99999999,
    )
    model.fit(X_fit, y_fit, 
              eval_set=[(X_val, y_val)],
              verbose=100, 
              early_stopping_rounds=100)
                  
    cv_val = model.predict_proba(X_val)[:]
    test_res = model.predict_proba(test.values)[:]
    #Save LightGBM Model
    #save_to = '{}{}_fold{}.txt'.format(lgb_path, name, counter+1)
    #model.booster_.save_model(save_to)
    print(test_res.shape)
    return cv_val,test_res,model
   


# In[ ]:


def train_stage(train,target,test):
   
   y_df = np.array(target)                
   df=train.copy()
   df_ids = np.array(df.index)                     
   lgb_cv_result = np.zeros((train.shape[0],4))
   lgb_result = np.zeros((test.shape[0],4))
   df_ids = np.array(train.index)              
   skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
   skf.get_n_splits(df_ids, y_df)
   model= lgb.LGBMClassifier()
   print('\nModel Fitting...')
   for counter, ids in enumerate(skf.split(df_ids, y_df)):
       print('\nFold {}'.format(counter+1))
       X_fit, y_fit = df.values[ids[0]], y_df[ids[0]]
       X_val, y_val = df.values[ids[1]], y_df[ids[1]]
   
       a,b,model= fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path,test, name='lgb')
       lgb_cv_result[ids[1],:]+=a
       lgb_result[:]+=b
       
       print(lgb_result[:,:])
   
   
   #auc_lgb  = round(roc_auc_score(y_df, lgb_cv_result),4)
   print('\nLightGBM VAL done: ')
   submission = pd.read_csv('../input/mobile-money-and-financial-inclusion-in-tanzania/sample_submission.csv')
   submission[['no_financial_services','other_only','mm_only','mm_plus']] = lgb_result/10
   return lgb_cv_result/10,submission,model


# In[ ]:


target_train=train_df['mobile_money_classification']
train_df.drop(columns=['mobile_money_classification'],inplace=True)


# In[ ]:


#Create dir for models
lgb_path = './lgb_models_stack/'
#os.mkdir(lgb_path)

print('Train Stage.\n')
lgb_cv_result,submission,model=train_stage(train_df,target_train,test_df)

submission.head()


# In[ ]:


create_submission(submission_file=submission, submission_name="light_gbm_10_with_location_and_all_fstd_without_mobile")


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
light_gbm_model=model
sns.barplot(x=light_gbm_model.feature_importances_,y=train_df.columns, orient='h')


# #### KERAS 

# In[ ]:


from keras import layers,models,callbacks
import matplotlib.pyplot as plt
from keras.utils import to_categorical

def smooth(points,factor=0.9): # fonction to smooth the output plot
    smoothed_points=[]
    for p in points:
        if(smoothed_points):
            prev=smoothed_points[-1]
            smoothed_points.append(prev*0.9+p*(1-factor))
        else:
            smoothed_points.append(p)
    return smoothed_points


# In[ ]:





# In[ ]:


# normilize train df
def normlize(col):
    return col.apply(lambda x :(x-np.min(col))/(np.max(col)-np.min(col)))
                     
from sklearn.preprocessing import StandardScaler
norml_train_df=train_df.copy()

for c in train_df.columns:
    norml_train_df[c]=normlize(norml_train_df[c])
    
norml_train_df.head()
    


# In[ ]:


# normilize test df
                     
norml_test_df=test_df.copy()
#during normalisation , i used train df because the other is already normilized
for c in train_df.columns:
    norml_test_df[c]=norml_test_df[c].apply(lambda x :
                                            (x-np.min(train_df[c]))/(np.max(train_df[c])-np.min(train_df[c])))
    
norml_test_df.head()


# In[ ]:


norml_train_df.drop(columns=["mobile_nearby"],inplace=True)
norml_test_df.drop(columns=["mobile_nearby"],inplace=True)


# In[ ]:


model=models.Sequential()
model.add(layers.Dense(32,activation='relu',input_shape=(norml_train_df.shape[1],)))
model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(16, activation='relu')) 
model.add(layers.Dense(4, activation='softmax')) 
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


# Set callback functions to early stop training and save the best model so far
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
             ]


# In[ ]:


# convert class vectors to binary class matrices
from sklearn.model_selection import train_test_split
import keras
X_train, X_test, y_train, y_test = train_test_split(norml_train_df.values, target_train, test_size=0.2, random_state=42,stratify=target_train)
y_train = keras.utils.to_categorical(y_train, 4)
y_test = keras.utils.to_categorical(y_test, 4)
nb_epochs=50


# In[ ]:


history=model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=nb_epochs,batch_size=128,callbacks=callbacks, verbose=2)


# In[ ]:


tr_loss=smooth(history.history['loss'])
val_loss=smooth(history.history['val_loss'])
tr_acc=smooth(history.history['acc'])
val_acc = smooth(history.history['val_acc'])
f, axes = plt.subplots(1, 2)

steps=[x for x in range (nb_epochs)]
d = {'tr_loss': tr_loss,'steps':steps[:len(tr_loss)],'val_loss':val_loss,'tr_acc':tr_acc,'val_acc':val_acc}
df = pd.DataFrame(data=d)
sns.lineplot(  y="tr_acc", x= "steps", data=df,  ax=axes[0],color='r',label="Training acc")
sns.lineplot(  y="val_acc", x= "steps", data=df,  ax=axes[0],color='b',label="Validation acc")
sns.lineplot(  y="tr_loss", x= "steps", data=df,  ax=axes[1],color='r',label="Training loss")
sns.lineplot(  y="val_loss", x= "steps", data=df,  ax=axes[1],color='b',label="Validation loss")


# In[ ]:


predictions=model.predict(norml_test_df)


# In[ ]:


submission[['no_financial_services','other_only','mm_only','mm_plus']]=predictions
submission.head()


# In[ ]:


create_submission(submission_file=submission, submission_name="Keras_first_try")


# In[ ]:




