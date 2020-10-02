#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
from kaggle.competitions import twosigmanews
# call it only once
# making the enviornment
env=twosigmanews.make_env()


# In[ ]:





# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgbm
import xgboost as xgb

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[ ]:


market_data,news_data=env.get_training_data()

print(market_data.shape,news_data.shape)

# converting universe to int
market_data['universe']=market_data['universe'].astype(int)

display(market_data.head())
display(news_data.head())


# ## MARKET DATA

# In[ ]:


about=pd.DataFrame(index=market_data.columns)
about['types']=market_data.dtypes.values
about['unique_values']=market_data.nunique().values
about['missing_values']=market_data.isnull().sum().values
about['missing_values_percentage']=about['missing_values']/market_data.shape[0]

about


# In[ ]:


print("Training data starts at",market_data['time'].dt.date.min(),
              "and it ends at",market_data['time'].dt.date.max())


# So it starts at 2nd feb 2007 and ends at 30 december 2016.

# ## REMOVING 2007 AND 2008 DATA

# In[ ]:


# removing the data from news and market
market_data=market_data.loc[market_data['time'].dt.date>=pd.datetime(2009,1,1).date()]
news_data=news_data.loc[news_data['time'].dt.date>=pd.datetime(2009,1,1).date()]
print(market_data.shape,news_data.shape)


# ## LABEL ENCODERS BEFORE SPLITTING THE DATA INTO TRAIN AND VALID
# 
# Before splitting I am making a global label encoders for asset name and asset code. Actually there can be more asset code in test data. 

# In[ ]:


gblabel_encoder_asset_code={name:idx for idx,name in enumerate(market_data['assetCode'].unique())}
gblabel_encoder_asset_name={name:idx for idx,name in enumerate(market_data['assetName'].unique())}


# In[ ]:


def make_the_dataframe(df,assetcode_encoder,assetname_encoder,is_train):
        
    # label encoding for asset code
    unique_values_asset_code=df['assetCode'].unique()
    for name in unique_values_asset_code:
        if name not in assetcode_encoder.keys():
            # so this asset code is not there in the given train data, so we will add new mapping for this
#             print("New asset code is added and the name is",name)
            assetcode_encoder[name]=max(assetcode_encoder.values())+1
            
    df['assetCode_encoding']=df['assetCode'].map(assetcode_encoder)
    
    # label encoding for asset name
    unique_values_asset_name=df['assetName'].unique()
    for name in unique_values_asset_name:
        if name not in assetname_encoder.keys():
            # so this asset name is not there in the given train data
#             print("New asset name added and the name is",name)
            assetname_encoder[name]=max(assetname_encoder.values())+1
    
    df['assetName_encoding']=df['assetName'].map(assetname_encoder)

    # deriving time features
    df['weekofyear']=df['time'].dt.weekofyear
    df['dayofweek']=df['time'].dt.dayofweek
#     df['isleapyear']=df['time'].dt.is_leap_year.astype(int)
#     df['isquarterstart']=df['time'].dt.is_quarter_start.astype(int)
    
    # other features
    df['close_to_open_ratio']=df['close']/df['open']
#     df['number_of_stocks*price']=df['volume']*(df['close']+df['open'])/2
#     df['mrktres_retu_raw_retu1']=df['returnsClosePrevMktres1']-df['returnsClosePrevRaw1']
#     df['mrktres_retu_raw_retu10']=df['returnsClosePrevMktres10']-df['returnsClosePrevRaw10']
    
    if is_train:
        # creating the label (only for train or validation and not for test data)
        df['label']=(df['returnsOpenNextMktres10']>0).astype(int)

        # changing time to date
        df['time']=df['time'].dt.date
    
        necessary_df=df[['time','assetCode','assetName','returnsOpenNextMktres10','universe']].copy()
    
        # dropping off unnecessary columns for training we are using asset name/code encoding
        df.drop(columns=['time','assetName','assetCode','returnsOpenNextMktres10','universe'],inplace=True)
    
        return df,necessary_df
    else:
        # universe is not given in the test data
        df.drop(columns=['time','assetName','assetCode'],inplace=True)
        return df

# making the numpy array  
def make_the_numpy_array(market_data_train,market_data_val):
    X_train=market_data_train.iloc[:,:-1].values
    X_val=market_data_val.iloc[:,:-1].values

    y_train=market_data_train.iloc[:,-1].values
    y_val=market_data_val.iloc[:,-1].values
        
    return X_train,X_val,y_train,y_val

def competitoin_metric(extract,probs):
    extract['confidence']=2*probs-1
    
    extract['score']=extract['universe']*extract['returnsOpenNextMktres10']*extract['confidence']
    
    x=extract.groupby('time')['score'].sum().values
    
    return np.mean(x)/np.std(x)


# In[ ]:


splits=[2012,2013,2014,2015]
models=[]
for idx,year in enumerate(splits):
    print("Year = ",year)
    
    yr,mo,dte=year,1,1
    #there must be minimum 10 day gap between validation and train to ensure there is no overlap of target variable
    print("Splitting the data with respect to time")
    market_data_val=market_data.loc[(market_data['time'].dt.date>=pd.datetime(yr,mo+2,dte).date())]
    market_data_train=market_data.loc[market_data['time'].dt.date<pd.datetime(yr,mo,dte).date()]
    
    market_data_train,extract_train=make_the_dataframe(market_data_train.copy(),gblabel_encoder_asset_code.copy(),
                                            gblabel_encoder_asset_name.copy(),is_train=True)

    market_data_val,extract_val=make_the_dataframe(market_data_val.copy(),gblabel_encoder_asset_code.copy(),
                                            gblabel_encoder_asset_name.copy(),is_train=True)
    
    X_train,X_val,y_train,y_val=make_the_numpy_array(market_data_train,market_data_val)
    
    print("The training data shape is",X_train.shape,y_train.shape)
    print("The validation data shape is",X_val.shape,y_val.shape)
    
    gc.enable()
    del market_data_train,market_data_val    
    gc.collect()
    
    clf=xgb.XGBClassifier(n_jobs=4,max_depth=7)
    clf.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_val,y_val)],eval_metric='logloss',
            early_stopping_rounds=25,verbose=10)
    
    print("The Training Competition metric is:",competitoin_metric(extract_train,
                             clf.predict_proba(X_train)[:,1]))
    print("The validation competition metric is:",competitoin_metric(extract_val,
                             clf.predict_proba(X_val)[:,1]))    
    models.append(clf)
    
    gc.enable()
    del X_train,X_val,y_train,y_val
    gc.collect()


# In[ ]:


# # cross check code
# for idx,year in enumerate(splits):
#     print("Year = ",year)
    
#     yr,mo,dte=year,1,1
#     #there must be minimum 10 day gap between validation and train to ensure there is no overlap of target variable
#     print("Splitting the data with respect to time")
#     market_data_val=market_data.loc[(market_data['time'].dt.date>=pd.datetime(yr,mo+2,dte).date())]
#     market_data_train=market_data.loc[market_data['time'].dt.date<pd.datetime(yr,mo,dte).date()]
    
#     market_data_train,extract_train=make_the_dataframe(market_data_train.copy(),gblabel_encoder_asset_code.copy(),
#                                             gblabel_encoder_asset_name.copy(),is_train=True)

#     market_data_val,extract_val=make_the_dataframe(market_data_val.copy(),gblabel_encoder_asset_code.copy(),
#                                             gblabel_encoder_asset_name.copy(),is_train=True)
    
#     X_train,X_val,y_train,y_val=make_the_numpy_array(market_data_train,market_data_val)
    
#     print("The training data shape is",X_train.shape,y_train.shape)
#     print("The validation data shape is",X_val.shape,y_val.shape)
    
#     gc.enable()
#     del market_data_train,market_data_val    
#     gc.collect()
    
#     clf=models[idx]
    
#     print("The Training Competition metric is:",competitoin_metric(extract_train,
#                          clf.predict_proba(X_train,num_iteration=clf.best_iteration_)[:,1]))
#     print("The validation competition metric is:",competitoin_metric(extract_val,
#                          clf.predict_proba(X_val,num_iteration=clf.best_iteration_)[:,1]))
    
#     models.append(clf)
    
#     gc.enable()
#     del X_train,X_val,y_train,y_val
#     gc.collect()


# In[ ]:





# In[ ]:





# ## MAKING THE PREDICTIONS 

# In[ ]:


def make_the_predictions(models,test_market_data,test_news_data,predictions_template_df):
    test=make_the_dataframe(test_market_data,gblabel_encoder_asset_code.copy(),
                             gblabel_encoder_asset_name.copy(),is_train=False)
    
    X_test=test.iloc[:,:].values
    
    gc.enable()
    del test
    gc.collect()
    
    # making inital predictions to zeros
    
    predictions_template_df['confidenceValue']=0
    
    for clf in models:
        confidences=2*clf.predict_proba(X_test)[:,1]-1
        predictions_template_df['confidenceValue']=predictions_template_df['confidenceValue']+confidences
        
    predictions_template_df['confidenceValue']=predictions_template_df['confidenceValue']/len(models)
    
    return predictions_template_df


# In[ ]:


for day,(market_obs_df, news_obs_df, predictions_template_df) in enumerate(env.get_prediction_days()):
    if(day==0):
        print("The columns of market are",market_obs_df.columns)
    
    predictions=make_the_predictions(models,market_obs_df, news_obs_df, predictions_template_df)
    env.predict(predictions)
    
    if((day+1)%20==0):
        print("Predictions for",day+1,"days done")


# In[ ]:


# writing the submission file
env.write_submission_file()


# In[ ]:





# In[ ]:


# a={'a':0,'b':1,'c':2,'d':3}

# def test(di):
#     cols=['d','e','a','e','f']
#     for name in cols:
#         if name not in di.keys():
#             di[name]=max(di.values())+1
    
#     return di

# new=test(a.copy())
# new1=test(a.copy())
# print(a)
# print(new)
# print(new1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




