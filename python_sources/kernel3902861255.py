#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('rm -r /opt/conda/lib/python3.6/site-packages/lightgbm')
get_ipython().system('git clone --recursive https://github.com/Microsoft/LightGBM')
    
get_ipython().system('apt-get install -y -qq libboost-all-dev')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'cd LightGBM\nrm -r build\nmkdir build\ncd build\ncmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..\nmake -j$(nproc)\n')


# In[ ]:


get_ipython().system('cd LightGBM/python-package/;python3 setup.py install --precompile')

get_ipython().system('mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd')
get_ipython().system('rm -r LightGBM')


# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().system('pip install swifter')
import swifter
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
import os, sys, gc, warnings, random, datetime, gc

from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
from datetime import datetime

import datetime
import math
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# sorted(zip(clf.feature_importances_, X.columns), reverse=True)
gc.collect()


# In[ ]:


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props#, NAlist


# In[ ]:


TARGET = 'label'
features_columns =[]
SEED =1234
LOCAL_TEST = False

rows = 29989752#1000000


# In[ ]:


#del train_df
train_df= pd.read_csv("/kaggle/input/sna-texts/train.csv", 
                      sep=';', 
                      error_bad_lines=False, 
                      dtype  = {'timestamp': np.uint32, 'label': np.uint8, 'C1': np.uint32,
                       'C1': np.uint32, 'C2': np.uint32, 'C3': np.uint16,
                       'C4': np.uint16,'C5': np.uint8, 'C6': np.uint16,
                       'C7': np.uint8,'C8': np.uint16, 'C9': np.uint8,
                       'C10': np.uint16,
                       'CG1': str, 'CG2': str,
                       'CG3': str,'l1': np.uint16, 'l2': np.uint8,
                       'C11': np.uint8, 'C12': np.uint8             
                      } )
#train_df = train_df.head(50000000)
gc.collect()
train_df['index'] = train_df.index


# In[ ]:


train_df.shape


# In[ ]:


#print(train_df.columns, train_df.info, train_df.describe())
#29989752



train_df = reduce_mem_usage(train_df)
#rows = len(train_df)
train_df = train_df.sample(frac=1,  random_state=1)
train_df = train_df.sort_values(by='label', ascending=False)
#stad = train_df.tail(29989752 - rows)
gc.collect()
train_df = train_df.head(rows)


train_df = train_df.sample(frac=1,  random_state=1)

#stat = train_df.sample(frac=0.8, random_state=99)
# you can't simply split 0.75 and 0.25 without overlapping
# this code tries to find that train = 75% and test = 25%
#train_df = train_df.loc[~train_df.index.isin(stat.index), :]


# In[ ]:


train_df.label.mean()


# In[ ]:





# In[ ]:


slovar = {}

def get_count_cg1(x):
    try:
        
        for i in x['CG1'].split(','):
            
            if i not in slovar:
                slovar[i] = 1
            else:
                slovar[i] +=1
    except:
        pass
        
#stad.loc[stad.label ==1].apply(lambda x: get_count_cg1(x), axis=1)


# In[ ]:


cou = 0  
def count_CG(x , colname):
    global cou
    cou+=1
    
    if cou%2000000 == 0:
        print(cou/len(train_df))
    try:
        #if  (len(x.split(","))) > 19:
        #    print( len(x.split(",")))
            
        return  len(x.split(","))
           
    except:
        return 0
def parse_data(x):
    global cou
    cou+=1
    
    temp_full = datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')
    #temp = datetime.datetime.fromtimestamp(x['timestamp']).strftime('%H:%M:%S')
    if cou%1000000 ==0:
        print(cou/len(train_df))
    datetime_o =  datetime.datetime.strptime(temp_full,'%Y-%m-%d %H:%M:%S' )
    #day_of_week = datetime_o.weekday()
    #if (temp > '01:00:00') & (temp < '07:00:00'):
    #return pd.Series([ datetime_o.day, datetime_o.hour])
    return datetime_o.hour
gc.collect()

train_df['CG1_count'] = 0

#train_df['CG1_index'] = train_df.apply(lambda x: mean_CG1(x), axis=1)
train_df['CG1_count'] = train_df.CG1.apply(lambda x: count_CG(x,'CG1'))
gc.collect()
train_df['CG2_count'] = 0    



# In[ ]:


train_df.CG1_count.mean()


# In[ ]:


train_df= train_df.drop(['CG1'], axis=1)
train_df['CG2_count'] = train_df.CG2.apply(lambda x: count_CG(x,'CG2'))
#train_df.swifter.apply(lambda x: count_CG(x,'CG2'), axis=1)
gc.collect()
train_df['CG3_count'] = 0
train_df= train_df.drop(['CG2'], axis=1)
train_df['CG3_count'] = train_df.CG3.apply(lambda x: count_CG(x,'CG3'))
train_df= train_df.drop(['CG3'], axis=1)


   


# In[ ]:


train_df[  'hour'] = train_df.timestamp.apply(lambda x: parse_data(x))


# In[ ]:


#import collections



#od = collections.OrderedDict(sorted(slovar.items()))
#del od 
gc.collect()

#sorted_x = sorted(slovar.items(), key=operator.itemgetter(1))
#for i in sorted_x.keys():
#    print("{0}  {1}".format(i, k))
    


# In[ ]:


train_df.head()


# In[ ]:


#for i in train_df.drop(['label', 'timestamp','CG1', 'CG2', 'CG3' ], axis=1).columns: 
#    if len(list(set(list(train_df[i])))) <100:
#        print(i, list(set(list(train_df[i]))))
        
#categ = [ 'C5','C7','C12']


# In[ ]:



cou = 0

    

    

def min_CG(x , colname):
    global cou
    cou+=1
    
    #if cou%500000 == 0:
    #    print(cou/len(train_df))
    try:
        return  pd.Series([ min([int(i) for i in x[colname].split(",")]),
                          max([int(i) for i in x[colname].split(",")])])
           
    except:
        return pd.Series([0, 0])
    


#train_df['CG1_index'] = 0

#train_df[['CG2_min','CG2_max']] = train_df.swifter.apply(lambda x: min_CG(x,'CG2'), axis=1)
#train_df[['CG3_min', 'CG2_max']] = train_df.swifter.apply(lambda x: min_CG(x,'CG3'), axis=1)

train_df = reduce_mem_usage(train_df)
train_df.to_csv("train.csv", index = False)


# In[ ]:


train_df.head()


# In[ ]:


test_df = pd.read_csv("/kaggle/input/testmail/test.csv", sep=';',
                      error_bad_lines=False, 
                      dtype  = {'timestamp': np.uint32, 'label': np.uint8, 'C1': np.uint32,
                       'C1': np.uint32, 'C2': np.uint32, 'C3': np.uint16,
                       'C4': np.uint16,'C5': np.uint8, 'C6': np.uint16,
                       'C7': np.uint8,'C8': np.uint16, 'C9': np.uint8,
                       'C10': np.uint16,
                       'CG1': str, 'CG2': str,
                       'CG3': str,'l1': np.uint16, 'l2': np.uint8,
                       'C11': np.uint8, 'C12': np.uint8             
                      } )
test_df['index'] = test_df.index



#test_df['CG1_index'] = 0
test_df['CG1_count'] = 0
test_df['CG2_count'] = 0
test_df['CG3_count'] = 0
#test_df['CG1_index'] = test_df.apply(lambda x: mean_CG1(x), axis=1)
test_df['CG1_count'] = test_df.swifter.apply(lambda x: count_CG(x,'CG1'), axis=1)
test_df['CG2_count'] = test_df.swifter.apply(lambda x: count_CG(x,'CG2'), axis=1)
test_df['CG3_count'] = test_df.swifter.apply(lambda x: count_CG(x,'CG3'), axis=1)

test_df= test_df.drop(['CG1',	'CG2',	'CG3'], axis=1)


# In[ ]:


#test_df[['CG2_min','CG2_max']] = test_df.swifter.apply(lambda x: min_CG(x,'CG2'), axis=1)
#test_df[['CG3_min', 'CG2_max']] = test_df.swifter.apply(lambda x: min_CG(x,'CG3'), axis=1)


test_df[ 'hour'] = test_df.timestamp.apply(lambda x: parse_data(x))
test_df = reduce_mem_usage(test_df)


# In[ ]:


#train_df.drop(['CG1',	'CG2',	'CG3'], axis=1).to_csv("train_li.csv", index = False)
#test_df.drop(['CG1',	'CG2',	'CG3'], axis=1).to_csv("train_li.csv", index = False)

test_df.to_csv("test.csv", index = False)


# In[ ]:


test_df.head()





#test_df = test_df.drop(['label'], axis=1)


# In[ ]:





# In[ ]:


test_df.info()


# In[ ]:


convert_dict = {#'C5': 'category', 
                'C7': 'category',
                'C12': 'category',
               } 
  
#train_df = train_df.astype(convert_dict)
#test_df = test_df.astype(convert_dict)
#test_df.info()

categorical_features  = [ 'C7', 'C12']



# In[ ]:


k = len(test_df)
f= len(train_df)
train_df=  train_df.append(test_df)
train_df = pd.get_dummies(train_df, columns=categorical_features)

test_df = train_df.tail(k)
train_df = train_df.head(f)
gc.collect()


# In[ ]:


train_df.info()


# In[ ]:


gc.collect()
test_df = reduce_mem_usage(test_df)
train_df = reduce_mem_usage(train_df)


# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
def make_predictions(tr_df, tt_df, features_columns, target, lgb_params, NFOLDS=2):
    folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

    #X,y = tr_df[features_columns], tr_df[target]    
    P,P_y = tt_df[features_columns], tt_df[target]  

    tt_df = tt_df[['index',target]]    
    predictions = np.zeros(len(tt_df))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(tr_df[features_columns],
                                                           tr_df[target])):
        print('Fold:',fold_)
        tr_x, tr_y = tr_df.loc[trn_idx,features_columns], tr_df[trn_idx,target]
        vl_x, vl_y = tr_df.loc[val_idx,features_columns], tr_df[val_idx,target]
            
        print(len(tr_x),len(vl_x))
        tr_data = lgb.Dataset(tr_x, label=tr_y)

        if LOCAL_TEST:
            vl_data = lgb.Dataset(P, label=P_y) 
        else:
            vl_data = lgb.Dataset(vl_x, label=vl_y)  

        estimator = lgb.train(
            lgb_params,
            tr_data,
            valid_sets = [tr_data, vl_data],
            verbose_eval = 200
        )   
        
        pp_p = estimator.predict(P)
        predictions += pp_p/NFOLDS

        if LOCAL_TEST:
            feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),X.columns)), columns=['Value','Feature'])
            print(feature_imp)
        
        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
        
        
        feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),features_columns)), columns=['Value','Feature'])

        plt.figure(figsize=(20, 10))
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.show()
        plt.savefig('lgbm_importances-01.png')
        
        
    tt_df['prediction'] = predictions
    gc.collect()
    return tt_df
## -------------------


# In[ ]:


lgb_params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'binary_logloss',
                    'n_jobs':-1,
                    'learning_rate':0.01,
                    #'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.7,
                    'subsample_freq':1,
                    'subsample':0.7,
                    'n_estimators':800,
                    'max_bin':255,
                    'device':'gpu', 
                    'verbose':-1,
                    'seed': SEED,
                    'early_stopping_rounds':100, 
    
                } 
#print(train_df.columns)
#print(test_df.columns)
#test_df= test_df.drop(['label'], axis=1)

features_columns = train_df.drop(['label', 'index', 'timestamp'] , axis=1).columns 
if LOCAL_TEST:
    lgb_params['learning_rate'] = 0.01
    lgb_params['n_estimators'] = 800
    lgb_params['early_stopping_rounds'] = 100
    test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params, NFOLDS=5)
    #print(metrics.roc_auc_score(test_predictions[TARGET], test_predictions['prediction']))
else:
    lgb_params['learning_rate'] = 0.005
    #lgb_params['learning_rate'] = 0.01
    lgb_params['n_estimators'] = 1800
    lgb_params['early_stopping_rounds'] = 100    
    test_predictions = make_predictions(train_df.head(22000000), test_df, features_columns, TARGET, lgb_params, NFOLDS=5)
#[200]	training's binary_logloss: 0.0122029	valid_1's binary_logloss: 0.0137984
#[400]	training's binary_logloss: 0.0110326	valid_1's binary_logloss: 0.0134888
#Fold: 0
#3999999 1000001
#[200]	training's binary_logloss: 0.0132474	valid_1's binary_logloss: 0.0135375
#[400]	training's binary_logloss: 0.0128195	valid_1's binary_logloss: 0.013369

#Fold: 0
#3999999 1000001
#Training until validation scores don't improve for 100 rounds.
#[200]	training's binary_logloss: 0.0137024	valid_1's binary_logloss: 0.0138937

#Early stopping, best iteration is:
#[47]	training's binary_logloss: 0.0167729	valid_1's binary_logloss: 0.0190796


# In[ ]:


#test_predictions.head()


# In[ ]:


sam = pd.read_csv("/kaggle/input/testmail/sample_submission.csv")

sam.head(n=5)
#sam.join(test_predictions)



# In[ ]:


sam['0'] = test_predictions['prediction']


# In[ ]:


sam.head()


# In[ ]:


sam.shape


# In[ ]:


sam.to_csv("submisson.csv", index=False)

