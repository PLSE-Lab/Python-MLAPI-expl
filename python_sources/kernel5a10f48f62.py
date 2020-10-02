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


# 

# In[ ]:


TARGET = 'label'
features_columns =[]
SEED =1234
LOCAL_TEST = False

rows = 29989752#26000000#24000000#23000000#22000000#18000000


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


train_df.shape[0]


# In[ ]:


train_df[train_df.label == 1].shape[0]


# In[ ]:


rel = train_df[train_df.label == 1].shape[0]/train_df.shape[0]
print(rel)


# In[ ]:


#print(train_df.columns, train_df.info, train_df.describe())
#29989752



train_df = reduce_mem_usage(train_df)
#rows = len(train_df)
#train_df = train_df.sample(frac=1,  random_state=1)
#train_df = train_df.sort_values(by='label', ascending=False)
#stad = train_df.tail(29989752 - rows)


gc.collect()
#train_df = train_df.head(rows)


#train_df = train_df.sample(frac=1,  random_state=1)

#stat = train_df.sample(frac=0.8, random_state=99)
# you can't simply split 0.75 and 0.25 without overlapping
# this code tries to find that train = 75% and test = 25%
#train_df = train_df.loc[~train_df.index.isin(stat.index), :]


# In[ ]:


#list(x)[:100]


# In[ ]:





# In[ ]:


cou = 0  
def count_CG(x , colname):
    global cou
    cou+=1
    
    if cou%2000000 == 0:
        #print(cou/len(train_df))
        print(cou)
    try:
        #if  (len(x.split(","))) > 19:
        #    print( len(x.split(",")))
            
        return  len(x.split(","))
           
    except:
        return 0


# In[ ]:



def parse_data(x):
    global cou
    cou+=1
    
    temp_full = datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')
    #temp = datetime.datetime.fromtimestamp(x['timestamp']).strftime('%H:%M:%S')
    if cou%1000000 ==0:
        print(cou)
    datetime_o =  datetime.datetime.strptime(temp_full,'%Y-%m-%d %H:%M:%S' )
    #day_of_week = datetime_o.weekday()
    #if (temp > '01:00:00') & (temp < '07:00:00'):
    #return pd.Series([ datetime_o.day, datetime_o.hour])
    return datetime_o.hour




# In[ ]:


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


x, x_test, y, y_test = train_test_split(train_df['index'].values, 
                                            train_df['label'].values, 
                                            test_size=0.15, random_state=42, 
                                            stratify=train_df['label'].values)
del y, y_test
print(x.shape, x_test.shape)


# In[ ]:


train_df = reduce_mem_usage(train_df)

train_df85 = train_df[train_df.index.isin(x)]
train_df15 = train_df[train_df.index.isin(x_test)]

del train_df, x, x_test


# In[ ]:


#import collections

train_df85[  'hour'] = train_df85.timestamp.apply(lambda x: parse_data(x))
train_df15[  'hour'] = train_df15.timestamp.apply(lambda x: parse_data(x))

#od = collections.OrderedDict(sorted(slovar.items()))
#del od 
gc.collect()

#sorted_x = sorted(slovar.items(), key=operator.itemgetter(1))
#for i in sorted_x.keys():
#    print("{0}  {1}".format(i, k))
    


# In[ ]:


#train_df.head()


# In[ ]:


#for i in train_df.drop(['label', 'timestamp' ], axis=1).columns: 
#    if len(list(set(list(train_df[i])))) <100:
#        print(i, list(set(list(train_df[i]))))
        
#categ = [ 'C5','C7','C12']


# In[ ]:


#gc.collect()
#train_df.groupby(['label']).agg([  'mean', 'count'])


# In[ ]:


#sns.catplot(x="C12", y="label",  palette="ch:.25", data=train_df.head(10000));
gc.collect()
#df =train_df.groupby(['C7']).agg([  'mean', 'count'])

cat = 0
def pr(x):
    global cat
    
    print(x.name,  x['label']['mean'], x['label']['count'] )
    cat+=1
#df.apply(lambda x: pr(x), axis=1)


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
sns.set(color_codes=True)


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
train_df85 = reduce_mem_usage(train_df85)
train_df15 = reduce_mem_usage(train_df15)
#train_df.to_csv("train.csv", index = False)


# In[ ]:


#train_df.head()


# In[ ]:


#test_df[['CG2_min','CG2_max']] = test_df.swifter.apply(lambda x: min_CG(x,'CG2'), axis=1)
#test_df[['CG3_min', 'CG2_max']] = test_df.swifter.apply(lambda x: min_CG(x,'CG3'), axis=1)



# In[ ]:


#train_df.drop(['CG1',	'CG2',	'CG3'], axis=1).to_csv("train_li.csv", index = False)
#test_df.drop(['CG1',	'CG2',	'CG3'], axis=1).to_csv("train_li.csv", index = False)


# In[ ]:


#test_df.head()





#test_df = test_df.drop(['label'], axis=1)


# In[ ]:





# In[ ]:


#test_df.info()


# In[ ]:


convert_dict = {'C5': 'category', 
                'C7': 'category',
                'C12': 'category',
               } 
  
#train_df = train_df.astype(convert_dict)
#test_df = test_df.astype(convert_dict)
#test_df.info()

categorical_features  = [  'C12']



# In[ ]:


#k = len(test_df)
#f= len(train_df)
#train_df=  train_df.append(test_df)
#train_df = pd.get_dummies(train_df, columns=categorical_features)

#test_df = train_df.tail(k)
#train_df = train_df.head(f)
gc.collect()


# In[ ]:


#print(f)


# In[ ]:


train_df85['mulC2C3'] = train_df85['C2'] * train_df85['C3']

train_df85['mulC3C10'] = train_df85['C3'] * train_df85['C10']
train_df85['mulC2C10'] = train_df85['C2'] * train_df85['C10']
train_df85['mulC1C2'] = train_df85['C1'] * train_df85['C2']
train_df85['mulC1C3'] = train_df85['C1'] * train_df85['C3']

train_df85['diffC1C2'] = train_df85['C1'] - train_df85['C2']
#train_df85['multC8C4'] = train_df85['C8'] * train_df85['C4']
#train_df85['multC8C3'] = train_df85['C8'] * train_df85['C3']
#train_df['mulL1L2'] = train_df['l1'] * train_df['l2'] 

gc.collect()

train_df85 = reduce_mem_usage(train_df85.fillna(0))
train_df85['partC5_1']  = False
#train_df['partC5_58'] = False
train_df85['partC5_30'] = False
#train_df['partC5_31'] = False
#train_df['partC5_52'] = False
#train_df['partC5_88'] = False
train_df85['partC5_all_other'] = False

train_df85.loc[train_df85.C5 == 1,   'partC5_1']  = True
#train_df.loc[train_df.C5 == 58,  'partC5_58'] = True
#train_df.loc[train_df.C5 == 88,  'partC5_88'] = True
train_df85.loc[train_df85.C5 == 30,  'partC5_30'] = True
#train_df.loc[train_df.C5 == 31,  'partC5_31'] = True
#train_df.loc[train_df.C5 == 52,  'partC5_52'] = True
train_df85.loc[~train_df85.C5.isin([1,58,88,30,31,52]),  'partC5_all_other'] = True
#train_df['mulL1L2'] =test_df['l1'] * test_df['l2'] 


# In[ ]:


train_df15['mulC2C3'] = train_df15['C2'] * train_df15['C3']

train_df15['mulC3C10'] = train_df15['C3'] * train_df15['C10']
train_df15['mulC2C10'] = train_df15['C2'] * train_df15['C10']
train_df15['mulC1C2'] = train_df15['C1'] * train_df15['C2']
train_df15['mulC1C3'] = train_df15['C1'] * train_df15['C3']

train_df15['diffC1C2'] = train_df15['C1'] - train_df15['C2']
#train_df15['multC8C4'] = train_df15['C8'] * train_df15['C4']
#train_df15['multC8C3'] = train_df15['C8'] * train_df15['C3']
#train_df['mulL1L2'] = train_df['l1'] * train_df['l2'] 

gc.collect()

train_df15 = reduce_mem_usage(train_df15.fillna(0))
train_df15['partC5_1']  = False
#train_df['partC5_58'] = False
train_df15['partC5_30'] = False
#train_df['partC5_31'] = False
#train_df['partC5_52'] = False
#train_df['partC5_88'] = False
train_df15['partC5_all_other'] = False

train_df15.loc[train_df15.C5 == 1,   'partC5_1']  = True
#train_df.loc[train_df.C5 == 58,  'partC5_58'] = True
#train_df.loc[train_df.C5 == 88,  'partC5_88'] = True
train_df15.loc[train_df15.C5 == 30,  'partC5_30'] = True
#train_df.loc[train_df.C5 == 31,  'partC5_31'] = True
#train_df.loc[train_df.C5 == 52,  'partC5_52'] = True
train_df15.loc[~train_df15.C5.isin([1,58,88,30,31,52]),  'partC5_all_other'] = True
#train_df['mulL1L2'] =test_df['l1'] * test_df['l2'] 


# In[ ]:


train_df15.head()


# In[ ]:


gc.collect()


train_df15 = reduce_mem_usage(train_df15.fillna(0))
train_df85 = reduce_mem_usage(train_df85.fillna(0))
features_columns = train_df15.drop(['label'] , axis=1).columns 
#features_columns = train_df.drop(['label', 'index', 'timestamp'] , axis=1).columns 


# In[ ]:



#train_df = train_df.drop(['index'], axis=1)


# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold


# In[ ]:



def make_predictions(tr_df, tt_df, features_columns, target, lgb_params, NFOLDS=2):
    folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

    #X,y = tr_df[features_columns], tr_df[target]    
    P,P_y = tt_df[features_columns], tt_df[target]  

    tt_df = tt_df[['index',target]]    
    predictions = np.zeros(len(tt_df))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(tr_df[features_columns],
                                                           tr_df[target])):
        print('Fold:',fold_)
        tr_x, tr_y = tr_df.loc[trn_idx,features_columns], tr_df.loc[trn_idx,target]
        vl_x, vl_y = tr_df.loc[val_idx,features_columns], tr_df.loc[val_idx,target]
            
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





# In[ ]:


from sklearn.model_selection import train_test_split
lgb_params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'binary_logloss',
                    'n_jobs':-1,
                    'learning_rate':0.01,
                    'num_leaves': 2**8,
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

#
if LOCAL_TEST:
    lgb_params['learning_rate'] = 0.01
    lgb_params['n_estimators'] = 800
    lgb_params['early_stopping_rounds'] = 100
    test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params, NFOLDS=5)
    #print(metrics.roc_auc_score(test_predictions[TARGET], test_predictions['prediction']))
else:
    lgb_params['learning_rate'] = 0.005
    #lgb_params['learning_rate'] = 0.01
    lgb_params['n_estimators'] = 3000
    lgb_params['early_stopping_rounds'] = 100  
    
    
    #x, x_test, y, y_test = train_test_split(train_df[features_columns].values, 
    #                                        train_df['label'].values, 
    #                                        test_size=0.15, random_state=42, 
    #                                        stratify=train_df['label'].values)
    #del train_df
    
    gc.collect()
    test_data = lgb.Dataset(np.array(train_df15[features_columns].values), label=np.array(train_df15['label'].values))
    del train_df15
    train_data = lgb.Dataset(np.array(train_df85[features_columns].values), label=np.array(train_df85['label'].values))
    del train_df85
    


#
# Train the model
#



    model = lgb.train(lgb_params,
                       train_data,
                       valid_sets=[train_data, test_data],
                       #num_boost_round=5000,
                       early_stopping_rounds=100,
                       verbose_eval = 200)
    feature_imp = pd.DataFrame(sorted(zip(model.feature_importance(),features_columns)), columns=['Value','Feature'])

    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()
    plt.savefig('lgbm_importances-01.png')
        
#
    
    
    #test_predictions = make_predictions(train_df.head(12000000), test_df, features_columns, TARGET, lgb_params, NFOLDS=5)

    
#Training until validation scores don't improve for 100 rounds.
#[200]	training's binary_logloss: 0.0148189	valid_1's binary_logloss: 0.0151401
#[400]	training's binary_logloss: 0.0142033	valid_1's binary_logloss: 0.0147117
#[600]	training's binary_logloss: 0.0138752	valid_1's binary_logloss: 0.0145791
#[800]	training's binary_logloss: 0.0136128	valid_1's binary_logloss: 0.0145258
#[1000]	training's binary_logloss: 0.0133741	valid_1's binary_logloss: 0.0144938
#[1200]	training's binary_logloss: 0.0131609	valid_1's binary_logloss: 0.0144757
#[1400]	training's binary_logloss: 0.012965	valid_1's binary_logloss: 0.0144641
#[1600]	training's binary_logloss: 0.0127857	valid_1's binar_logloss: 0.0144562
#[1800]	training's binary_logloss: 0.0126186	valid_1's binary_logloss: 0.0144505
#Did not meet early stopping. Best iteration is:
#[1800]	training's binary_logloss: 0.0126186	valid_1's binary_logloss: 0.0144505
del train_data, test_data


# In[ ]:


import pandas as pd
#test_df = pd.read_csv(,  sep=',',error_bad_lines=False)

test_df = pd.read_csv("/kaggle/input/fixeddataset1/test-datad.csv", sep=',',
                      error_bad_lines=False, 
                      dtype  = {"Unnamed: 0": np.uint32,
                                'test.csv': np.uint32, 'label': np.uint8, 'C1': np.uint32,
                       'C1': np.uint32, 'C2': np.uint32, 'C3': np.uint16,
                       'C4': np.uint16,'C5': np.uint8, 'C6': np.uint16,
                       'C7': np.uint8,'C8': np.uint16, 'C9': np.uint8,
                       'C10': np.uint16,
                       'CG1': str, 'CG2': str,
                       'CG3': str,'l1': np.uint16, 'l2': np.uint8,
                       'C11': np.uint8, 'C12': np.uint8             
                      } )
test_df = test_df.drop(["Unnamed: 0"], axis=1)

names = list(test_df.columns)

names[0] = 'timestamp'
test_df.columns  = names
test_df.shape


# In[ ]:


test_df.head()


# In[ ]:


#test_predictions.head()



test_df['index'] = test_df.index



#test_df['CG1_index'] = 0
test_df['CG1_count'] = 0
test_df['CG2_count'] = 0
test_df['CG3_count'] = 0
#test_df['CG1_index'] = test_df.apply(lambda x: mean_CG1(x), axis=1)
test_df['CG1_count'] = test_df.CG1.apply(lambda x: count_CG(x,'CG1'))
test_df['CG2_count'] = test_df.CG2.apply(lambda x: count_CG(x,'CG2'))
test_df['CG3_count'] = test_df.CG3.apply(lambda x: count_CG(x,'CG3'))


test_df= test_df.drop(['CG1',	'CG2',	'CG3'], axis=1)

test_df.to_csv("test.csv", index = False)


# In[ ]:


test_df[ 'hour'] = test_df.timestamp.apply(lambda x: parse_data(x))
test_df = reduce_mem_usage(test_df)


# In[ ]:


#test_df = pd.get_dummies(test_df, columns=categorical_features)


# In[ ]:


test_df['mulC2C3'] = test_df['C2'] * test_df['C3']

test_df['mulC3C10'] = test_df['C3'] * test_df['C10']
test_df['mulC2C10'] = test_df['C2'] * test_df['C10']


test_df['mulC1C2'] = test_df['C1'] * test_df['C2']
test_df['mulC1C3'] = test_df['C1'] * test_df['C3']


test_df['diffC1C2'] = test_df['C1'] - test_df['C2']
#test_df['multC8C4'] = test_df['C8'] * test_df['C4']
#test_df['multC8C3'] = test_df['C8'] * test_df['C3']


test_df = reduce_mem_usage(test_df.fillna(0))
test_df['partC5_1']  = False
#test_df['partC5_58'] = False
#test_df['partC5_88'] = False
test_df['partC5_30'] = False
#test_df['partC5_31'] = False
#test_df['partC5_52'] = False
#test_df['partC5_88'] = False
test_df['partC5_all_other'] = False


test_df.loc[test_df.C5 == 1,   'partC5_1']  = True
#test_df.loc[test_df.C5 == 58,  'partC5_58'] = True
#test_df.loc[test_df.C5 == 88,  'partC5_88'] = True
test_df.loc[test_df.C5 == 30,  'partC5_30'] = True
#test_df.loc[test_df.C5 == 31,  'partC5_31'] = True
#test_df.loc[test_df.C5 == 52,  'partC5_52'] = True

test_df.loc[~test_df.C5.isin([1,58,88,30,31,52]),  'partC5_all_other'] = True

test_df = reduce_mem_usage(test_df.fillna(0))


# In[ ]:


sam = pd.read_csv("/kaggle/input/testmail/sample_submission.csv")

sam.head(n=5)
#sam.join(test_predictions)



# In[ ]:


print(sam.shape )
print(test_df.shape)


# In[ ]:




##
features_columns = test_df.drop(['label'] , axis=1).columns 
test_df['0'] = model.predict(test_df[features_columns].values)


# In[ ]:


sam = test_df[[ '0']]


# In[ ]:


sam.shape


# In[ ]:


sam.head(n=10)


# In[ ]:



out = sorted(list(sam['0']), reverse = True)
print(int(rel*len(out)))
first  = out[3351]


# In[ ]:


#out[int(3351/2)]
for i in range(len(out)):
    if out[i] <=0.5:
        
        second = out[i]
        break


# In[ ]:


print(first, second)


# In[ ]:


sam.to_csv("submisson3000boosting.csv", header= False,index=False)
sam1=sam.copy()
sam2=sam.copy()


# In[ ]:


sam1.loc[sam1['0']>= second, '0'] = 1
print(sam1['0'].mean())
sam1.to_csv("submisson3000boosting05porog.csv", header= False,index=False)


# In[ ]:


sam2.loc[sam2['0']>= first, '0'] = 1
print(sam2['0'].mean())
sam2.to_csv("submissonboosting023porogcount.csv", header= False,index=False)

