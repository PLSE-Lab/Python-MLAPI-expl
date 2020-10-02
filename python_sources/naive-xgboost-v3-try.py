#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import xgboost as xgb
import time

def change_datatype(df):
    for col in list(df.select_dtypes(include=['int']).columns):
        if np.max(df[col]) <= 127 and np.min(df[col]) >= -128:
            df[col] = df[col].astype(np.int8)
        elif np.max(df[col]) <= 255 and np.min(df[col]) >= 0:
            df[col] = df[col].astype(np.uint8)
        elif np.max(df[col]) <= 32767 and np.min(df[col]) >= -32768:
            df[col] = df[col].astype(np.int16)
        elif np.max(df[col]) <= 65535 and np.min(df[col]) >= 0:
            df[col] = df[col].astype(np.uint16)
        elif np.max(df[col]) <= 2147483647 and np.min(df[col]) >= -2147483648:
            df[col] = df[col].astype(np.int32)
        elif np.max(df[col]) <= 4294967296 and np.min(df[col]) >= 0:
            df[col] = df[col].astype(np.uint32)
    for col in list(df.select_dtypes(include=['float']).columns):
        df[col] = df[col].astype(np.float32)
        
def count_words(key):
    return len(str(key).split())

def count_numbers(key):
    return sum(c.isalpha() for c in key)

def count_upper(key):
    return sum(c.isupper() for c in key)

def get_mean(df, name, target, alpha=0):
    group = df.groupby(name)[target].agg([np.sum, np.size])
    mean = train[target].mean()
    series = (group['sum'] + mean*alpha)/(group['size']+alpha)
    series.name = name + '_mean'
    return series.to_frame().reset_index()

def add_words(df, name, length):
    x_data = []
    for x in df[name].values:
        x_row = np.ones(length, dtype=np.uint16)*0
        for xi, i in zip(list(str(x)), np.arange(length)):
            x_row[i] = ord(xi)
        x_data.append(x_row)
    return pd.concat([df, pd.DataFrame(x_data, columns=[name+str(c) for c in range(length)]).astype(np.uint16)], axis=1)

start_time = time.time()
c_categories = ['name', 'category_name', 'brand_name', 'item_description']
c_means = ['category_name', 'item_condition_id', 'brand_name']
c_texts = ['name', 'item_description']
c_ignors = ['name', 'item_description', 'brand_name', 'category_name', 'train_id', 'test_id', 'price']

train = pd.read_csv('../input/train.tsv', sep='\t')
test = pd.read_csv('../input/test.tsv', sep='\t')
test['price'] = -1

df = pd.concat([train, test]).reset_index()
change_datatype(df)
df = df.fillna('')
df = add_words(df, 'name', 43) 
df = add_words(df, 'item_description', 60)

for c in c_categories:
     df[c+'_cat'] = pd.factorize(df[c])[0]

for c in c_texts:
    df[c + '_c_words'] = df[c].apply(count_words)
    df[c + '_c_upper'] = df[c].apply(count_upper)
    df[c + '_c_numbers'] = df[c].apply(count_numbers)
    df[c + '_len'] = df[c].str.len()
    df[c + '_mean_len_words'] = df[c + '_len']/df[c + '_c_words']
    df[c + '_mean_upper'] = df[c + '_len']/df[c + '_c_upper']
    df[c + '_mean_numbers'] = df[c + '_len']/df[c + '_c_numbers']
    
#------- begin feature engineering (Leandro dos Santos Coelho)
df['fe001'] = np.square(df["name_mean_len_words"])
df['fe002'] = np.square(df["item_description_mean_len_words"])
df['fe003'] = np.tanh(df["name_mean_len_words"])
df['fe004'] = np.tanh(df["item_description_mean_len_words"])
df['fe005'] = df["name_mean_len_words"]**2.37
df['fe006'] = df["item_description_mean_len_words"]**2.15
#------- end feature engineering (Leandro dos Santos Coelho)

d_median = df.median(axis=0)
d_mean = df.mean(axis=0)

def transform_df(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in c_ignors]
    df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
    for c in dcol: #standard arithmetic
        df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
        df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)
        df[c+str('_sq')] = np.power(df[c].values,2).astype(np.float32)
        df[c+str('_sqr')] = np.square(df[c].values).astype(np.float32)
    return df

def multi_transform(df):
    print('Init Shape: ', df.shape)
    p = Pool(cpu_count())
    df = p.map(transform_df, np.array_split(df, cpu_count()))
    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
    p.close(); p.join()
    print('After Shape: ', df.shape)
    return df

df = multi_transform(df)
print("Final DF shape: ", df.shape())
test = df[df['price'] == -1]
train = df[df['price'] != -1]
del df
print("Final Train shape: ", train.shape())
print("Final Test shape: ", test.shape())

train, valid = np.split(train.sample(frac=1), [int(.75*train.shape[0])])

for c in c_means:
    mean = get_mean(train, c, 'price')
    test = test.merge(mean, on=[c], how='left')
    train = train.merge(mean, on=[c], how='left')
    valid = valid.merge(mean, on=[c], how='left')

col = [c for c in train.columns if c not in c_ignors]

dtrain = xgb.DMatrix(train[col], train['price'])
dvalid  = xgb.DMatrix(valid[col],  valid['price'])
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

params = {'eta': 0.02, 'max_depth': 10, 'subsample': 0.9, 'colsample_bytree': 0.9, 'booster' : 'gbtree',
          'objective': 'reg:linear', 'eval_metric': 'rmse', 'seed': 99, 'silent': True}

model = xgb.train(params, dtrain, 1000, watchlist, verbose_eval=10, early_stopping_rounds=20)
test['price'] = model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit)

test.loc[test['price'] < 0, 'price'] = 0
test['test_id'] = test['test_id'].astype(int)
test[['test_id', 'price']].to_csv("xgb_submission.csv", index = False)
print("Finished ...")
tt = (time.time() - start_time)/60
print("Total time %s min" % tt)


# In[ ]:




