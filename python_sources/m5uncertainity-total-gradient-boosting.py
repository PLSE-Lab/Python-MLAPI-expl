#!/usr/bin/env python
# coding: utf-8

# # Here we have used Gradient Bossting aaproach Total Sales cases:
# 
# <hr>
# 
# 1. Data Preparation for all levels of Aggregation is done at this notebook: https://www.kaggle.com/kamalnaithani/m5unceratinityadddata
# 2. With this approach we can merge sales level data as well
# 3. There will be seperate model for all aggregated form in same way as we have calculated for Total sales, in same it can be done for category, store-category etc
# 4. Applying gradient boosting approcah and mergin the output of all final models
# 5. Do upvote in case you find this notebook helpful
# 
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import gc
# General imports
import os, sys, gc, time, warnings, pickle, psutil, random

# custom imports
from multiprocessing import Pool        # Multiprocess Runs

warnings.filterwarnings('ignore')


# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
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
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
#
def autocorrelation(ys, t=1):
    return np.corrcoef(ys[:-t], ys[t:])


# In[ ]:


#==========================================================================
def preprocess_sales(sales, start=1400, upper=1970):
    if start is not None:
        print("dropping...")
        to_drop = [f"d_{i+1}" for i in range(start-1)]
        print(sales.shape)
        sales.drop(to_drop, axis=1, inplace=True)
        print(sales.shape)
    #=======
    print("adding...")
    new_columns = ['d_%i'%i for i in range(1942, upper, 1)]
    for col in new_columns:
        sales[col] = np.nan
    print("melting...")
    sales = sales.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id","scale","start"],
                        var_name='d', value_name='demand')

    print("generating order")
    if start is not None:
        skip = start
    else:
        skip = 1
    sales["nb"] =sales.index // 1 + skip
    return sales
#===============================================================
def preprocess_calendar(calendar):
    global maps, mods
    calendar["event_name"] = calendar["event_name_1"]
    calendar["event_type"] = calendar["event_type_1"]

    map1 = {mod:i for i,mod in enumerate(calendar['event_name'].unique())}
    calendar['event_name'] = calendar['event_name'].map(map1)
    map2 = {mod:i for i,mod in enumerate(calendar['event_type'].unique())}
    calendar['event_type'] = calendar['event_type'].map(map2)
    calendar['nday'] = calendar['date'].str[-2:].astype(int)
    maps["event_name"] = map1
    maps["event_type"] = map2
    mods["event_name"] = len(map1)
    mods["event_type"] = len(map2)
    calendar["wday"] -=1
    calendar["month"] -=1
    #calendar["year"] -= 2011
    mods["month"] = 12
    mods["year"] = 6
    mods["wday"] = 7
    mods['snap_CA'] = 2
    mods['snap_TX'] = 2
    mods['snap_WI'] = 2

    calendar.drop(["event_name_1", "event_name_2", "event_type_1", "event_type_2", "date", "weekday"], 
                  axis=1, inplace=True)
    return calendar
#=========================================================
def make_dataset(categorize=False ,start=1400, upper= 1970):
    global maps, mods
    print("loading calendar...")
    calendar = pd.read_csv("../input/m5-forecasting-uncertainty/calendar.csv")
    print("loading sales...")
    sales = pd.read_pickle("../input/m5unceratinityadddata/TotalSales.pkl")
    cols = ["item_id", "dept_id", "cat_id","store_id","state_id"]
    if categorize:
        for col in cols:
            temp_dct = {mod:i for i, mod in enumerate(sales[col].unique())}
            mods[col] = len(temp_dct)
            maps[col] = temp_dct
        for col in cols:
            sales[col] = sales[col].map(maps[col])
        #

    sales =preprocess_sales(sales, start=start, upper= upper)
    calendar = preprocess_calendar(calendar)
    calendar = reduce_mem_usage(calendar)
    print("merge with calendar...")
    sales = sales.merge(calendar, on='d', how='left')
    del calendar

    print("reordering...")
    sales.sort_values(by=["id","nb"], inplace=True)
    print("re-indexing..")
    sales.reset_index(inplace=True, drop=True)
    gc.collect()

    sales['n_week'] = (sales['nb']-1)//7
    sales["nday"] -= 1
    mods['nday'] = 31
    sales = reduce_mem_usage(sales)
    gc.collect()
    return sales
#===================


# In[ ]:


get_ipython().run_cell_magic('time', '', 'CATEGORIZE = True;\nSTART = 1; UPPER = 1970;\nmaps = {}\nmods = {}\nsales = make_dataset(categorize=CATEGORIZE ,start=START, upper= UPPER)')


# As we are asked to predict a time window of 28 days, the easiest way to go now is to use the last 28 days for validation:

# In[ ]:


sales['d'] = sales['d'].apply(lambda x: x[2:]).astype(np.int16)


# In[ ]:


START


# In[ ]:


sales.head()


# In[ ]:





# In[ ]:


sales.head()


# In[ ]:


TARGET      = 'demand'
BASE     = '../input/m5unceratinityadddata/TotalSales.pkl'
START_TRAIN = 1 
END_TRAIN   = 1913
P_HORIZON   = 28
remove_features = ['id','state_id','store_id',
                   'date','wm_yr_wk','d',TARGET]
grid_df=sales
del sales


# In[ ]:


grid_df.info()


# In[ ]:


########################### Apply on grid_df
grid_df1 = grid_df[['id','d','demand']]
SHIFT_DAY = 28
SHIFT_DAY1 = 1

# Lags
# with 28 day shift
start_time = time.time()
print('Create lags')

#LAG_DAYS = [col for col in range(SHIFT_DAY,SHIFT_DAY+15)]
LAG_DAYS = [col for col in range(SHIFT_DAY,SHIFT_DAY+14)]
grid_df1 = grid_df1.assign(**{
        '{}_lag_{}'.format(col, l): grid_df1.groupby(['id'])[col].transform(lambda x: x.shift(l))
        for l in LAG_DAYS
        for col in [TARGET]
    })

# Minify lag columns
for col in list(grid_df1):
    if 'lag' in col:
        grid_df1[col] = grid_df1[col].astype(np.float16)

print('%0.2f min: Lags' % ((time.time() - start_time) / 60))


# Rollings
# with 28 day shift
start_time = time.time()
print('Create rolling aggs')



#for i in [7,14,30,60,180]:
#for i in [7,14,35,63,168]:
for i in [7,14,30]:
    print('Rolling period:', i)
    grid_df1['rolling_mean_'+str(i)] = grid_df1.groupby(['id'])[TARGET].transform(lambda x: x.shift(SHIFT_DAY).rolling(i).mean()).astype(np.float16)
    grid_df1['rolling_std_'+str(i)]  = grid_df1.groupby(['id'])[TARGET].transform(lambda x: x.shift(SHIFT_DAY).rolling(i).std()).astype(np.float16)


    
    
print('%0.2f min: Lags' % ((time.time() - start_time) / 60))


# In[ ]:


grid_df1=grid_df1.drop(['demand'],axis=1)


# In[ ]:


grid_df1.info(50)


# In[ ]:


########################### Merge prices and save part 2
#################################################################################
print('Merge prices and save part 2')
# Merge Prices
grid_df = grid_df.merge(grid_df1, on=['id','d'], how='left')
#keep_columns = [col for col in list(sales) if col not in original_columns]
#grid_df = grid_df[MAIN_INDEX+keep_columns]
#grid_df = reduce_mem_usage(grid_df)

# Safe part 2
#grid_df.to_pickle('State_Item_1.pkl')
print('Size:', grid_df.shape)

# We don't need prices_df anymore
del grid_df1
grid_df.head()

# We can remove new columns
# or just load part_1
#grid_df = pd.read_pickle('grid_part_1.pkl')


# In[ ]:


grid_df.info()


# In[ ]:


print(END_TRAIN)
print('-'*80)
print(START_TRAIN)
print('-'*80)


# In[ ]:


nb = grid_df['d'].values
#MAX_LAG = max(LAGS)
tr_mask = np.logical_and(nb>START + 42, nb<=1913)
val_mask = np.logical_and(nb>1913, nb<=1941)
te_mask = np.logical_and(nb>1941, nb<=1969)


# In[ ]:


grid_df.info()


# In[ ]:


features = [col for col in list(grid_df) if col not in remove_features]


# In[ ]:


grid_df = grid_df[['id','d',TARGET]+features]
grid_df = grid_df[grid_df['d']>=START_TRAIN].reset_index(drop=True)
grid_df.info()


# In[ ]:


#scale = sales['scale'].values
ids = grid_df['id'].values


# In[ ]:


ids = grid_df['id'].values
ids


# In[ ]:


grid_df.info()


# In[ ]:


#sv = scale[val_mask]
#se = scale[te_mask]
ids = ids[te_mask]
ids = ids.reshape((-1, 28))


# **Creating dataframe**

# In[ ]:


train_df=grid_df.loc[grid_df['d'] > 100]
train_df=train_df.loc[grid_df['d'] <= 1941]

validation_df=grid_df.loc[grid_df['d'] > 1913]
validation_df=validation_df.loc[grid_df['d'] <= 1941]

evaluation_df=grid_df.loc[grid_df['d'] > 1941]
evaluation_df=evaluation_df.loc[grid_df['d'] <= 1969]

X_train=train_df.drop(['demand','id'],axis=1)
y_train=train_df[['demand']]

X_validation=validation_df.drop(['demand','id'],axis=1)
y_validation=validation_df[['demand']]

X_evaluation=evaluation_df.drop(['demand','id'],axis=1)
y_evaluation=evaluation_df[['demand']]
#X_train=X_train.astype(int)
#X_validation=X_validation.astype(int)
X_train.info()


# In[ ]:


X_train.isnull().sum()


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
# Set lower and upper quantile
Quantile1 = 0.005
Quantile2 = 0.025
Quantile3 = 0.165
Quantile4 = 0.250
Quantile5 = 0.500
Quantile6 = 0.750
Quantile7 = 0.835
Quantile8 = 0.975
Quantile9 = 0.995

# Each model has to be separate
model_1 =GradientBoostingRegressor(loss="quantile",learning_rate=.046,n_estimators=100,subsample=0.5,criterion='mse',min_samples_split=5,min_samples_leaf=9,random_state=5,max_depth=9,max_features=3,                 
                                        alpha=Quantile1)
model_2 = GradientBoostingRegressor(loss="quantile",learning_rate=.046,n_estimators=100,subsample=0.5,criterion='mse',min_samples_split=5,min_samples_leaf=9,random_state=5,max_depth=9,max_features=3,                 
                                        alpha=Quantile2)
model_3 = GradientBoostingRegressor(loss="quantile",learning_rate=.046,n_estimators=100,subsample=0.5,criterion='mse',min_samples_split=5,min_samples_leaf=9,random_state=5,max_depth=9,max_features=3,                 
                                        alpha=Quantile3)


model_4 = GradientBoostingRegressor(loss="quantile",learning_rate=.046,n_estimators=100,subsample=0.5,criterion='mse',min_samples_split=5,min_samples_leaf=9,random_state=5,max_depth=9,max_features=3,                 
                                        alpha=Quantile4)
model_5 = GradientBoostingRegressor(loss="quantile",learning_rate=.046,n_estimators=100,subsample=0.5,criterion='mse',min_samples_split=5,min_samples_leaf=9,random_state=5,max_depth=9,max_features=3,                 
                                        alpha=Quantile5)
model_6 = GradientBoostingRegressor(loss="quantile",learning_rate=.046,n_estimators=100,subsample=0.5,criterion='mse',min_samples_split=5,min_samples_leaf=9,random_state=5,max_depth=9,max_features=3,                 
                                        alpha=Quantile6)


model_7 = GradientBoostingRegressor(loss="quantile",learning_rate=.046,n_estimators=100,subsample=0.5,criterion='mse',min_samples_split=5,min_samples_leaf=9,random_state=5,max_depth=9,max_features=3,                 
                                        alpha=Quantile7)
model_8 =GradientBoostingRegressor(loss="quantile",learning_rate=.046,n_estimators=100,subsample=0.5,criterion='mse',min_samples_split=5,min_samples_leaf=9,random_state=5,max_depth=9,max_features=3,                 
                                        alpha=Quantile8)
model_9 = GradientBoostingRegressor(loss="quantile",learning_rate=.046,n_estimators=100,subsample=0.5,criterion='mse',min_samples_split=5,min_samples_leaf=9,random_state=5,max_depth=9,max_features=3,                 
                                        alpha=Quantile9)

# The mid model will use the default loss
#mid_model = GradientBoostingRegressor(loss="ls")
#upper_model = GradientBoostingRegressor(loss="quantile",
#                                        alpha=UPPER_ALPHA)


# In[ ]:


# Fit models
model_1.fit(X_train, y_train)
model_2.fit(X_train, y_train)
model_3.fit(X_train, y_train)
model_4.fit(X_train, y_train)
model_5.fit(X_train, y_train)
model_6.fit(X_train, y_train)
model_7.fit(X_train, y_train)
model_8.fit(X_train, y_train)
model_9.fit(X_train, y_train)
# Record actual values on validation data
validation = pd.DataFrame(y_validation)
validation['0.005'] = model_1.predict(X_validation)
validation['0.025'] = model_2.predict(X_validation)
validation['0.165'] = model_3.predict(X_validation)
validation['0.250'] = model_4.predict(X_validation)
validation['0.500'] = model_5.predict(X_validation)
validation['0.750'] = model_6.predict(X_validation)
validation['0.835'] = model_7.predict(X_validation)
validation['0.975'] = model_8.predict(X_validation)
validation['0.995'] = model_9.predict(X_validation)

# Record actual values on evaluation data
evaluation = pd.DataFrame(y_evaluation)
evaluation['0.005'] = model_1.predict(X_evaluation)
evaluation['0.025'] = model_2.predict(X_evaluation)
evaluation['0.165'] = model_3.predict(X_evaluation)
evaluation['0.250'] = model_4.predict(X_evaluation)
evaluation['0.500'] = model_5.predict(X_evaluation)
evaluation['0.750'] = model_6.predict(X_evaluation)
evaluation['0.835'] = model_7.predict(X_evaluation)
evaluation['0.975'] = model_8.predict(X_evaluation)
evaluation['0.995'] = model_9.predict(X_evaluation)


# In[ ]:


print("Accuracy score (training): {0:.3f}".format(model_8.score(X_train, y_train)))


# In[ ]:


#print(model_1.score(X_validation,model_1.predict(X_validation)))


# In[ ]:


validation


# In[ ]:


validation1=validation.drop(['demand'],axis=1)


# In[ ]:


pv=validation1.values


# In[ ]:


evaluation1=evaluation.drop(['demand'],axis=1)


# In[ ]:


pe=evaluation1.values


# In[ ]:


pv


# In[ ]:


names = [f"F{i+1}" for i in range(28)]


# In[ ]:


pv = pv.reshape((-1, 28, 9))
pe = pe.reshape((-1, 28, 9))

pv


# In[ ]:


piv = pd.DataFrame(ids[:, 0], columns=["id"])


# In[ ]:


piv.head()


# In[ ]:


QUANTILES = ["0.005", "0.025", "0.165", "0.250", "0.500", "0.750", "0.835", "0.975", "0.995"]
VALID = []
EVAL = []

for i, quantile in tqdm(enumerate(QUANTILES)):
    t1 = pd.DataFrame(pv[:,:, i], columns=names)
    t1 = piv.join(t1)
    t1["id"] = t1["id"] + f"_{quantile}_validation"
    t2 = pd.DataFrame(pe[:,:, i], columns=names)
    t2 = piv.join(t2)
    t2["id"] = t2["id"] + f"_{quantile}_evaluation"
    VALID.append(t1)
    EVAL.append(t2)


# In[ ]:


sub = pd.DataFrame()
sub = sub.append(VALID + EVAL)
del VALID, EVAL, t1, t2


# In[ ]:


sub


# In[ ]:


sub.to_csv("submission_Total.csv", index=False)


# In[ ]:





# In[ ]:




