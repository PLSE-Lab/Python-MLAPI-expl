#!/usr/bin/env python
# coding: utf-8

# # import

# In[ ]:


import numpy as np
import pandas as pd
import gc
import tqdm
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import KFold
import lightgbm as lgb

LOCAL = False


# # preprocessing
# I created the data to be used this time with the following code, but this time it is running in a local environment because it takes more than 2 hours.

# In[ ]:


if LOCAL:
    train_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')
    test_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')
    building = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')
    train_weather = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')
    train_df = train_df.merge(building, on='building_id', how='left')
    train_df = train_df.merge(train_weather, on=['site_id', 'timestamp'], how='left')
    del train_weather
    test_weather = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')
    test_df = test_df.merge(building, on='building_id', how='left')
    del building
    test_df = test_df.merge(test_weather, on=['site_id', 'timestamp'], how='left')
    del test_weather
    gc.collect()

    train_df["primary_use_cate"]=float(np.nan)
    test_df["primary_use_cate"]=float(np.nan)
    train_df["time_year"]=float(np.nan)
    train_df["time_month"]=float(np.nan)
    train_df["time_day"]=float(np.nan)
    train_df["time_hour"]=float(np.nan)
    test_df["time_year"]=float(np.nan)
    test_df["time_month"]=float(np.nan)
    test_df["time_day"]=float(np.nan)
    test_df["time_hour"]=float(np.nan)

    primary_use_cate = list(set(train_df["primary_use"].values)|set(test_df["primary_use"].values))
    primary_use_label2int = {c:i for i,c in enumerate(primary_use_cate)}

    for i in tqdm.tqdm(range(len(train_df))):
        s = train_df.at[i,"timestamp"]
        train_df.at[i,"primary_use_cate"] = primary_use_label2int[train_df.at[i,"primary_use"]]
        train_df.at[i,"time_year"] = int(s[0:4])
        train_df.at[i,"time_month"] = int(s[5:7])
        train_df.at[i,"time_day"] = int(s[8:10])
        train_df.at[i,"time_hour"] = int(s[11:13])
    
    for i in tqdm.tqdm(range(len(test_df))):
        s = test_df.at[i,"timestamp"]
        test_df.at[i,"primary_use_cate"] = primary_use_label2int[test_df.at[i,"primary_use"]]
        test_df.at[i,"time_year"] = int(s[0:4])
        test_df.at[i,"time_month"] = int(s[5:7])
        test_df.at[i,"time_day"] = int(s[8:10])
        test_df.at[i,"time_hour"] = int(s[11:13])


# The elements that seem to contribute little was removed due to memory constraints.

# In[ ]:


if not LOCAL:
    train_df = pickle.load(open("/kaggle/input/ashrae-preprocessed-data/train.pickle","rb"))
    test_df = pickle.load(open("/kaggle/input/ashrae-preprocessed-data/test.pickle","rb"))


# In[ ]:


# from https://www.kaggle.com/divrikwicky/ashrae-lofo-feature-importance

y_train = np.array(train_df["meter_reading"])
del train_df['primary_use']
del train_df['meter_reading']
del train_df['year_built']
del train_df['floor_count']
del train_df['precip_depth_1_hr']
del train_df['wind_direction']
del train_df['sea_level_pressure']
del train_df['time_hour']
del train_df['timestamp']

del test_df['primary_use']
del test_df['row_id']
del test_df['year_built']
del test_df['floor_count']
del test_df['precip_depth_1_hr']
del test_df['wind_direction']
del test_df['sea_level_pressure']
del test_df['time_hour']
del test_df['timestamp']

X_train = train_df
X_test = test_df


# In[ ]:


data=[0 for i in range(200)]
data_ori=[0 for i in range(21904701)]
for p in tqdm.tqdm(y_train):
    data[int(np.log(p+1)*10)]+=1
    data_ori[int(p)]+=1


# In[ ]:


plt.plot([i for i in range(21904701)],data_ori)


# As you can see, it is difficult to predict because of the wide distribution range.

# In[ ]:


plt.plot([i for i in range(200)],data)


# So I decided to take the logarithm.

# In[ ]:


y_train = np.log(y_train+1)


# # train
# I think it's intuitive to use a regression algorithm in this competition.

# In[ ]:


gc.collect()
folds = 3
seed = 222
kf = KFold(n_splits = folds, shuffle = True, random_state=seed)
y_valid_pred = np.zeros(X_train.shape[0])
models = []

for tr_idx, val_idx in kf.split(X_train, y_train):
    tr_x, tr_y = X_train.iloc[tr_idx,:], y_train[tr_idx]
    vl_x, vl_y = X_train.iloc[val_idx,:], y_train[val_idx]
            
    print(len(tr_x),len(vl_x))
    tr_data = lgb.Dataset(tr_x, label=tr_y)
    vl_data = lgb.Dataset(vl_x, label=vl_y)  
    clf = lgb.LGBMRegressor(n_estimators=200,learning_rate=0.5,feature_fraction=0.9,
            bagging_fraction=0.9,early_stopping_rounds=50)
    clf.fit(tr_x, tr_y,
        eval_set=[(vl_x, vl_y)],
        verbose=True)
    y_valid_pred[val_idx] += clf.predict(vl_x, num_iteration=clf.best_iteration_)
    models.append(clf)
    gc.collect()


# # evaluation
# Since the minimum value is 0, it was clipped to do so.

# In[ ]:


print("valid score is",np.sqrt(sum(np.power(y_train-np.clip(y_valid_pred,0,None),2))/y_train.shape[0]))


# # make submission
# I need to do the opposite of what was done in preprocessing.

# In[ ]:


res=np.zeros(41697600,dtype=float)
for i in tqdm.tqdm(range(0,41697600,27200)):
    res[i:i+27200] = sum([np.clip(np.exp(model.predict(X_test.iloc[i:i+27200]))-1,0,None) for model in models])/folds


# In[ ]:


submission = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')
submission['meter_reading'] = res
submission.to_csv('submission.csv', index=False)
submission


# There is a way to reduce memory, and using it does not seem to require this effort to save memory for https://www.kaggle.com/hamditarek/reducing-memory-size-for-great-energy-predictor.<br>
# Please let me know if you have any opinions or advice.
