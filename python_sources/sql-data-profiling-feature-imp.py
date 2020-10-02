#!/usr/bin/env python
# coding: utf-8

# 56 features - Data Preparation done through SQL.
# I am a newbie to Python and Machine Learning.
# Hence used SQL for data preparation.Suggestions are welcome
# 
# Sample Scripts:
# 
# *create table air_train as 
# select a.air_store_id,d.air_store_id as hpg_store_id,a.visit_date,a.visitors,b.day_of_week,b.holiday_flg,c.total_reserve_visitors 
# as total_air_reservations,e.total_hpg_reserve_visitors as total_hpg_reservations,
# c.total_reserve_visitors+ e.total_hpg_reserve_visitors as total_reservations,f.air_genre_name,f.air_area_name,f.latitude as air_latitude,f.longitude as air_longitude,g.hpg_genre_name,g.hpg_area_name,g.latitude as hpg_latitude,g.longitude as hpg_longitude from air_visit_data a left join date_info b on a.visit_date=b.calendar_date left join air_reserve_grouped c on a.air_store_id=c.air_store_id and trunc(b.calendar_date)=c.trunc left join store_id_relation d on a.air_store_id=d.hpg_store_id
# left join hpg_reserve_grouped e on d.air_store_id=e.hpg_store_id and trunc(b.calendar_date)=e.visit_datetime left  join air_store_info f on a.air_store_id=f.air_store_id left  join hpg_store_info g on e.hpg_store_id=g.hpg_store_id;*
#                                 
# *create table air_test as 
# select a.air_store_id,d.air_store_id as hpg_store_id,a.visit_date,0 as visitors,b.day_of_week,b.holiday_flg,
# f.air_genre_name,f.air_area_name,f.latitude as air_latitude,f.longitude as air_longitude,g.hpg_genre_name,
# g.hpg_area_name,g.latitude as hpg_latitude,g.longitude as hpg_longitude from sample_submission a left join date_info b on a.visit_date=trunc(b.calendar_date) left join store_id_relation d on a.air_store_id=d.hpg_store_id left  join air_store_info f on a.air_store_id=f.air_store_id left  join hpg_store_info g on d.air_store_id=g.hpg_store_id;*
# 
# Other columns were added to these base table.
# 

# In[ ]:


#56 features - Data Preparation done through SQL
#I am a newbie to Python and Machine Learning. Hence used SQL for data preparation.
#Suggestions are welcome

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from subprocess import check_output
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from xgboost import plot_importance
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
print(check_output(["ls", "../input/"]).decode("utf8"))


# In[ ]:


train_prepared = pd.read_csv('../input/agg-data-2/air_train_agg8000.csv')
test_prepared = pd.read_csv('../input/agg-data-3/air_test_14000.csv')


# In[ ]:


train_prepared.head()


# In[ ]:


test_prepared.head()


# In[ ]:


lbl = preprocessing.LabelEncoder()
train_prepared['air_genre_name'] = lbl.fit_transform(train_prepared['air_genre_name'])
train_prepared['air_area_name'] = lbl.fit_transform(train_prepared['air_area_name'])
train_prepared['holiday_flg']= lbl.fit_transform(train_prepared['holiday_flg'])
train_prepared['day_of_week']= lbl.fit_transform(train_prepared['day_of_week'])
train_prepared.head()


# In[ ]:


train_x = train_prepared.drop(['air_store_id', 'visit_date', 'visitors','total_air_reservations', 'total_hpg_reservations', 'total_reservations'], axis=1)


# In[ ]:


train_y = np.log1p(train_prepared['visitors'].values)


# In[ ]:


test_prepared['air_genre_name'] = lbl.fit_transform(test_prepared['air_genre_name'])
test_prepared['air_area_name'] = lbl.fit_transform(test_prepared['air_area_name'])
test_prepared['holiday_flg']= lbl.fit_transform(test_prepared['holiday_flg'])
test_prepared['day_of_week']= lbl.fit_transform(test_prepared['day_of_week'])
train_prepared.head()


# In[ ]:


test_prepared['id'] = test_prepared[['air_store_id', 'visit_date']].apply(lambda x: '_'.join(x.astype(str)), axis=1)


# In[ ]:


test_x = test_prepared.drop([ 'id','air_store_id', 'visit_date', 'visitors'], axis=1)


# In[ ]:


train_x['hpg_store_id'] = pd.to_numeric(train_x['hpg_store_id'], errors='coerce')
train_x['hpg_genre_name'] = pd.to_numeric(train_x['hpg_genre_name'], errors='coerce')
train_x['hpg_area_name'] = pd.to_numeric(train_x['hpg_genre_name'], errors='coerce')
test_x['hpg_store_id'] = pd.to_numeric(test_x['hpg_store_id'], errors='coerce')
test_x['hpg_genre_name'] = pd.to_numeric(test_x['hpg_genre_name'], errors='coerce')
test_x['hpg_area_name'] = pd.to_numeric(test_x['hpg_area_name'], errors='coerce')


# In[ ]:


train_x['var_max_lat'] = train_x['air_latitude'].max() - train_x['air_latitude']
train_x['var_max_long'] = train_x['air_longitude'].max() - train_x['air_longitude']
test_x['var_max_lat'] = test_x['air_latitude'].max() - test_x['air_latitude']
test_x['var_max_long'] = test_x['air_longitude'].max() - test_x['air_longitude']
train_x['lon_plus_lat'] = train_x['air_longitude'] + train_x['air_latitude'] 
test_x['lon_plus_lat'] = test_x['air_longitude'] + test_x['air_latitude']


# In[ ]:


train_x = train_x.fillna(-1)
test_x = test_x.fillna(-1)


# In[ ]:


validation = 0.1
mask = np.random.rand(len(train_x)) < validation
X_train = train_x[~mask]
y_train = train_y[~mask]
X_validation = train_x[mask]
y_validation = train_y[mask]


# In[ ]:


xgb0 = xgb.XGBRegressor()
xgb0.fit(X_train, y_train)
print('done')


# In[ ]:


fig = plt.figure(figsize = (20, 14))
ax = fig.add_subplot(111)

colours = plt.cm.Set1(np.linspace(0, 1, 9))

ax = plot_importance(xgb0, height = 1, color = colours, grid = False,                      show_values = False, importance_type = 'cover', ax = ax);
for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        
ax.set_xlabel('importance score', size = 70);
ax.set_ylabel('features', size = 20);
ax.set_yticklabels(ax.get_yticklabels(), size = 10);
ax.set_title('Ordering of features by importance to the model learnt', size = 25);


# In[ ]:


train_x.head()


# In[ ]:


test_x.head()


# In[ ]:


train_x.shape


# In[ ]:


test_x.shape


# In[ ]:


boost_params = {'eval_metric': 'rmse'}
xgb0 = xgb.XGBRegressor(max_depth=10,
    learning_rate=0.01,
    n_estimators=1000,
    objective='reg:linear',
    gamma=0,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    seed=27,
    **boost_params)


xgb0.fit(train_x, train_y)
predict_y = xgb0.predict(test_x)
test_prepared['visitors'] = np.expm1(predict_y)
test_prepared[['id', 'visitors']].to_csv(
    'xgb_submission.csv', index=False, float_format='%.7f') 
print('done')

