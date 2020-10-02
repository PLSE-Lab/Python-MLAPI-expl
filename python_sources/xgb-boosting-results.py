#!/usr/bin/env python
# coding: utf-8

# Load the required libraries and data. 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
#now = datetime.datetime.now()

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
macro = pd.read_csv('../input/macro.csv')
id_test = test.id
train.sample(3)
# Any results you write to the current directory are saved as output.


# In[ ]:



y_train = train["price_doc"]
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values)) 
        x_train[c] = lbl.transform(list(x_train[c].values))
        #x_train.drop(c,axis=1,inplace=True)
        
for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values)) 
        x_test[c] = lbl.transform(list(x_test[c].values))
        #x_test.drop(c,axis=1,inplace=True)    
        
#features= ['big_market_km', 'preschool_km', 'exhibition_km', 'prom_part_5000', 'indust_part', 'museum_km', 'ice_rink_km', 'office_km', 'zd_vokzaly_avto_km', 'prom_part_3000', 'detention_facility_km', 'railroad_station_walk_km', 'nuclear_reactor_km', 'area_m', 'water_km', 'church_synagogue_km', 'hospice_morgue_km', 'cemetery_km', 'stadium_km', 'school_km', 'green_part_1000', 'green_part_500', 'catering_km', 'public_healthcare_km', 'big_road1_km', 'additional_education_km', 'preschool_quota', 'ID_metro', 'theater_km', 'park_km', 'power_transmission_line_km', 'metro_min_walk', 'big_church_km', 'green_part_1500', 'hospital_beds_raion', 'university_km', 'workplaces_km', 'big_road2_km', 'swim_pool_km', 'metro_km_avto', 'fitness_km', 'mosque_km', 'industrial_km', 'ttk_km', 'sub_area', 'metro_min_avto', 'public_transport_station_km', 'material', 'radiation_km', 'green_zone_km', 'railroad_km', 'kindergarten_km', 'state', 'num_room', 'kitch_sq', 'max_floor', 'build_year', 'floor', 'life_sq', 'full_sq']
#print(features)
#print(x_train.sample(1))
#print(y_train.sample(1))
#print(x_test.sample(1))
#print(y_test.sample(1))
#x_train=x_train[features]
#x_test=x_test[features]
#y_train=y_train[features]


# In[ ]:


xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)


# In[ ]:


cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()


# In[ ]:


num_boost_rounds = 400
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)


# In[ ]:


import operator

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()


#ceate_feature_map(features)

#importance = model.get_fscore(fmap='xgb.fmap')
#importance = sorted(importance.items(), key=operator.itemgetter(1))

#df = pd.DataFrame(importance, columns=['feature', 'fscore'])
#df['fscore'] = df['fscore'] / df['fscore'].sum()

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')
#fig, ax = plt.subplots(1, 1, figsize=(8, 13))
#xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)


# In[ ]:


y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
output.head()


# In[ ]:


output.to_csv('xgbMySubmission289.csv', index=False)


# In[ ]:


Now is submission time

