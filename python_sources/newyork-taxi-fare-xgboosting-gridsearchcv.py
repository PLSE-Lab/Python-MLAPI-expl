#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
import pandas as pd 
import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#loading dataset
train_iop_path='/kaggle/input/new-york-city-taxi-fare-prediction/train.csv'
test_iop_path='/kaggle/input/new-york-city-taxi-fare-prediction/test.csv'
dataset_train=pd.read_csv(train_iop_path, nrows=1000000, index_col='key')
dataset_test=pd.read_csv(test_iop_path, nrows=1000000, index_col='key')


# In[ ]:



print("dataset_test old size", len(dataset_test))

dataset_test = dataset_test[dataset_test.dropoff_longitude != 0]
print("new size", len(dataset_test))
dataset_test.head()


# In[ ]:


print("old size", len(dataset_train))

dataset_train = dataset_train[dataset_train.dropoff_longitude != 0]
print("new size", len(dataset_train))


# In[ ]:


from datetime import datetime as dt
import warnings
warnings.filterwarnings("ignore")
def preparedataset(datasetname):
    datasetname['pickup_year']=0
    datasetname['pickup_month']=0
    datasetname['pickup_day']=0
    datasetname['pickup_hour']=0
  #  datasetname['pickup_minute']=0
 #   datasetname['pickup_second']=0
    datasetname['dis'] =0
    datasetname['x_dis']=0
    datasetname['y_dis']=0
    
    datasetname.head()
#print(datetime.strptime(df['pickup_datetime'][0].replace("UTC",''),"%Y-%m-%d %H:%M:%S "))

    for k in range(len(datasetname.index)):
        datetime=dt.strptime(datasetname['pickup_datetime'][k].replace("UTC",''),"%Y-%m-%d %H:%M:%S ")
        datasetname['pickup_year'][k]=datetime.year
        datasetname['pickup_month'][k]=datetime.month
        datasetname['pickup_day'][k]=datetime.day
        datasetname['pickup_hour'][k]=datetime.hour
      # datasetname['pickup_minute'][k]=datetime.minute
      # datasetname['pickup_second'][k]=datetime.second
        
    datasetname['x_dis'] = (datasetname['dropoff_longitude'] - datasetname['pickup_longitude'])  
    datasetname['y_dis'] = (datasetname['dropoff_latitude'] - datasetname['pickup_latitude']) 
    datasetname['dis'] = ((datasetname['dropoff_longitude'] - datasetname['pickup_longitude'])**2 + (datasetname['dropoff_latitude']-datasetname['pickup_latitude'])**2)**.5 
    datasetname=datasetname.drop(['pickup_datetime'],axis=1)
    datasetname=datasetname.drop(['pickup_longitude'],axis=1)
    datasetname=datasetname.drop(['dropoff_latitude'],axis=1)
    datasetname=datasetname.drop(['dropoff_longitude'],axis=1)
    datasetname=datasetname.drop(['pickup_latitude'],axis=1)

    return datasetname



# In[ ]:


df=preparedataset(dataset_train)


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.to_csv('train_prepared.csv')


# In[ ]:


test_df=preparedataset(dataset_test)


# In[ ]:


test_df.head()


# In[ ]:


test_df.tail()


# In[ ]:


test_df.to_csv('test_prepared.csv')


# In[ ]:


n_esitmators = list(range(100, 1001, 100))
print('n_esitmators', n_esitmators)
learning_rates = [x / 100 for x in range(5, 101, 5)]
print('learning_rates', learning_rates)


# In[ ]:


parameters = [{'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 
                     'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                                       0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
                    }]

# parameters = [{'n_estimators': [100, 200], 
#                      'learning_rate': [0.05, 0.1, 0.15]
#                     }]


# In[ ]:


#spliting the dataset 
y_train = df.fare_amount
X_train =df.drop('fare_amount',axis=1)


# In[ ]:


from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
gsearch = GridSearchCV(estimator=XGBRegressor(),
                       param_grid = parameters, 
                       scoring='neg_mean_absolute_error',
                       n_jobs=4,cv=3)

gsearch.fit(X_train, y_train)


# In[ ]:


gsearch.best_params_.get('n_estimators'), gsearch.best_params_.get('learning_rate')


# In[ ]:


final_model = XGBRegressor(n_estimators=gsearch.best_params_.get('n_estimators'), 
                           learning_rate=gsearch.best_params_.get('learning_rate'), 
                           n_jobs=4)


# In[ ]:


final_model.fit(X_train, y_train)


# In[ ]:


test_preds = final_model.predict(test_df)


# In[ ]:


output = pd.DataFrame({'key': test_df.index,
                      'fare_amount': test_preds})
output.to_csv('submission.csv', index=False)
print('done')

