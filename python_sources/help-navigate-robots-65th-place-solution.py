#!/usr/bin/env python
# coding: utf-8

# ![](https://s3.eu-north-1.amazonaws.com/ammar-files/kaggle-kernels/Help-Navigate-Robots+-+65th-Place+Solution/robc-1.jpg)
# 
# This is my simple solution to "CareerCon 2019 - Help Navigate Robots" competition which achieved the 65th place on the private leaderboard. 
# 
# It uses **tsfresh** to extract and select features from our time-series data. tsfresh calculates a lot of features with many parameters including the minimum value of the signal, the maximum, standard deviation, approximate entropy, skewness, number of peaks, etc. 
# 
# After that, LightGBM is used to build the model and generate predictions. The generated predictions are the probabilities of each type of surfaces for each test case. The surface with the highest probability was chosen.

# # Importing Libraries and Reading Data

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import gc
import scipy.signal as signal
import scipy.stats as stats
import time
import warnings
from tsfresh import extract_relevant_features, extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)
pd.set_option("max_columns", 200)
pd.set_option("max_rows", 200)
gc.enable()


# In[ ]:


x_train = pd.read_csv('../input/X_train.csv')
y_train = pd.read_csv('../input/y_train.csv')
x_test = pd.read_csv('../input/X_test.csv')


# In[ ]:


print(x_train.shape)
x_train.head(10)


# In[ ]:


print(y_train.shape)
y_train.head()


# In[ ]:


y_train.surface.unique()


# # Feature Extraction and Selection Using tsfresh

# In[ ]:


tsfresh_train = extract_features(x_train.drop('row_id', axis=1), column_id='series_id', 
                                 column_sort='measurement_number')
impute(tsfresh_train);


# In[ ]:


relevant_train_features = set()
for label in y_train['surface'].unique():
    y_train_binary = (y_train['surface'].values == label).astype(int)
    print('=='*20); print(y_train_binary); print('=='*20);
    X_train_filtered = select_features(tsfresh_train, y_train_binary, fdr_level=0.382)
    print('=='*20);
    print("Number of relevant features for class {}: {}/{}".format(
        label, X_train_filtered.shape[1], tsfresh_train.shape[1]))
    print('=='*20);
    relevant_train_features = relevant_train_features.union(set(X_train_filtered.columns))


# In[ ]:


tsfresh_test = extract_features(x_test.drop('row_id', axis=1), column_id='series_id', 
                                column_sort='measurement_number')
impute(tsfresh_test);


# In[ ]:


len(relevant_train_features)


# In[ ]:


tsfresh_train = tsfresh_train[list(relevant_train_features)]
tsfresh_test = tsfresh_test[list(relevant_train_features)]


# In[ ]:


print(tsfresh_train.shape)
tsfresh_test.head()


# # Moeling and Generating Final Preictions

# In[ ]:


fac_surfaces, surfaces = y_train['surface'].factorize()
train_data = lgb.Dataset(tsfresh_train, label=fac_surfaces)
params={'learning_rate': 0.1, 'objective':'multiclass', 'metric':'multi_error', 
        'num_class':9, 'verbose': 1, 'random_state':311,
        'bagging_fraction': 0.7, 'feature_fraction': 1.0}
num_round = 15000
light = lgb.train(params, train_data, num_round)
pred = light.predict(tsfresh_test)
feature_importances = light.feature_importance()
feature_names = tsfresh_test.columns.values


# In[ ]:


pred[:,0].shape
fac_surfaces, surfaces = y_train['surface'].factorize()
final_pred = pd.Series(np.argmax(pred, axis=1))
surface_dict = {}
for n, s in enumerate(surfaces):
    surface_dict[n] = s
final_pred = final_pred.map(surface_dict)


# In[ ]:


submission = pd.DataFrame({
        "series_id": list(range(3816)),
        "surface": final_pred
})

submission.to_csv('submission.csv', index=False)

