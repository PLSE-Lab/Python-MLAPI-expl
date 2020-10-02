#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import pandas as pd
import numpy as np
import pandas_profiling
import datetime
from pathlib import Path

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.set_config_file(offline=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


root = Path('../input/great-energy-predictor-shootout-i')

df_submit = pd.read_csv(root/'sample_submission.csv')
df_submit


# In[ ]:


df_test_A = pd.read_csv(root/'Atest.dat',delim_whitespace=True)
df_test_A['Datetime'] = pd.to_datetime((df_test_A['YEAR']+1900).astype('str')                                        +'-'+(df_test_A['MONTH']).astype('str')                                        +'-'+(df_test_A['DAY']).astype('str')                                        +' '+(df_test_A['HOUR']/100).astype('int').astype('str')                                        +':00')
df_test_A.set_index('Datetime', inplace=True)
df_test_A = df_test_A.drop(['MONTH','DAY','YEAR','HOUR'],axis=1)
df_test_A['hourOfDay'] = df_test_A.index.hour
df_test_A['dayOfWeek'] = df_test_A.index.dayofweek


# In[ ]:


df_test_B = pd.read_csv(root/'Btest.dat',header=None,delim_whitespace=True)                        .rename(columns={0:'Date',1:'HorizRad',2:'SE_Rad',3:'S_Rad',4:'SW_Rad'})
df_test_B.set_index('Date', inplace=True)
df_test_B.reset_index(inplace=True)
df_test_B['Day'] = np.floor(df_test_B['Date'])
df_test_B['Time'] = df_test_B['Date'] - df_test_B['Day']
df_test_B.drop('Date', axis=1, inplace=True)


# In[ ]:


df_train_A = pd.read_csv(root/'Atrain.dat',delim_whitespace=True)
df_train_A['Datetime'] = pd.to_datetime((df_train_A['YEAR']+1900).astype('str')                                        +'-'+(df_train_A['MONTH']).astype('str')                                        +'-'+(df_train_A['DAY']).astype('str')                                        +' '+(df_train_A['HOUR']/100).astype('int').astype('str')                                        +':00')
df_train_A.set_index('Datetime', inplace=True)
df_train_A = df_train_A.drop(['MONTH','DAY','YEAR','HOUR'],axis=1)
df_train_A['hourOfDay'] = df_train_A.index.hour
df_train_A['dayOfWeek'] = df_train_A.index.dayofweek
df_train_A


# In[ ]:


df_train_B = pd.read_csv(root/'Btrain.dat',header=None,delim_whitespace=True)                        .rename(columns={0:'Date',1:'HorizRad',2:'SE_Rad',3:'S_Rad',4:'SW_Rad',5:'Beam_Rad'})
df_train_B.set_index('Date', inplace=True)
df_train_B.sort_index(inplace=True)
df_train_B.reset_index(inplace=True)
df_train_B['Day'] = np.floor(df_train_B['Date'])
df_train_B['Time'] = df_train_B['Date'] - df_train_B['Day']
df_train_B.drop('Date', axis=1, inplace=True)

df_train_B


# In[ ]:


df_weather_merged = df_train_A[['TEMP','SOLAR']].append(df_test_A[['TEMP','SOLAR']])
df_weather_merged = df_weather_merged.sort_index()  
    
df_weather_merged


# In[ ]:


alpha_data = {}
MAPE_data = {}
RSQUARED_data = {}
NMBE_data = {}
CVRSME_data = {}   

data_test_A = df_test_A.copy()
data_test_B = df_test_B.copy()


# In[ ]:


for name_meter in ['WBE','WBCW','WBHW']:
    data_train_A = df_train_A.copy()
    data_train_A = data_train_A.merge(df_weather_merged, left_index=True, right_index=True)
    
    feat_train_A = data_train_A.drop(['WBE','WBCW','WBHW'],axis=1).copy()
    label_train_A = data_train_A[name_meter].copy()

    feat_test_A = df_test_A.copy()    
    feat_test_A = feat_test_A.merge(df_weather_merged, left_index=True, right_index=True)
    
    print('Power Meter: '+name_meter)
    
    LGB_model = LGBMRegressor()
    data_train_A[name_meter + '_pred'] = cross_val_predict(LGB_model, feat_train_A, label_train_A, cv=5)
    
    LGB_model.fit(feat_train_A, label_train_A)
    data_test_A[name_meter + '_pred'] = LGB_model.predict(feat_test_A)
     
    errors = abs(data_train_A[name_meter + '_pred'] - label_train_A)
    # Print out the mean absolute error (mae)
    errors_mean = round(np.mean(errors), 2)   

    # Calculate mean absolute percentage error (MAPE) and add to list
    MAPE = 100 * np.mean((errors / label_train_A))
    NMBE = 100 * (sum(label_train_A - data_train_A[name_meter + '_pred']) / (pd.Series(label_train_A).count() * np.mean(label_train_A)))
    CVRSME = 100 * ((sum((label_train_A - data_train_A[name_meter + '_pred'])**2) / (pd.Series(label_train_A).count()-1))**(0.5)) / np.mean(label_train_A)
    RSQUARED = r2_score(label_train_A, data_train_A[name_meter + '_pred'])

    print("MAPE: "+str(round(MAPE,2)))
    print("NMBE: "+str(round(NMBE,2)))
    print("CVRSME: "+str(round(CVRSME,2)))
    print("R SQUARED: "+str(round(RSQUARED,2)))   

    data_train_A[[name_meter,name_meter + '_pred']].iplot(kind='scatter', filename='cufflinks/cf-simple-line')    
    
    print('-----------------------------------------------------------')


# In[ ]:


name_meter = 'Beam_Rad'

data_train_B = df_train_B.copy()
feat_train_B = data_train_B.drop('Beam_Rad',axis=1).copy()
label_train_B = data_train_B[name_meter].copy()

feat_test_B = df_test_B.copy()    

print('Power Meter: '+name_meter)

LGB_model = LGBMRegressor()
data_train_B[name_meter + '_pred'] = cross_val_predict(LGB_model, feat_train_B, label_train_B, cv=5)

LGB_model.fit(feat_train_B, label_train_B)
data_test_B[name_meter + '_pred'] = LGB_model.predict(feat_test_B)

errors = abs(data_train_B[name_meter + '_pred'] - label_train_B)
# Print out the mean absolute error (mae)
errors_mean = round(np.mean(errors), 2)   

# Calculate mean absolute percentage error (MAPE) and add to list
MAPE = 100 * np.mean((errors / label_train_B))
NMBE = 100 * (sum(label_train_B - data_train_B[name_meter + '_pred']) / (pd.Series(label_train_B).count() * np.mean(label_train_B)))
CVRSME = 100 * ((sum((label_train_B - data_train_B[name_meter + '_pred'])**2) / (pd.Series(label_train_B).count()-1))**(0.5)) / np.mean(label_train_B)
RSQUARED = r2_score(label_train_B, data_train_B[name_meter + '_pred'])

print("MAPE: "+str(round(MAPE,2)))
print("NMBE: "+str(round(NMBE,2)))
print("CVRSME: "+str(round(CVRSME,2)))
print("R SQUARED: "+str(round(RSQUARED,2)))   

data_train_B[[name_meter,name_meter + '_pred']].iplot(kind='scatter', filename='cufflinks/cf-simple-line')    

print('-----------------------------------------------------------')


# In[ ]:


df_submit.loc[df_submit['row_id'].str.endswith('WBE'), 'target'] = data_test_A[['WBE_pred']].melt()['value'].values
df_submit.loc[df_submit['row_id'].str.endswith('WBCW'), 'target'] = data_test_A[['WBCW_pred']].melt()['value'].values
df_submit.loc[df_submit['row_id'].str.endswith('WBHW'), 'target'] = data_test_A[['WBHW_pred']].melt()['value'].values
df_submit.loc[df_submit['row_id'].str.endswith('true_beam_insolation'), 'target'] = data_test_B['Beam_Rad_pred'].values

df_submit


# In[ ]:


df_submit.to_csv('submission.csv', index=False)

