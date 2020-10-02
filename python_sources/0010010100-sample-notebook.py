#!/usr/bin/env python
# coding: utf-8
This notebook is not the original one on that is worked during the competition. There may be some differences from the main notebook such as lack of additional columns, or changing prediction due to execution of deep learning model.

Please contact us for more detailed information and all bunch of additional data.
# # Libraries

# ## General Purposes

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error


# ## Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression


# ## Sequential

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop


# # Data
'ISLEM_TUTARI' : Transaction amount
'ISLEM_ADEDI' : Number of transactions related to expenditure information given in the same row
'ISLEM_TURU' : Type of transaction (2 options)
'YIL_AY' : Date given in a format YYYYMM (from 2017/11 to 2019/01)
'SEKTOR' : Sector name (23 different sectors)
'CUSTOMER' : Customer ID
# In[ ]:


# Train Data from 201711 ro 201901
data = pd.read_csv('train.csv')
data.drop("Record_Count", axis = 1, inplace = True)
data.head(3)

It is desired to predict transaction amounts in 201902 as close as we can. In the test data, there is an extra information, 'Id', which indicates the transaction in its row.
# In[ ]:


# Test Data - 201902
data_test = pd.read_csv('test.csv')
data_test.drop("Record_Count", axis = 1, inplace = True)
data_test.head(3)


# # Additional Info

# ## Inflation and Currency
Monthly Inflation and TRY=X
# In[ ]:


add_info = pd.DataFrame(np.array([[201711,1.49,3.878],[201712,0.69,3.847],[201801,1.02,3.768],[201802,0.73,3.777],[201803,0.99,3.881],
                                 [201804,1.87,4.05],[201805,1.62,4.405],[201806,2.61,4.623],[201807,0.55,4.749],[201808,2.30,5.792],
                                 [201809,6.30,6.334],[201810,2.67,5.842],[201811,-1.44,5.378],[201812,-0.40,5.297],[201901,1.06,5.364],
                                 [201902,0.16,5.260]]),columns=['YIL_AY','INFLATION','TRY=X'])
add_info['YIL_AY'] = add_info['YIL_AY'].astype(int)
add_info


# ## Customer Budget

# In[ ]:


budget = data.groupby(by=['CUSTOMER','ISLEM_TURU'])['ISLEM_TUTARI'].sum()
budget = pd.DataFrame(budget)
budget.rename(columns={'ISLEM_TUTARI':'BUDGET'},inplace=True)
budget.head(3)


# ## Customer Sector Budget

# In[ ]:


sector_budget = data.groupby(by=['CUSTOMER','ISLEM_TURU','SEKTOR'])['ISLEM_TUTARI'].mean()
sector_budget = pd.DataFrame(sector_budget)
sector_budget.rename(columns={'ISLEM_TUTARI':'SECTOR_BUDGET'},inplace=True)
sector_budget.head(3)


# ## Customer Sector Budget per Transaction

# In[ ]:


temp = data.copy()
temp = temp.merge(sector_budget,left_on=['CUSTOMER','ISLEM_TURU','SEKTOR'],
                  right_on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left')
temp['SB_PT'] = temp['SECTOR_BUDGET'] / temp['ISLEM_ADEDI']
temp.replace([np.inf, -np.inf],0,inplace=True)
sb_pt = temp[['CUSTOMER','ISLEM_TURU','SEKTOR','ISLEM_ADEDI','SB_PT']]
sb_pt.head(3)


# ## Customer Min, Max, Median Transaction Amount

# In[ ]:


temp = data.groupby(by=['CUSTOMER','ISLEM_TURU','SEKTOR'])['ISLEM_TUTARI'].min()
minmaxmedian = pd.DataFrame(temp)
minmaxmedian.rename(columns={'ISLEM_TUTARI':'MIN_TA'},inplace=True)

temp = data.groupby(by=['CUSTOMER','ISLEM_TURU','SEKTOR'])['ISLEM_TUTARI'].max()
temp = pd.DataFrame(temp)
temp.rename(columns={'ISLEM_TUTARI':'MAX_TA'},inplace=True)
minmaxmedian = minmaxmedian.merge(temp,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left')

temp = data.groupby(by=['CUSTOMER','ISLEM_TURU','SEKTOR'])['ISLEM_TUTARI'].median()
temp = pd.DataFrame(temp)
temp.rename(columns={'ISLEM_TUTARI':'MEDIAN_TA'},inplace=True)
minmaxmedian = minmaxmedian.merge(temp,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left')

minmaxmedian.head(3)


# ## Mean Customer Consumption in Sectors per Transaction

# In[ ]:


temp = data.copy()
temp['CONS_PT'] = temp['ISLEM_TUTARI'] / temp['ISLEM_ADEDI']
temp.replace([np.inf, -np.inf],0,inplace=True)
cons_pt = temp.groupby(by=['CUSTOMER','ISLEM_TURU','SEKTOR'])['CONS_PT'].mean()
cons_pt = pd.DataFrame(cons_pt)
cons_pt.head(3)


# ## Customer Expected Consumption

# In[ ]:


temp = data.copy()
temp = temp.merge(cons_pt,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left')
temp['EXP_CONS'] = temp['CONS_PT'] * temp['ISLEM_ADEDI']
exp_cons = temp.drop(columns='CONS_PT',axis=1)
exp_cons.head(3)


# ## Deviation of Customer Consumption in Sectors

# In[ ]:


temp = data.copy()
temp['DEV'] = temp['ISLEM_TUTARI'] / temp['ISLEM_ADEDI']
temp.replace([np.inf, -np.inf],0,inplace=True)
dev = temp.groupby(by=['CUSTOMER','ISLEM_TURU','SEKTOR'])['DEV'].std()
dev[dev.isnull()] = 0
dev = pd.DataFrame(dev)
dev.rename(columns={'DEV':'DEV_P'},inplace=True)
dev['DEV_N'] = -1*dev['DEV_P']
dev_temp = dev.copy()
dev = temp.merge(dev,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left')
dev.drop(columns='DEV',axis=1,inplace=True)
dev['DEV_P'] = dev['DEV_P'] * dev['ISLEM_ADEDI']
dev['DEV_N'] = dev['DEV_N'] * dev['ISLEM_ADEDI']
dev.head(3)


# ## Customer Min-Max Expected Consumption

# In[ ]:


temp = data.copy()
temp['MIN_EXP'] = temp['ISLEM_TUTARI'] / temp['ISLEM_ADEDI']
temp.replace([np.inf, -np.inf],0,inplace=True)
min_exp = temp.groupby(by=['CUSTOMER','ISLEM_TURU','SEKTOR'])['MIN_EXP'].min()
min_exp[min_exp.isnull()] = 0
min_exp = pd.DataFrame(min_exp)

temp['MAX_EXP'] = temp['ISLEM_TUTARI'] / temp['ISLEM_ADEDI']
temp.replace([np.inf, -np.inf],0,inplace=True)
max_exp = temp.groupby(by=['CUSTOMER','ISLEM_TURU','SEKTOR'])['MAX_EXP'].max()
max_exp[max_exp.isnull()] = 0
max_exp = pd.DataFrame(max_exp)

temp.drop(['MIN_EXP','MAX_EXP'],axis=1,inplace=True)

minmax_exp = temp.merge(min_exp,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left')
minmax_exp = minmax_exp.merge(max_exp,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left')

minmax_exp['MIN_EXP'] = minmax_exp['MIN_EXP'] * minmax_exp['ISLEM_ADEDI']
minmax_exp['MAX_EXP'] = minmax_exp['MAX_EXP'] * minmax_exp['ISLEM_ADEDI']

minmax_exp.head(3)


# # Plots
Some possible anomalies after prediction can be detected by examining plots.
# ## Merging Train Data and Additional Info

# In[ ]:


raw_data = data.merge(budget,on=['CUSTOMER','ISLEM_TURU'],how='left')
raw_data = raw_data.merge(sector_budget,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left')
raw_data['EXP_CONS'] = exp_cons['EXP_CONS']
raw_data['MON'] = (raw_data['YIL_AY'] % 20).astype(int)
raw_data.head(3)


# ## All Sectors

# In[ ]:


fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(16,16))

ax[0,0].scatter(raw_data['BUDGET'],raw_data['ISLEM_TUTARI'])
ax[0,0].set(title="TA vs BUDGET",ylabel="TA")

ax[0,1].scatter(raw_data['SECTOR_BUDGET'],raw_data['ISLEM_TUTARI'])
ax[0,1].set(title="TA vs SECTOR BUDGET",ylabel="TA")

ax[1,0].scatter(raw_data['EXP_CONS'],raw_data['ISLEM_TUTARI'])
ax[1,0].set(title="TA vs EXPECTED CONSUMPTION",ylabel="TA")

ax[1,1].scatter(raw_data['MON'],raw_data['ISLEM_TUTARI'])
ax[1,1].set(title="TA in MONTHS",ylabel="TA");


# ## Sector Data

# ### Sector List

# In[ ]:


data['SEKTOR'].unique()


# ### Pick a Sector

# In[ ]:


sec = 'BENZIN VE YAKIT ISTASYONLARI'


# In[ ]:


raw_sub = raw_data.loc[raw_data['SEKTOR']==sec]

fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(16,16))

ax[0,0].scatter(raw_sub['BUDGET'],raw_sub['ISLEM_TUTARI'])
ax[0,0].set(title="TA vs BUDGET",ylabel="TA")

ax[0,1].scatter(raw_sub['SECTOR_BUDGET'],raw_sub['ISLEM_TUTARI'])
ax[0,1].set(title="TA vs SECTOR BUDGET",ylabel="TA")

ax[1,0].scatter(raw_sub['EXP_CONS'],raw_sub['ISLEM_TUTARI'])
ax[1,0].set(title="TA vs EXPECTED CONSUMPTION",ylabel="TA")

ax[1,1].scatter(raw_sub['MON'],raw_sub['ISLEM_TUTARI'])
ax[1,1].set(title="TA in MONTHS",ylabel="TA");


# # Model : Linear Regression

# ## Predict 201901

# ### Raw Data

# In[ ]:


raw_data = data.copy()
raw_data['MON'] = (raw_data['YIL_AY'] % 20).astype(str) # Month
raw_data = raw_data.merge(add_info,on=['YIL_AY'],how='left') # Inflation and Currency
raw_data = raw_data.merge(budget,on=['CUSTOMER','ISLEM_TURU'],how='left') # Budget
raw_data = raw_data.merge(sector_budget,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left') # Sector Budget
raw_data['SB_PT'] = sb_pt['SB_PT'] # Sector Budget per T.
raw_data['EXP_CONS'] = exp_cons['EXP_CONS'] # Expected Consumption
raw_data = raw_data.merge(minmaxmedian,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left') # Min-Max-Median TA
raw_data['MIN_EXP'] = minmax_exp['MIN_EXP']
raw_data['MAX_EXP'] = minmax_exp['MAX_EXP']
raw_data['DEV_P'] = dev['DEV_P']
raw_data['DEV_N'] = dev['DEV_N']

raw_data.head(3)


# In[ ]:


raw_data.drop(["CUSTOMER","YIL_AY"], axis = 1, inplace = True)
raw_data = pd.get_dummies(raw_data,columns=["MON","ISLEM_TURU","SEKTOR"])
raw_data.head(3)


# ### Model and Prediction

# In[ ]:


train_y = raw_data.loc[(data['YIL_AY'] != 201901)]['ISLEM_TUTARI']
train_x = raw_data.loc[(data['YIL_AY'] != 201901)].drop("ISLEM_TUTARI", axis=1)

test_y = raw_data.loc[(data['YIL_AY'] == 201901)]['ISLEM_TUTARI']
test_x = raw_data.loc[(data['YIL_AY'] == 201901)].drop("ISLEM_TUTARI", axis=1)

model = LinearRegression()

model.fit(train_x,train_y)

predicted_y = model.predict(test_x)


# ### RMSE

# In[ ]:


rmse = np.sqrt(mean_squared_error(test_y, predicted_y))
rmse


# ## Predict 201902

# ### Raw Data

# In[ ]:


test = data_test.copy()
test['ISLEM_TUTARI'] = 0

train = data.copy()
train['ID'] = 0

train = train.reindex(columns=test.columns)
raw_data = pd.concat([train,test])
raw_data.head(3)


# In[ ]:


raw_data['MON'] = (raw_data['YIL_AY'] % 20).astype(str) # Month

raw_data = raw_data.merge(add_info,on=['YIL_AY'],how='left') # Inflation and Currency

raw_data = raw_data.merge(budget,on=['CUSTOMER','ISLEM_TURU'],how='left') # Budget
raw_data.fillna(0,inplace=True)

raw_data = raw_data.merge(sector_budget,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left') # Sector Budget
raw_data.fillna(0,inplace=True)

raw_data['SB_PT'] = raw_data['SECTOR_BUDGET'] / raw_data['ISLEM_ADEDI'] # Sector Budget per Transaction
raw_data.replace([np.inf, -np.inf],0,inplace=True)

raw_data = raw_data.merge(cons_pt,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left')
temp_cons = cons_pt.groupby(['ISLEM_TURU','SEKTOR'])["CONS_PT"].mean()
for i in temp_cons.index:
    tur = i[0]
    sec = i[1]
    raw_data.loc[(raw_data['SEKTOR']==sec)&(raw_data['ISLEM_TURU']==tur)&
                (raw_data['CONS_PT'].isnull()),'CONS_PT'] = temp_cons[i]
raw_data['EXP_CONS'] = raw_data['CONS_PT'] * raw_data['ISLEM_ADEDI'] # Expected Consumption
raw_data.drop(columns='CONS_PT',axis=1,inplace=True)

raw_data = raw_data.merge(minmaxmedian,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left')
raw_data.fillna(0,inplace=True)

raw_data = raw_data.merge(min_exp,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left')
raw_data = raw_data.merge(max_exp,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left')
min_m = min_exp.groupby(['ISLEM_TURU','SEKTOR'])["MIN_EXP"].mean()
min_m = pd.DataFrame(min_m)
max_m = max_exp.groupby(['ISLEM_TURU','SEKTOR'])["MAX_EXP"].mean()
max_m = pd.DataFrame(max_m)
for i in max_m.index:
    tur = i[0]
    sec = i[1]
    raw_data.loc[(raw_data['SEKTOR'] == sec) & (raw_data['ISLEM_TURU'] == tur) 
                 & (raw_data['MIN_EXP'].isnull()),'MIN_EXP'] = min_m.loc[i,'MIN_EXP']
    raw_data.loc[(raw_data['SEKTOR'] == sec) & (raw_data['ISLEM_TURU'] == tur) 
                 & (raw_data['MAX_EXP'].isnull()),'MAX_EXP'] = max_m.loc[i,'MAX_EXP']
raw_data['MIN_EXP'] = raw_data['MIN_EXP'] * raw_data['ISLEM_ADEDI']
raw_data['MAX_EXP'] = raw_data['MAX_EXP'] * raw_data['ISLEM_ADEDI']

raw_data = raw_data.merge(dev_temp,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left')
devp = dev_temp.groupby(['ISLEM_TURU','SEKTOR'])["DEV_P"].mean()
devp = pd.DataFrame(devp)
devn = dev_temp.groupby(['ISLEM_TURU','SEKTOR'])["DEV_N"].mean()
devn = pd.DataFrame(devn)
for i in devp.index:
    tur = i[0]
    sec = i[1]
    raw_data.loc[(raw_data['SEKTOR'] == sec) & (raw_data['ISLEM_TURU'] == tur) 
                 & (raw_data['DEV_P'].isnull()),'DEV_P'] = devp.loc[i,'DEV_P']
    raw_data.loc[(raw_data['SEKTOR'] == sec) & (raw_data['ISLEM_TURU'] == tur) 
                 & (raw_data['DEV_N'].isnull()),'DEV_N'] = devn.loc[i,'DEV_N']

raw_data['DEV_P'] = raw_data['DEV_P'] * raw_data['ISLEM_ADEDI']
raw_data['DEV_N'] = raw_data['DEV_N'] * raw_data['ISLEM_ADEDI']


# In[ ]:


raw_data.drop(["CUSTOMER"], axis = 1, inplace = True)
raw_data = pd.get_dummies(raw_data,columns=["MON","ISLEM_TURU","SEKTOR"])
raw_data.head(3)


# ### Model and Prediction

# In[ ]:


train_y = raw_data.loc[(raw_data['YIL_AY'] != 201902)]['ISLEM_TUTARI']
train_x = raw_data.loc[(raw_data['YIL_AY'] != 201902)].drop("ISLEM_TUTARI", axis=1)
train_x.drop(["YIL_AY","ID"], axis=1,inplace=True)

test_x = raw_data.loc[(raw_data['YIL_AY'] == 201902)].drop("ISLEM_TUTARI", axis=1)
ids = np.array(test_x['ID'])
test_x.drop(["YIL_AY","ID"], axis=1,inplace=True)

model = LinearRegression()

model.fit(train_x,train_y)

predicted_y = model.predict(test_x)


# ### Submission

# In[ ]:


submission_data = pd.DataFrame()
submission_data['Id'] = ids
submission_data['Predicted'] = predicted_y

submission_data['Id'] = submission_data['Id'].astype(int)
submission_data = submission_data.sort_values(by ='Id')
submission_data = submission_data.reset_index(drop=True)

submission_data.head()


# In[ ]:


submission_data.to_csv('submission_lr.csv',index=False)


# # Model : Sequential

# ## Predict 201901

# ### Raw Data

# In[ ]:


raw_data = data.copy()
raw_data['MON'] = (raw_data['YIL_AY'] % 20).astype(str) # Month
raw_data = raw_data.merge(add_info,on=['YIL_AY'],how='left') # Inflation and Currency
raw_data = raw_data.merge(budget,on=['CUSTOMER','ISLEM_TURU'],how='left') # Budget
raw_data = raw_data.merge(sector_budget,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left') # Sector Budget
raw_data['SB_PT'] = sb_pt['SB_PT'] # Sector Budget per T.
raw_data['EXP_CONS'] = exp_cons['EXP_CONS'] # Expected Consumption
raw_data = raw_data.merge(minmaxmedian,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left') # Min-Max-Median TA
raw_data['MIN_EXP'] = minmax_exp['MIN_EXP']
raw_data['MAX_EXP'] = minmax_exp['MAX_EXP']
raw_data['DEV_P'] = dev['DEV_P']
raw_data['DEV_N'] = dev['DEV_N']

raw_data.head(3)


# In[ ]:


raw_data.drop(["CUSTOMER","YIL_AY"], axis = 1, inplace = True)
raw_data = pd.get_dummies(raw_data,columns=["MON","ISLEM_TURU","SEKTOR"])
raw_data.head(3)


# ### Model and Prediction

# In[ ]:


train_y = raw_data.loc[(data['YIL_AY'] != 201901)]['ISLEM_TUTARI']
train_x = raw_data.loc[(data['YIL_AY'] != 201901)].drop("ISLEM_TUTARI", axis=1)

test_y = raw_data.loc[(data['YIL_AY'] == 201901)]['ISLEM_TUTARI']
test_x = raw_data.loc[(data['YIL_AY'] == 201901)].drop("ISLEM_TUTARI", axis=1)

model = Sequential()
model.add(Dense(1000, input_dim=51,activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(250, activation= "relu"))
model.add(Dense(100, activation= "relu"))
model.add(Dense(10,activation= "relu"))
model.add(Dense(1))
optimizer = RMSprop(0.001)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse','mae'])

model.fit(train_x,train_y)

predicted_y = model.predict(test_x)


# ### RMSE

# In[ ]:


rmse = np.sqrt(mean_squared_error(test_y, predicted_y))
rmse

With different parameters for layers of Sequential, the best RMSE value obtained for prediction of 201901 is 742.
# ## Predict 201902

# ### Raw Data

# In[ ]:


test = data_test.copy()
test['ISLEM_TUTARI'] = 0

train = data.copy()
train['ID'] = 0

train = train.reindex(columns=test.columns)
raw_data = pd.concat([train,test])
raw_data.head(3)


# In[ ]:


raw_data['MON'] = (raw_data['YIL_AY'] % 20).astype(str) # Month

raw_data = raw_data.merge(add_info,on=['YIL_AY'],how='left') # Inflation and Currency

raw_data = raw_data.merge(budget,on=['CUSTOMER','ISLEM_TURU'],how='left') # Budget
raw_data.fillna(0,inplace=True)

raw_data = raw_data.merge(sector_budget,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left') # Sector Budget
raw_data.fillna(0,inplace=True)

raw_data['SB_PT'] = raw_data['SECTOR_BUDGET'] / raw_data['ISLEM_ADEDI'] # Sector Budget per Transaction
raw_data.replace([np.inf, -np.inf],0,inplace=True)

raw_data = raw_data.merge(cons_pt,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left')
temp_cons = cons_pt.groupby(['ISLEM_TURU','SEKTOR'])["CONS_PT"].mean()
for i in temp_cons.index:
    tur = i[0]
    sec = i[1]
    raw_data.loc[(raw_data['SEKTOR']==sec)&(raw_data['ISLEM_TURU']==tur)&
                (raw_data['CONS_PT'].isnull()),'CONS_PT'] = temp_cons[i]
raw_data['EXP_CONS'] = raw_data['CONS_PT'] * raw_data['ISLEM_ADEDI'] # Expected Consumption

raw_data = raw_data.merge(minmaxmedian,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left')
raw_data.fillna(0,inplace=True)

raw_data = raw_data.merge(min_exp,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left')
raw_data = raw_data.merge(max_exp,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left')
min_m = min_exp.groupby(['ISLEM_TURU','SEKTOR'])["MIN_EXP"].mean()
min_m = pd.DataFrame(min_m)
max_m = max_exp.groupby(['ISLEM_TURU','SEKTOR'])["MAX_EXP"].mean()
max_m = pd.DataFrame(max_m)
for i in max_m.index:
    tur = i[0]
    sec = i[1]
    raw_data.loc[(raw_data['SEKTOR'] == sec) & (raw_data['ISLEM_TURU'] == tur) 
                 & (raw_data['MIN_EXP'].isnull()),'MIN_EXP'] = min_m.loc[i,'MIN_EXP']
    raw_data.loc[(raw_data['SEKTOR'] == sec) & (raw_data['ISLEM_TURU'] == tur) 
                 & (raw_data['MAX_EXP'].isnull()),'MAX_EXP'] = max_m.loc[i,'MAX_EXP']
raw_data['MIN_EXP'] = raw_data['MIN_EXP'] * raw_data['ISLEM_ADEDI']
raw_data['MAX_EXP'] = raw_data['MAX_EXP'] * raw_data['ISLEM_ADEDI']

raw_data = raw_data.merge(dev_temp,on=['CUSTOMER','ISLEM_TURU','SEKTOR'],how='left')

devp = dev_temp.groupby(['ISLEM_TURU','SEKTOR'])["DEV_P"].mean()
devp = pd.DataFrame(devp)
devn = dev_temp.groupby(['ISLEM_TURU','SEKTOR'])["DEV_N"].mean()
devn = pd.DataFrame(devn)
for i in devp.index:
    tur = i[0]
    sec = i[1]
    raw_data.loc[(raw_data['SEKTOR'] == sec) & (raw_data['ISLEM_TURU'] == tur) 
                 & (raw_data['DEV_P'].isnull()),'DEV_P'] = devp.loc[i,'DEV_P']
    raw_data.loc[(raw_data['SEKTOR'] == sec) & (raw_data['ISLEM_TURU'] == tur) 
                 & (raw_data['DEV_N'].isnull()),'DEV_N'] = devn.loc[i,'DEV_N']

raw_data['DEV_P'] = raw_data['DEV_P'] * raw_data['ISLEM_ADEDI']
raw_data['DEV_N'] = raw_data['DEV_N'] * raw_data['ISLEM_ADEDI']


# In[ ]:


raw_data.drop(["CUSTOMER"], axis = 1, inplace = True)
raw_data = pd.get_dummies(raw_data,columns=["MON","ISLEM_TURU","SEKTOR"])
raw_data.head(3)


# ### Model and Prediction

# In[ ]:


train_y = raw_data.loc[(raw_data['YIL_AY'] != 201902)]['ISLEM_TUTARI']
train_x = raw_data.loc[(raw_data['YIL_AY'] != 201902)].drop("ISLEM_TUTARI", axis=1)
train_x.drop(["YIL_AY","ID"], axis=1,inplace=True)

test_x = raw_data.loc[(raw_data['YIL_AY'] == 201902)].drop("ISLEM_TUTARI", axis=1)
ids = np.array(test_x['ID'])
test_x.drop(["YIL_AY","ID"], axis=1,inplace=True)

predicted_y = model.predict(test_x)


# ### Submission

# In[ ]:


submission_data = pd.DataFrame()
submission_data['Id'] = ids
submission_data['Predicted'] = predicted_y

submission_data['Id'] = submission_data['Id'].astype(int)
submission_data = submission_data.sort_values(by ='Id')
submission_data = submission_data.reset_index(drop=True)

submission_data.head()


# In[ ]:


submission_data.to_csv('submission_dnn.csv',index=False)

