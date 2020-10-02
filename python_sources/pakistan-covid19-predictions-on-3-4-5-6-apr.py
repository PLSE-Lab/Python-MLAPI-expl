#!/usr/bin/env python
# coding: utf-8

# # Pakistan COVID19 Prediction
# ## Data From [Kaggle covid19 global forcasting week3](https://www.kaggle.com/c/covid19-global-forecasting-week-3)

# In[ ]:


# IMPORTS
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().run_cell_magic('time', '', "# LOAD TRAIN DATA\ntrain_new = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')")


# In[ ]:


#New train data correction
df = train_new[train_new['Country_Region']=='Pakistan']
Pakistan_data_new = df.copy()
Pakistan_data_new= Pakistan_data_new[Pakistan_data_new.ConfirmedCases > 0.0]
Pakistan_data_new.reset_index(inplace=True)
Pakistan_data_new.drop(columns= ['index','Id','Province_State'],inplace=True)
Pakistan_data_new['ConfirmedCases'] = Pakistan_data_new['ConfirmedCases'].astype(int) 
Pakistan_data_new['Fatalities'] = Pakistan_data_new['Fatalities'].astype(int) 
Pakistan_data_new.head()


# In[ ]:


#new data month adjust
month_day_list = []
for date in Pakistan_data_new['Date']:
    month_day_list.append(date.split('2020-0')[1])

Pakistan_data_new['Month_Day'] = month_day_list
Pakistan_data_new.head()


# In[ ]:


def Calculate_Table ( X_train ):
    # CALCULATE EXPANSION TABLE
    diff_conf, conf_old = [], 0 
    diff_fat, fat_old = [], 0
    dd_conf, dc_old = [], 0
    dd_fat, df_old = [], 0
    ratios = []
    for row in X_train.values:
        diff_conf.append(row[2]-conf_old)
        conf_old = row[2]
        diff_fat.append(row[3]-fat_old)
        fat_old = row[3]
        dd_conf.append(diff_conf[-1]-dc_old)
        dc_old = diff_conf[-1]
        dd_fat.append(diff_fat[-1]-df_old)
        df_old = diff_fat[-1]
        ratios.append(fat_old / conf_old)
        ratio = fat_old / conf_old
        
    return diff_conf, conf_old, diff_fat, fat_old, dd_conf, dc_old, dd_fat, df_old, ratios, ratio


# In[ ]:


def populate_df_features(X_train,diff_conf, diff_fat, dd_conf, dd_fat, ratios):    
    # POPULATE DATAFRAME FEATURES
    pd.options.mode.chained_assignment = None  # default='warn'
    X_train['diff_confirmed'] = diff_conf
    X_train['diff_fatalities'] = diff_fat
    X_train['dd_confirmed'] = dd_conf
    X_train['dd_fatalities'] = dd_fat
    X_train['ratios'] = ratios
    return X_train


# In[ ]:


def fill_nan ( variable):
    if math.isnan(variable):
        return 0
    else:
        return variable


# In[ ]:


def Cal_Series_Avg(X_train,ratio):
    # CALCULATE SERIES AVERAGES
    d_c = fill_nan( X_train.diff_confirmed[X_train.diff_confirmed != 0].mean() )
    dd_c = fill_nan( X_train.dd_confirmed[X_train.dd_confirmed != 0].mean() )
    d_f = fill_nan( X_train.diff_fatalities[X_train.diff_fatalities != 0].mean() )
    dd_f = fill_nan( X_train.dd_fatalities[X_train.dd_fatalities != 0].mean() )
    rate = fill_nan( X_train.ratios[X_train.ratios != 0].mean() )
    rate = max(rate,ratio)
    return d_c, dd_c, d_f, dd_f, rate


# In[ ]:


def apply_taylor(train, d_c, dd_c, d_f, dd_f, rate):
    # ITERATE TAYLOR SERIES
    pred_c, pred_f = [],[]
    for i in range(1, 34):
        pred_c.append(int( ( train.ConfirmedCases[len(train)-1] + d_c*i + 0.5*dd_c*(i**2)) ) )
        pred_f.append(pred_c[-1]*rate )
    return pred_c, pred_f


# In[ ]:


def Prediction(Pakistan_data_new):
    diff_conf, conf_old, diff_fat, fat_old, dd_conf, dc_old, dd_fat, df_old, ratios, ratio                                                = Calculate_Table(Pakistan_data_new)

    Pakistan_data_new = populate_df_features(Pakistan_data_new,diff_conf, diff_fat, dd_conf, dd_fat, ratios)

    d_c, dd_c, d_f, dd_f, rate = Cal_Series_Avg(Pakistan_data_new, ratio)
    
    pc_new, pf_new = apply_taylor(Pakistan_data_new, d_c, dd_c, d_f, dd_f, rate)
    
    return pc_new, pf_new


# In[ ]:


pc_03, pf_03 = Prediction(Pakistan_data_new.iloc[:-3,:])
pc_04, pf_04 = Prediction(Pakistan_data_new.iloc[:-2,:])
pc_05, pf_05 = Prediction(Pakistan_data_new.iloc[:-1,:])
pc_06, pf_06 = Prediction(Pakistan_data_new)


# In[ ]:


print(list(map(len,[pc_03, pf_03,pc_04, pf_04,pc_05, pf_05,pc_06, pf_06])))


# In[ ]:


Pakistan_data_new


# In[ ]:


Pakistan_data_new.shape


# In[ ]:


def date_format(dls):
    for i,d in enumerate(dls):
        dls[i] = d.replace("3-","Mar-")
        dls[i] = dls[i].replace("4-","Apr-")
        dls[i] = dls[i].replace("5-","May-")
    return dls


# In[ ]:


dates = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')
dates.drop(columns= ['ForecastId','Province_State','Country_Region'],inplace=True)
dates = dates.iloc[10:43,:]
start_cut= 12
pd_list = []
for date in dates.Date:
    pd_list.append(date.split('2020-0')[1])

Date_list = list(Pakistan_data_new.Month_Day[start_cut:-2].copy())
Date_list.extend(pd_list)


# In[ ]:


plt.figure(figsize=(23,6))
plt.xticks(rotation = 45)

Date_list = date_format(Date_list)

cc_03 = list(Pakistan_data_new.ConfirmedCases[start_cut:-3].copy())
cc_03.extend(pc_03[:-2])
cc_04 = list(Pakistan_data_new.ConfirmedCases[start_cut:-2].copy())
cc_04.extend(pc_04[:-2])
cc_05 = list(Pakistan_data_new.ConfirmedCases[start_cut:-1].copy())
cc_05.extend(pc_05[:-2])
cc_06 = list(Pakistan_data_new.ConfirmedCases[start_cut:].copy())
cc_06.extend(pc_06[:-2])
cc = Pakistan_data_new.ConfirmedCases[start_cut:].reset_index().drop('index',axis=1)

plt.plot(Date_list,cc_06,'r',linestyle='-.',label='Prediction 06-APR')
plt.plot(Date_list[:-1],cc_05,'g',linestyle=':',label='Prediction 05-APR')
plt.plot(Date_list[:-2],cc_04,'y',linestyle='dashed',label='Prediction 04-APR')
plt.plot(Date_list[:-3],cc_03,'m',linestyle='-.',label='Prediction 03-APR')
plt.plot(cc,'b',label='Confirmed')

plt.xlabel("Date",fontdict={'fontsize': 18})
plt.ylabel("Total Confirmed Cases",fontdict={'fontsize': 18})
plt.legend(fontsize= 18)
plt.title('Pakistan COVID19 Confirmed Cases',fontdict={'fontsize': 28})

plt.grid()
plt.show()


# In[ ]:


plt.figure(figsize=(23,6))
plt.xticks(rotation = 45)

cf_03 = list(Pakistan_data_new.Fatalities[start_cut:-3].copy())
cf_03.extend(pf_03[:-2])
cf_04 = list(Pakistan_data_new.Fatalities[start_cut:-2].copy())
cf_04.extend(pf_04[:-2])
cf_05 = list(Pakistan_data_new.Fatalities[start_cut:-1].copy())
cf_05.extend(pf_05[:-2])
cf_06 = list(Pakistan_data_new.Fatalities[start_cut:].copy())
cf_06.extend(pf_06[:-2])
cf = Pakistan_data_new.Fatalities[start_cut:].reset_index().drop('index',axis=1)

plt.plot(Date_list,cf_06,'r',linestyle='-.',label='Prediction 06-APR')
plt.plot(Date_list[:-1],cf_05,'g',linestyle=':',label='Prediction 05-APR')
plt.plot(Date_list[:-2],cf_04,'y',linestyle='dashed',label='Prediction 04-APR')
plt.plot(Date_list[:-3],cf_03,'m',linestyle='-.',label='Prediction 03-APR')
plt.plot(cf,'b',label='Fatalities')

plt.xlabel("Date",fontdict={'fontsize': 18})
plt.ylabel("Total Fatalities Cases",fontdict={'fontsize': 18})
plt.legend(fontsize= 18)
plt.title('Pakistan COVID19 Fatalities',fontdict={'fontsize': 28})

plt.grid()
plt.show()


# In[ ]:




