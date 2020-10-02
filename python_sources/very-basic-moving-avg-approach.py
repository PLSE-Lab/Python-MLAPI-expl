#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import datetime

import lightgbm as lgb
import numpy as np


# In[ ]:


#RISCRIVI SCRIPT CELLA PER CELLA


# In[ ]:


lat_long = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")


# In[ ]:


lat_long = lat_long[['Province/State','Country/Region','Lat','Long']].drop_duplicates() 


# In[ ]:


lat_long.columns = ['Province_State', 'Country_Region', 'Lat', 'Long']


# In[ ]:


train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")
test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")
sub = pd.read_csv("../input/covid19-global-forecasting-week-3/submission.csv")


# In[ ]:


for cat_col in ['Province_State', 'Country_Region']:
    train[cat_col].fillna('no_value', inplace = True)
    test[cat_col].fillna('no_value', inplace = True)
    lat_long[cat_col].fillna('no_value', inplace = True)


# In[ ]:




train['place'] = train['Province_State']+'_'+train['Country_Region']
test['place'] = test['Province_State']+'_'+test['Country_Region']
lat_long['place'] = lat_long['Province_State']+'_'+lat_long['Country_Region']


# In[ ]:


train = train.sort_values(by='Date')
train['Fatalities'] = train.groupby(['place'])['Fatalities'].cummax()
train['ConfirmedCases'] = train.groupby(['place'])['ConfirmedCases'].cummax()
train = train.sort_values(by='Id')


# In[ ]:


train = pd.merge(train,lat_long[['place','Lat','Long']], on=['place'], how='left')
test = pd.merge(test,lat_long[['place','Lat','Long']], on=['place'],how='left')


# In[ ]:


train['shift_1_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(3)
train['shift_1_ft'] = train.groupby(['place'])['Fatalities'].shift(3)


# In[ ]:


train['diff_1_cc'] = (train['ConfirmedCases']-train['shift_1_cc'] )/3
train['diff_1_ft'] = (train['Fatalities']-train['shift_1_ft'] )/3


# In[ ]:


tmin = train[train['Date']==test['Date'].min()]
tmin.head()


# In[ ]:


tmax = train[train['Date']==train['Date'].max()]
tmax.head()


# In[ ]:


test1 = test[test['Date']<=train['Date'].max()]


# In[ ]:


test2 = test[test['Date']>train['Date'].max()]


# In[ ]:


test1 = pd.merge(test1, tmin[['place','diff_1_cc','diff_1_ft']], on='place')
test2 = pd.merge(test2, tmax[['place','diff_1_cc','diff_1_ft']], on='place')


# In[ ]:


test1['ConfirmedCases'] = test1['diff_1_cc']*1.2
test1['ConfirmedCases'] = test1.groupby(['place'])['ConfirmedCases'].cumsum()
test1['Fatalities'] = test1['diff_1_ft']*1.2
test1['Fatalities'] = test1.groupby(['place'])['Fatalities'].cumsum()


# In[ ]:


test2['ConfirmedCases'] = test2['diff_1_cc']
#test2['ConfirmedCases'] = test2.groupby(['place'])['ConfirmedCases'].cumsum()
test2['Fatalities'] = test2['diff_1_ft']
#test2['Fatalities'] = test2.groupby(['place'])['Fatalities'].cumsum()


# In[ ]:


test2.loc[test2['Date']==test2['Date'].min(),'ConfirmedCases'] = list(train.loc[train['Date']==test1['Date'].max(),'ConfirmedCases'])
test2.loc[test2['Date']==test2['Date'].min(),'Fatalities'] = list(train.loc[train['Date']==test1['Date'].max(),'Fatalities'])


# In[ ]:


#train.loc[train['Date']==test1['Date'].max(),'ConfirmedCases']


# In[ ]:


#test2['ConfirmedCases'] = test2['diff_1_cc']
test2['ConfirmedCases'] = test2.groupby(['place'])['ConfirmedCases'].cumsum()
#test2['Fatalities'] = test2['diff_1_ft']
test2['Fatalities'] = test2.groupby(['place'])['Fatalities'].cumsum()


# In[ ]:


test_final = test1.append(test2)


# In[ ]:


test_final[test_final['Country_Region']=='Spain']


# In[ ]:


sub = test_final[['ForecastId','Fatalities','ConfirmedCases']]


# In[ ]:


sub.to_csv('submission.csv',index=False)


# In[ ]:




