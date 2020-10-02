#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


## Importing Python libraries

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import datetime


# In[ ]:


train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


train_confm_date=train.groupby('Date')['ConfirmedCases','Fatalities'].sum()


# In[ ]:


train_confm_date.head()


# In[ ]:


plt.figure(figsize=(16,10))
train_confm_date.plot()
plt.title('Globally COnfirmed case and Fatalities')
plt.xticks(rotation=60)


# From the above graph, we can see that 12th March onward, the rate of confirm cases is increase exponialtially since other countries are also contributing to the cases.

# In[ ]:


train_confirm_country=train.groupby('Country/Region')['ConfirmedCases','Fatalities'].sum().reset_index().sort_values('ConfirmedCases',ascending=False)


# In[ ]:


train_confirm_country.head()


# In[ ]:


plt.figure(figsize=(12,6))
plt.bar(train_confirm_country['Country/Region'][:10],train_confirm_country['ConfirmedCases'][:10])
plt.bar(train_confirm_country['Country/Region'][:10],train_confirm_country['Fatalities'][:10])
plt.legend([' Blue Color: Confirmed cases and Yellow Color : Fatality'])


# We have seen from the above graph that highest no. of confirmed cases has been registered in China and followed by Italy and Iran. However , no of fatality  of China is comparatively low than Italy

# Now we will see the Fatality rate for each country

# In[ ]:


train_confirm_country['Fatality rate in %']=train_confirm_country['Fatalities']/train_confirm_country['ConfirmedCases']


# In[ ]:


train_confirm_country.sort_values('Fatality rate in %', ascending=False).head(10)


# from the above table we found that Sudan is having high Fatality rate that is 10 death oout of 15 confirmed case

# Let us see top 10 no of confirm case countries fatality rate in %

# In[ ]:


train_confirm_country.head(10)


# In[ ]:


plt.figure(figsize=(12,6))
plt.bar(train_confirm_country['Country/Region'][:10],train_confirm_country['Fatality rate in %'][:10])


# In[ ]:


df_top_10=train_confirm_country[:10]
df_top_10.head(5)


# In[ ]:


sns.barplot(y='Country/Region',x='Fatality rate in %',data=df_top_10)


# Observation: We see that the fatality rate in China is very less even though the no. of confirmed is very high. Itality is having Fatality rate of around 0.08%.

# In[ ]:


train_daily_report=train.groupby('Date').sum()

train_daily_report.head()


# In[ ]:


plt.figure(figsize=(18,10))
train_daily_report[['ConfirmedCases','Fatalities']].plot()
plt.xticks(rotation=60)


# In[ ]:


train_daily_report_china=train[train['Country/Region']=='China']
train_daily_report_china_sort=train_daily_report_china.groupby('Date')['ConfirmedCases','Fatalities'].sum()
plt.figure(figsize=(18,8))
train_daily_report_china_sort.plot()
plt.ylabel('No.of confirmed cases')
plt.legend(['China: COnfirmed cases till 2020-03-22'])
plt.xticks(rotation=60)


# In[ ]:


train_daily_report_india=train[train['Country/Region']=='India']
train_daily_report_india_sort=train_daily_report_india.groupby('Date')['ConfirmedCases','Fatalities'].sum()
plt.figure(figsize=(12,6))
train_daily_report_india_sort.plot()
plt.ylabel('No.of confirmed cases')
plt.legend(['India: COnfirmed cases till 2020-03-22'])
plt.xticks(rotation=60)


# In[ ]:


train_daily_report_italy=train[train['Country/Region']=='Italy']
train_daily_report_italy_sort=train_daily_report_italy.groupby('Date')['ConfirmedCases','Fatalities'].sum()
plt.figure(figsize=(12,6))
train_daily_report_italy_sort.plot()
plt.ylabel('No.of confirmed cases')
plt.legend(['Italy: COnfirmed cases till 2020-03-22'])
plt.xticks(rotation=60)


# In[ ]:


train_daily_report_iran=train[train['Country/Region']=='Iran']
train_daily_report_iran_sort=train_daily_report_iran.groupby('Date')['ConfirmedCases','Fatalities'].sum()
plt.figure(figsize=(12,6))
train_daily_report_iran_sort.plot()
plt.ylabel('No.of confirmed cases')
plt.legend(['Iran: COnfirmed cases till 2020-03-22'])
plt.xticks(rotation=60)


# ## Phase-2: Prediction Using Random Forest Regressor + GridSearch for optimal parameters

# In[ ]:


train.head()


# In[ ]:


test = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')
test.head()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False, cbar=False)


# In[ ]:


sns.heatmap(test.isnull(),yticklabels=False, cbar=False)


# We saw that both the cases in train and test dataset , missing entries are observed only in Province/State column

# In[ ]:


## Since we have Lat and long, we can drop province and country from the the dataset
train.drop(['Province/State','Country/Region'],axis=1,inplace=True)


# In[ ]:


test.drop(['Province/State','Country/Region'],axis=1,inplace=True)
display(train.info())
display(test.info())


# In[ ]:


train["Date"] = train["Date"].apply(lambda x: x.replace("-",""))
train['Date']=train['Date'].astype(int)
train.info()


# In[ ]:


test["Date"] = test["Date"].apply(lambda x: x.replace("-",""))
test["Date"]  = test["Date"].astype(int)
test.info()


# ## Creating train and Test dataset 

# In[ ]:


X_train=train.drop(['Id','ConfirmedCases','Fatalities'],axis=1)
y_confrm=train[['ConfirmedCases']]
y_fat=train[['Fatalities']]


# In[ ]:


X_test=test.drop('ForecastId',axis=1)
X_test.head()


# ## Applying ML RandomForestRegression

# from sklearn.ensemble import RandomForestRegressor
# rand_reg = RandomForestRegressor(n_estimators=100,max_depth=2,random_state=42)
# rand_reg.fit(X_train,y_confrm)
# 
# pred_ra1 = rand_reg.predict(X_test)
# pred_ra1 = pd.DataFrame(pred_ra1)
# pred_ra1.columns = ["ConfirmedCases_prediction"]
# pred_ra1.head()

# rand_reg.fit(X_train,y_fat)
# 
# pred_ra2 = rand_reg.predict(X_test)
# pred_ra2 = pd.DataFrame(pred_ra2)
# pred_ra2.columns = ["Fatality_prediction"]
# pred_ra2.head()

# ### Grid search for optimal parameter

# from sklearn.model_selection import GridSearchCV
# param_grid={
#     'n_estimators':[3,10,20,30],
#     'max_depth':[1,2,3,4]
#     
# }

# grid = GridSearchCV(RandomForestRegressor(),param_grid,refit=True,verbose=3)

# grid.fit(X_train,y_confrm)

# grid.best_params_

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rand_reg = RandomForestRegressor(random_state=42)
rand_reg.fit(X_train,y_confrm)

pred_grid1 = rand_reg.predict(X_test)
pred_grid1 = pd.DataFrame(pred_grid1).round()
pred_grid1.columns = ["ConfirmedCases_prediction"]
pred_grid1.head()


# In[ ]:


rand_reg.fit(X_train,y_fat)

pred_grid2 = rand_reg.predict(X_test)
pred_grid2 = pd.DataFrame(pred_grid2).round()
pred_grid2.columns = ["Fatality_prediction"]
pred_grid2.head()


# ##  Loading the sample_submission
# 

# In[ ]:


sample=pd.read_csv('../input/covid19-global-forecasting-week-1/submission.csv')


# In[ ]:


submission=sample[['ForecastId']]
submission.head()


# final_sub=pd.concat([submission,pred_grid1,pred_grid2],axis=1)
# final_sub.head()

# final_sub.info()

# final_sub.columns=[['ForecastId','ConfirmedCases', 'Fatalities']]
# final_sub.head()

# final_sub.to_csv("submission.csv",index=False)

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train,y_confrm)


# In[ ]:


y_tree_conf=tree_reg.predict(X_test)
y_tree_conf=pd.DataFrame(y_tree_conf)
y_tree_conf.columns=['Confrmed_prediction']
y_tree_conf.head()


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train,y_fat)

y_tree_fat=tree_reg.predict(X_test)
y_tree_fat=pd.DataFrame(y_tree_fat).round()
y_tree_fat.columns=['fatality_prediction']
y_tree_fat.head()


# In[ ]:


final_sub_tree=pd.concat([submission,y_tree_conf,y_tree_fat],axis=1)
final_sub_tree.head()


# In[ ]:


final_sub_tree.columns=[['ForecastId','ConfirmedCases', 'Fatalities']]
final_sub_tree.head()


# In[ ]:


final_sub_tree.to_csv("submission.csv",index=False)


# In[ ]:




