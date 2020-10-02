#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_df = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
test_df = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')


# In[ ]:


train_df.head()


# In[ ]:


train_df.shape


# In[ ]:


train_df.info()


# In[ ]:


train_df.tail()


# In[ ]:


l = train_df[['Country_Region','TargetValue']].groupby(['Country_Region'], as_index = False).sum().sort_values(by = 'TargetValue',ascending=False)
w = pd.DataFrame(l)
data1 = l.head(15)


# Plotting graph for top 15 countries

# In[ ]:


fig = px.bar(data1,x = 'Country_Region',y = 'TargetValue')
fig.show()


# In[ ]:


print(train_df[['Population','TargetValue']].groupby(['Population'], as_index = False).mean().sort_values(by = 'TargetValue',ascending=False))


# Observing trend of cases by plotting date vs cases graph

# In[ ]:


last_date = train_df.Date.max()
df_countries = train_df[train_df['Date']==last_date]
df_countries = df_countries.groupby('Country_Region', as_index=False)['TargetValue'].sum()
df_countries = df_countries.nlargest(10,'TargetValue')
df_trend = train_df.groupby(['Date','Country_Region'], as_index=False)['TargetValue'].sum()
df_trend = df_trend.merge(df_countries, on='Country_Region')
df_trend.rename(columns={'Country_Region':'Country', 'TargetValue_x':'Cases'}, inplace=True)
px.line(df_trend, x='Date', y='Cases', color='Country')


# In[ ]:


q = train_df[['Date','TargetValue']].groupby(['Date'], as_index = False).sum().sort_values(by = 'TargetValue',ascending=False)
#q = pd.DataFrame(q).head(30)
fig = px.bar(q,x = 'Date',y = 'TargetValue')
fig.show()


# In[ ]:


test_df['date_1'] = pd.to_datetime(test_df['Date'])
train_df['date_1'] = pd.to_datetime(train_df['Date'])


# Making two different columns 'date' and 'month'

# In[ ]:


test_df['month'] = 0
list1=[]
for i in test_df['date_1']:
    list1.append(i.month)
test_df['month'] = list1

train_df['month'] = 0
list1=[]
for i in train_df['date_1']:
    list1.append(i.month)
train_df['month'] = list1


# In[ ]:


test_df['date'] = 0
list1=[]
for i in test_df['date_1']:
    list1.append(i.day)
test_df['date'] = list1

train_df['date'] = 0
list1=[]
for i in train_df['date_1']:
    list1.append(i.day)
train_df['date'] = list1


# Checking correlation between features

# In[ ]:


plt.figure(figsize =(10,10))
sea.heatmap(train_df.corr(),annot=True)


# Dropping 'Date' coulumn because i have created new month and date column

# In[ ]:


train_df = train_df.drop(['Date'],axis=1)
test_df = test_df.drop(['Date'],axis=1)


# In[ ]:


train_df = train_df.drop(['date_1'],axis=1)
test_df = test_df.drop(['date_1'],axis=1)


# Dropping 'Country' and 'Province_state' column because of incompleteness

# In[ ]:


train_df = train_df.drop(['County'],axis=1)
test_df = test_df.drop(['County'],axis=1)


# In[ ]:


train_df = train_df.drop(['Province_State'],axis=1)
test_df = test_df.drop(['Province_State'],axis=1)


# converting 'ConfirmedCases' to 0 and 'Fatalities' to 1 

# In[ ]:


target_dict = {'ConfirmedCases':0,'Fatalities':1}
combine = [train_df,test_df]
for dataset in combine:
    dataset['Target'] = dataset['Target'].map(target_dict).astype(int)


# Converting country names to numeric value 

# In[ ]:


combine = [train_df,test_df]
country = train_df['Country_Region'].unique()
len(country)
num = [item for item in range(1,188)]
country_num = dict(zip(country,num))
for dataset in combine:
    dataset['Country_Region'] = dataset['Country_Region'].map(country_num).astype(int)


# In[ ]:


train_df = train_df.drop(['Id'],axis=1)


# In[ ]:


test_df = test_df.drop(['ForecastId'],axis=1)


# In[ ]:


test_df.head()


# In[ ]:


train_df.head()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


x = train_df.drop(['TargetValue'],axis=1)
y = train_df['TargetValue']


# In[ ]:


train_x,validate_x,train_y,validate_y = train_test_split(x,y,test_size=0.15,random_state=7)
train_x,train_y=x,y


# In[ ]:


reg = RandomForestRegressor(n_estimators=500,n_jobs=-1,verbose=1)


# In[ ]:


reg.fit(train_x,train_y)


# In[ ]:


pred_y = reg.predict(validate_x)


# In[ ]:


from sklearn.metrics import r2_score
print(r2_score(validate_y,pred_y))


# In[ ]:


test_df.head()


# In[ ]:





# In[ ]:


my_prediction = reg.predict(test_df)


# In[ ]:


my_prediction


# In[ ]:


train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
test =pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')
sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/submission.csv')


# In[ ]:


test=test.merge(train[['County','Province_State','Country_Region','Date','Target','TargetValue']],on=['County','Province_State','Country_Region','Date','Target'],how="left")


# In[ ]:


output = pd.DataFrame({'Id': test.ForecastId  , 'TargetValue': my_prediction})


# In[ ]:


a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()
b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()
c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()


# In[ ]:


a.columns=['Id','q0.05']
a=pd.concat([a,test['TargetValue']],1)
a['q0.05']=np.where(a['TargetValue'].notna(),a['TargetValue'],a['q0.05'])
a.drop(['TargetValue'],1,inplace=True)
b.columns=['Id','q0.5']
b=pd.concat([b,test['TargetValue']],1)
b['q0.5']=np.where(b['TargetValue'].notna(),b['TargetValue'],b['q0.5'])
b.drop(['TargetValue'],1,inplace=True)
c.columns=['Id','q0.95']
c=pd.concat([c,test['TargetValue']],1)
c['q0.95']=np.where(c['TargetValue'].notna(),c['TargetValue'],c['q0.95'])
c.drop(['TargetValue'],1,inplace=True)
# 	Id	TargetValue
# 0	1	140.870
# 1	2	5.352
# 2	3	132.242
# 3	4	2.584
# 4	5	126.220


# In[ ]:


# a.columns=['Id','q0.05']
# b.columns=['Id','q0.5']
# c.columns=['Id','q0.95']
a=pd.concat([a,b['q0.5'],c['q0.95']],1)
a['q0.05']=a['q0.05']
a['q0.5']=a['q0.5']
a['q0.95']=a['q0.95']


# In[ ]:


sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
sub['variable']=sub['variable'].str.replace("q","", regex=False)
sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']
sub['TargetValue']=sub['value']
sub=sub[['ForecastId_Quantile','TargetValue']]
sub.reset_index(drop=True,inplace=True)
sub.to_csv("submission.csv",index=False)
sub.head()


# In[ ]:


sub.info()


# In[ ]:




